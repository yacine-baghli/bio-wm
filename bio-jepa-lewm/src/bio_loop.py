import numpy as np
import torch
from . import cl_sdk as cl

def process_spikes_to_grid(spike_matrix: np.ndarray) -> np.ndarray:
    """
    Decodes raw biological spike matrices into a continuous 64-channel latent prediction vector.
    """
    # Slice the recording zone: electrodes 68 to 131 (64 output channels)
    recording_spikes = spike_matrix[:, 68:132]  # [frame_count, 64]
    
    # Average across the temporal frame window (e.g., 5 frames) to get a continuous activation rate
    mean_activations = np.mean(recording_spikes, axis=0)  # [64]
    
    # Normalize to form a valid probability distribution (sum to 1)
    act_sum = mean_activations.sum()
    if act_sum > 0:
        mean_activations = mean_activations / act_sum
    else:
        mean_activations = np.ones_like(mean_activations) / len(mean_activations)
        
    return mean_activations

def calculate_centroid(latent_vector: np.ndarray, grid_size: int = 8) -> np.ndarray:
    """
    Computes the continuous spatial center-of-mass (centroid) of an 8x8 latent grid activation.
    Applies power-sharpening to filter out background probability noise and extract a clean peak.
    """
    eps = 1e-25
    sharpened = np.power(np.clip(latent_vector, 0.0, None), 8)
    total_mass = np.sum(sharpened)
    
    if total_mass < eps:
        return np.array([grid_size / 2.0, grid_size / 2.0])
        
    grid = (sharpened / total_mass).reshape(grid_size, grid_size)
    
    cx = 0.0
    cy = 0.0
    for r in range(grid_size):
        for c in range(grid_size):
            cx += (c + 0.5) * grid[r, c]
            cy += (r + 0.5) * grid[r, c]
            
    return np.array([cx, cy])

def generate_target_grid_numpy(pos: np.ndarray, grid_size: int = 8, sigma: float = 0.8) -> np.ndarray:
    """
    Translates continuous [X, Y] coordinates (scaled to 0..8) back into a 64-channel Gaussian peak.
    """
    x, y = pos[0], pos[1]
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r in range(grid_size):
        for c in range(grid_size):
            cell_x = c + 0.5
            cell_y = r + 0.5
            dist_sq = (cell_x - x)**2 + (cell_y - y)**2
            grid[r, c] = np.exp(-dist_sq / (2 * sigma**2))
    grid_sum = grid.sum()
    if grid_sum > 0:
        grid = grid / grid_sum
    else:
        grid = np.ones_like(grid) / float(grid_size * grid_size)
    return grid.flatten()

class BioWorldModelLoop:
    """
    Closed-loop Biological World Model execution controller.
    
    Links the environment state, the digital CNN encoder, and the biological network.
    Uses an online velocity-aware linear decoder mapping spikes directly to [X, Y, dX, dY].
    """
    def __init__(self, bnn_client, use_boundary_penalty: bool = True, use_velocity_decoding: bool = True, gain_factor: float = 1.0, lr_dec_bias_scale: float = 0.1):
        self.bnn = bnn_client
        self.use_boundary_penalty = use_boundary_penalty
        self.use_velocity_decoding = use_velocity_decoding
        self.gain_factor = gain_factor
        self.lr_dec_bias_scale = lr_dec_bias_scale
        
        # W_dec shape: [64, 4] mapping 64 electrodes to [X, Y, dX, dY]
        self.W_dec = np.zeros((64, 4), dtype=np.float32)
        self.b_dec = np.zeros(4, dtype=np.float32)
        
        # Informed initialization of W_dec position heads (columns 0 & 1)
        for j in range(64):
            col = j % 8
            row = j // 8
            self.W_dec[j, 0] = col + 0.5
            self.W_dec[j, 1] = row + 0.5

    def predict_and_learn(self, z_curr: np.ndarray, action: int, z_next: np.ndarray, y_target: np.ndarray, boundary_penalty: bool = False, lr_dec: float = 0.15) -> np.ndarray:
        """
        Predicts the next continuous state [X, Y, dX, dY] and updates biological synapses
        and the online linear decoder.
        """
        active_pos_channel = int(np.argmax(z_curr))
        
        # Stimulate sensory, motor, and boundary channels in the BNN
        bp_active = boundary_penalty and self.use_boundary_penalty
        self.bnn.stimulate(active_pos_channel, action, boundary_penalty=bp_active)
        spike_frames = self.bnn.read_frames(frame_count=5)
        
        # 1. Extract raw mean spike activations (recording zone electrodes 68 to 131)
        recording_spikes = spike_frames[:, 68:132]
        s_rate = np.mean(recording_spikes, axis=0)  # [64]
        
        # Normalize spike rate to form a stable spatial combination
        s_sum = s_rate.sum()
        s_rate_norm = s_rate / s_sum if s_sum > 0 else s_rate
        
        # 2. Decode spike activations using the velocity-aware linear decoder
        y_pred = np.dot(s_rate_norm, self.W_dec) + self.b_dec
        
        # 3. Apply Hebbian updates to BNN physical synapses using one-hot representations
        z_curr_oh = np.zeros_like(z_curr)
        z_curr_oh[active_pos_channel] = 1.0
        
        z_next_oh = np.zeros_like(z_next)
        z_next_oh[np.argmax(z_next)] = 1.0
        
        self.bnn.update_weights(z_curr_oh, action, z_next_oh, lr=0.15)
        
        # 4. Update the linear decoder weights online via LMS gradient descent
        err = y_target - y_pred
        if not self.use_velocity_decoding:
            # Mask velocity heads to ignore velocity signal
            err[2:] = 0.0
            
        self.W_dec += lr_dec * np.outer(s_rate_norm, err)
        self.b_dec += (lr_dec * self.lr_dec_bias_scale) * err
        
        # Return probability grid decoded from predicted positions [X, Y]
        return generate_target_grid_numpy(y_pred[:2])

    def predict_trajectory(self, z_t: np.ndarray, action_sequence: list, current_velocity: np.ndarray, horizon: int = 8) -> list:
        """
        Recursively rolls out BNN predictions in [X, Y, dX, dY] continuous space (imagination loop).
        """
        c_start = calculate_centroid(z_t)
        # y_prev is the continuous state vector [X, Y, dX, dY]
        vel = current_velocity if self.use_velocity_decoding else np.zeros(2, dtype=np.float32)
        y_prev = np.array([c_start[0], c_start[1], vel[0], vel[1]], dtype=np.float32)
        
        predictions = []
        
        for h in range(horizon):
            if h < len(action_sequence):
                act = action_sequence[h]
            else:
                act = action_sequence[-1] if len(action_sequence) > 0 else 0
                
            if h == 0:
                # Phase Realignment: First step (h=1) is initialized using exact latent coordinate z_t + velocity
                y_pred = y_prev.copy()
                y_pred[0] += y_prev[2]
                y_pred[1] += y_prev[3]
            else:
                # Find active pos channel from continuous position y_prev[:2]
                col = int(np.clip(round(y_prev[0] - 0.5), 0, 7))
                row = int(np.clip(round(y_prev[1] - 0.5), 0, 7))
                active_pos_channel = row * 8 + col
                
                # Stimulate BNN (imagination loop does not trigger physical boundaries)
                self.bnn.stimulate(active_pos_channel, act, boundary_penalty=False)
                spike_frames = self.bnn.read_frames(frame_count=5)
                
                # Decode spikes to [X, Y, dX, dY] using trained linear decoder
                recording_spikes = spike_frames[:, 68:132]
                s_rate = np.mean(recording_spikes, axis=0)
                
                s_sum = s_rate.sum()
                s_rate_norm = s_rate / s_sum if s_sum > 0 else s_rate
                
                y_pred = np.dot(s_rate_norm, self.W_dec) + self.b_dec
                
                if not self.use_velocity_decoding:
                    # Force velocity to remain zero
                    y_pred[2:] = 0.0
                else:
                    # Velocity Integration: update position using the decoded velocity scaled by gain_factor
                    y_pred[0] = y_prev[0] + self.gain_factor * y_pred[2]
                    y_pred[1] = y_prev[1] + self.gain_factor * y_pred[3]
            
            # Clip position to grid bounds to prevent diverging trajectory drift
            y_pred[0] = np.clip(y_pred[0], 0.1, 7.9)
            y_pred[1] = np.clip(y_pred[1], 0.1, 7.9)
            # Clip velocity to physical speeds
            y_pred[2:] = np.clip(y_pred[2:], -0.5, 0.5)
            
            # Generate the latent grid output
            z_pred = generate_target_grid_numpy(y_pred[:2])
            predictions.append(z_pred)
            
            # Autoregressive feedback loop
            y_prev = y_pred
            
        return predictions

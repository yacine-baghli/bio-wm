import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Local imports
from .env import ParticleBoxEnv
from .encoder import ParticleEncoder, pretrain_encoder, SIGRegLoss
from .bio_loop import BioWorldModelLoop, calculate_centroid
from . import cl_sdk as cl

class FrameStacker:
    """
    Helper class to maintain a rolling stack of visual frames.
    For CONFIG_A: returns [1, 64, 64] (single frame).
    For CONFIG_B: returns [3, 64, 64] (concatenated 3 consecutive frames).
    """
    def __init__(self, num_frames=3, size=64):
        self.num_frames = num_frames
        self.size = size
        self.reset()
        
    def reset(self):
        self.stack = []
        
    def push_and_get(self, frame):
        if len(self.stack) == 0:
            # Pad by repeating the initial frame
            self.stack = [frame.copy() for _ in range(self.num_frames)]
        else:
            self.stack.pop(0)
            self.stack.append(frame.copy())
        return np.stack(self.stack, axis=0)

class ReplayBuffer:
    """
    Replay buffer to store stacked frames for SIGReg batch training.
    """
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        
    def push(self, stacked_frame):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(stacked_frame)
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), min(len(self.buffer), batch_size), replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return np.array(batch, dtype=np.float32)
        
    def __len__(self):
        return len(self.buffer)

def get_actual_future_latent_path(env, encoder, current_stack, action, horizon=8, device='cpu'):
    """
    Temporarily steps the environment forward in time to generate ground-truth future
    frames, encodes them, and extracts the sequence of actual 2D centroids.
    Restores the environment state back to the original step.
    """
    # Save original state variables
    orig_x, orig_y, orig_dx, orig_dy = env.x, env.y, env.dx, env.dy
    
    future_centroids = []
    
    # Create a local stack for rolling prediction (if stacked)
    local_stack = list(current_stack) if encoder.in_channels > 1 else [current_stack]
    
    for h in range(horizon):
        # Step physics forward assuming action continuation
        noisy_frame, _, _, _ = env.step(action)
        
        # Maintain local rolling stack
        if encoder.in_channels == 1:
            input_stack = np.expand_dims(noisy_frame, axis=0)
        else:
            local_stack.pop(0)
            local_stack.append(noisy_frame)
            input_stack = np.stack(local_stack, axis=0)
            
        # Run encoder forward to obtain actual latent grid
        input_t = torch.tensor(input_stack, dtype=torch.float32).unsqueeze(0).to(device)
            
        with torch.no_grad():
            z_raw = encoder(input_t).squeeze(0)
            z_prob = torch.softmax(z_raw, dim=0).cpu().numpy()
            
        c = calculate_centroid(z_prob)
        future_centroids.append(c)
        
    # Restore original environment physical state
    env.x, env.y, env.dx, env.dy = orig_x, orig_y, orig_dx, orig_dy
    
    return np.array(future_centroids, dtype=np.float32)

def train_and_evaluate_config(in_channels: int, use_boundary_penalty: bool, use_velocity_decoding: bool, 
                               steps_train=1000, steps_eval=500, gain_factor=1.0, lr_dec_bias_scale=0.1, 
                               hebb_lr=0.15, decay_factor=0.5, lr_dec=0.15, checkpoint_interval=5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==============================================")
    print(f"INITIALIZING CONFIG: channels={in_channels} | BP={use_boundary_penalty} | VD={use_velocity_decoding} | Gain={gain_factor}")
    print(f"==============================================")
    
    # 1. Initialize modules
    env = ParticleBoxEnv()
    encoder = ParticleEncoder(in_channels=in_channels).to(device)
    sigreg_loss_fn = SIGRegLoss(channels=64, sketch_dim=64).to(device)
    
    optimizer = optim.Adam(encoder.parameters(), lr=2e-4)
    criterion_pred = nn.MSELoss()
    
    # Pretrain the encoder at startup
    pretrain_encoder(encoder, num_samples=800, epochs=15)
    
    replay_buffer = ReplayBuffer(capacity=1000)
    stacker = FrameStacker(num_frames=in_channels)
    
    # Phase 1: Online Training
    print(f"\nRunning online training phase ({steps_train} steps)...")
    noisy_frame, _, pos = env.reset()
    stacker.reset()
    current_stack = stacker.push_and_get(noisy_frame)
    replay_buffer.push(current_stack)
    
    action = np.random.choice([0, 1, 2, 3])
    
    # Custom subclass of cl.MockDishBrain to override LTD decay_factor
    class ConfiguredMockDishBrain(cl.MockDishBrain):
        def read_frames(self, frame_count: int = 5) -> np.ndarray:
            micro_steps_per_frame = 10
            frames = []
            
            if self.active_boundary_penalty:
                # Use custom decay factor
                self.weights[self.active_position, :] *= decay_factor
                self.weights[64 + self.active_action, :] *= decay_factor
                self._normalize_weights()
            
            # Precompute input current since active_position and active_action are constant during frames read
            current_val = self.weights[self.active_position, :] + self.weights[64 + self.active_action, :]
            
            for _ in range(frame_count):
                frame_data = np.zeros(self.num_electrodes, dtype=np.float32)
                frame_data[self.active_position] = 1.0
                frame_data[64 + self.active_action] = 1.0
                if self.active_boundary_penalty:
                    frame_data[132] = 1.0
                    
                output_spikes = np.zeros(self.num_outputs, dtype=np.float32)
                for _ in range(micro_steps_per_frame):
                    if self.active_boundary_penalty:
                        noise = np.random.normal(0.0, 0.5, self.num_outputs)
                    else:
                        noise = np.random.normal(0.0, 0.08, self.num_outputs)
                    
                    self.v = self.v - self.leak * (self.v - self.v_rest) + current_val + noise
                    spiked = self.v >= self.v_thresh
                    output_spikes[spiked] += 1.0
                    self.v[spiked] = self.v_reset
                    
                frame_data[68:132] = output_spikes / float(micro_steps_per_frame)
                frames.append(frame_data)
                
            return np.array(frames, dtype=np.float32)

    with cl.open() as loop:
        # Swap the standard brain for our configured brain with customized LTD decay
        configured_brain = ConfiguredMockDishBrain()
        configured_brain.weights = loop.weights.copy()
        
        bio_loop = BioWorldModelLoop(
            configured_brain, 
            use_boundary_penalty=use_boundary_penalty, 
            use_velocity_decoding=use_velocity_decoding,
            gain_factor=gain_factor,
            lr_dec_bias_scale=lr_dec_bias_scale
        )
        
        for step in range(1, steps_train + 1):
            if step % 15 == 0:
                action = np.random.choice([0, 1, 2, 3])
                
            # Current stack forward
            s_curr_t = torch.tensor(current_stack, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                z_curr_raw = encoder(s_curr_t).squeeze(0)
                z_curr = torch.softmax(z_curr_raw, dim=0).cpu().numpy()
                
            # Environment step
            next_noisy, _, _, collision = env.step(action)
            next_stack = stacker.push_and_get(next_noisy)
            replay_buffer.push(next_stack)
            
            # Next stack forward
            s_next_t = torch.tensor(next_stack, dtype=torch.float32).unsqueeze(0).to(device)
            
            encoder.train()
            optimizer.zero_grad()
            
            z_next_t_raw = encoder(s_next_t)
            z_next_t = torch.softmax(z_next_t_raw, dim=1)
            z_next = z_next_t.squeeze(0).detach().cpu().numpy()
            
            # Compute actual next state vector (position and velocity scaled to 8x8)
            c_act = calculate_centroid(z_next)
            y_target = np.array([c_act[0], c_act[1], env.dx / 8.0, env.dy / 8.0], dtype=np.float32)
            
            # Predict and Learn online in the BNN
            z_pred = bio_loop.predict_and_learn(z_curr, action, z_next, y_target, boundary_penalty=collision, lr_dec=lr_dec)
            
            # Predict loss
            z_pred_t = torch.tensor(z_pred, dtype=torch.float32).unsqueeze(0).to(device)
            loss_pred = criterion_pred(z_next_t, z_pred_t)
            
            # SIGReg loss
            if len(replay_buffer) >= 64:
                batch = replay_buffer.sample(64)
                batch_t = torch.tensor(batch, dtype=torch.float32).to(device)
                z_batch_t = encoder(batch_t)
                loss_sigreg = sigreg_loss_fn(z_batch_t)
            else:
                loss_sigreg = torch.tensor(0.0, device=device)
                
            loss_total = loss_pred + 50.0 * loss_sigreg
            
            loss_total.backward()
            optimizer.step()
            
            # Save checkpoints of the model weights
            if step % checkpoint_interval == 0:
                os.makedirs("checkpoints", exist_ok=True)
                checkpoint_path = f"checkpoints/checkpoint_step_{step}.pt"
                torch.save({
                    'step': step,
                    'encoder_state_dict': encoder.state_dict(),
                    'W_dec': bio_loop.W_dec,
                    'b_dec': bio_loop.b_dec,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Saved weights checkpoint to {checkpoint_path}")
                
            current_stack = next_stack
            
        print("Online training phase complete.")
        
        # Phase 2: Evaluation Phase (500 steps)
        print(f"\nRunning evaluation phase ({steps_eval} steps)...")
        encoder.eval()
        
        eval_steps = []
        raw_frames_log = []
        true_paths_log = []
        pred_paths_log = []
        actions_log = []
        start_centroids_log = []
        
        for step in range(1, steps_eval + 1):
            if step % 15 == 0:
                action = np.random.choice([0, 1, 2, 3])
                
            # Current stack forward
            s_curr_t = torch.tensor(current_stack, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                z_curr_raw = encoder(s_curr_t).squeeze(0)
                z_curr = torch.softmax(z_curr_raw, dim=0).cpu().numpy()
                
            c_start = calculate_centroid(z_curr)
            start_centroids_log.append(c_start)
            
            # Get actual future H=8 latent path (centroids)
            true_future_path = get_actual_future_latent_path(
                env, encoder, current_stack, action, horizon=8, device=device
            )
            
            # Get predicted autoregressive rollout path (H=8 steps)
            future_actions = [action] * 8
            current_velocity = np.array([env.dx / 8.0, env.dy / 8.0], dtype=np.float32)
            z_pred_trajectory = bio_loop.predict_trajectory(z_curr, future_actions, current_velocity, horizon=8)
            pred_future_path = np.array([calculate_centroid(z) for z in z_pred_trajectory], dtype=np.float32)
            
            # Step environment forward physically
            next_noisy, _, _, collision = env.step(action)
            next_stack = stacker.push_and_get(next_noisy)
            
            # Log data
            eval_steps.append(step)
            # Log the latest visual frame of the stack
            raw_frames_log.append(current_stack[-1])
            true_paths_log.append(true_future_path)
            pred_paths_log.append(pred_future_path)
            actions_log.append(action)
            
            current_stack = next_stack
            
    # Calculate metrics
    true_paths_log = np.array(true_paths_log)  # [steps_eval, H, 2]
    pred_paths_log = np.array(pred_paths_log)  # [steps_eval, H, 2]
    start_centroids_log = np.array(start_centroids_log)  # [steps_eval, 2]
    
    # 1. Multi-Step Prediction Error (MSPE)
    mspe = np.mean(np.sum((pred_paths_log - true_paths_log)**2, axis=2), axis=0)
    
    # 2. Trajectory Variance Decay (TVD)
    pred_means = np.mean(pred_paths_log, axis=1, keepdims=True)  # [steps_eval, 1, 2]
    pred_vars = np.mean(np.sum((pred_paths_log - pred_means)**2, axis=2), axis=1)  # [steps_eval]
    
    true_means = np.mean(true_paths_log, axis=1, keepdims=True)  # [steps_eval, 1, 2]
    true_vars = np.mean(np.sum((true_paths_log - true_means)**2, axis=2), axis=1)  # [steps_eval]
    
    tvd = np.mean(pred_vars) / (np.mean(true_vars) + 1e-6)
    
    # 3. Effective Prediction Horizon (EPH)
    baseline_paths = np.repeat(np.expand_dims(start_centroids_log, 1), 8, axis=1)  # [steps_eval, 8, 2]
    mspe_baseline = np.mean(np.sum((true_paths_log - baseline_paths)**2, axis=2), axis=0)
    
    eph = 0
    for h in range(8):
        if mspe[h] < mspe_baseline[h]:
            eph = h + 1
            
    # 4. Strict Failure Metrics
    # True Endpoint & Predicted Endpoint (h=8 is index 7)
    true_endpoints = true_paths_log[:, 7, :]  # [steps_eval, 2]
    pred_endpoints = pred_paths_log[:, 7, :]  # [steps_eval, 2]
    
    # Horizon Failure Rate (HFR): % of endpoints exceeding 1.5 grid cells Euclidean error
    endpoint_dists = np.sqrt(np.sum((pred_endpoints - true_endpoints)**2, axis=1))
    hfr = np.mean(endpoint_dists > 1.5) * 100.0
    
    # Directional Failure Rate (DFR)
    v_true = true_endpoints - start_centroids_log  # [steps_eval, 2]
    v_pred = pred_endpoints - start_centroids_log  # [steps_eval, 2]
    
    dot_prod = np.sum(v_true * v_pred, axis=1)
    norm_true = np.linalg.norm(v_true, axis=1) + 1e-8
    norm_pred = np.linalg.norm(v_pred, axis=1) + 1e-8
    cosine_sim = dot_prod / (norm_true * norm_pred)
    dfr = np.mean(cosine_sim <= 0.0) * 100.0
    
    # Trajectory Magnitude Error (TME)
    tme = np.mean(norm_pred) / (np.mean(norm_true) + 1e-8)
    
    print(f"\nConfiguration Finished:")
    print(f"  TVD Score: {tvd:.4f}")
    print(f"  EPH Horizon: {eph} steps")
    print(f"  HFR Failure Rate: {hfr:.2f}%")
    print(f"  DFR Directional Failure Rate: {dfr:.2f}%")
    print(f"  TME Trajectory Magnitude Ratio: {tme:.4f}")
    print(f"  MSPE: {np.array2string(mspe, precision=4, separator=', ')}")
    
    return {
        'steps': np.array(eval_steps),
        'raw_frames': np.array(raw_frames_log),
        'true_paths': true_paths_log,
        'pred_paths': pred_paths_log,
        'actions': np.array(actions_log),
        'mspe': mspe,
        'tvd': tvd,
        'eph': eph,
        'mspe_baseline': mspe_baseline,
        'hfr': hfr,
        'dfr': dfr,
        'tme': tme,
        'weights': bio_loop.W_dec.copy()
    }

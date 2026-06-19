import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import custom modules
from env import ParticleBoxEnv
from encoder import ParticleEncoder, pretrain_encoder, generate_target_grid
from sigreg import SIGRegLoss
from bio_loop import BioWorldModelLoop, calculate_centroid
import cl_sdk as cl

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
        # Step physics forward assuming action continuation (constant velocity inertia)
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

def train_and_evaluate_config(in_channels: int, use_boundary_penalty: bool, use_velocity_decoding: bool, steps_train=1000, steps_eval=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==============================================")
    print(f"INITIALIZING CONFIG: channels={in_channels} | BP={use_boundary_penalty} | VD={use_velocity_decoding}")
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
    
    with cl.open() as loop:
        bio_loop = BioWorldModelLoop(
            loop, 
            use_boundary_penalty=use_boundary_penalty, 
            use_velocity_decoding=use_velocity_decoding
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
            z_pred = bio_loop.predict_and_learn(z_curr, action, z_next, y_target, boundary_penalty=collision)
            
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
            
    # Calculate professional metrics
    true_paths_log = np.array(true_paths_log)  # [steps_eval, H, 2]
    pred_paths_log = np.array(pred_paths_log)  # [steps_eval, H, 2]
    start_centroids_log = np.array(start_centroids_log)  # [steps_eval, 2]
    
    # 1. Multi-Step Prediction Error (MSPE)
    # Mean Squared Error for each horizon step h=1..8
    # Shape: [H]
    mspe = np.mean(np.sum((pred_paths_log - true_paths_log)**2, axis=2), axis=0)
    
    # 2. Trajectory Variance Decay (TVD)
    # Calculate spatial variance of predicted vs true paths
    # Variance computed as mean squared distance from path mean
    pred_means = np.mean(pred_paths_log, axis=1, keepdims=True)  # [steps_eval, 1, 2]
    pred_vars = np.mean(np.sum((pred_paths_log - pred_means)**2, axis=2), axis=1)  # [steps_eval]
    
    true_means = np.mean(true_paths_log, axis=1, keepdims=True)  # [steps_eval, 1, 2]
    true_vars = np.mean(np.sum((true_paths_log - true_means)**2, axis=2), axis=1)  # [steps_eval]
    
    tvd = np.mean(pred_vars) / (np.mean(true_vars) + 1e-6)
    
    # 3. Effective Prediction Horizon (EPH)
    # Compute baseline guess MSPE (assuming no movement from starting position z_t)
    # We reconstruct baseline path [steps_eval, H, 2] where all elements are the current centroid at time t
    baseline_paths = np.repeat(np.expand_dims(start_centroids_log, 1), 8, axis=1)  # [steps_eval, 8, 2]
    mspe_baseline = np.mean(np.sum((true_paths_log - baseline_paths)**2, axis=2), axis=0)
    
    eph = 0
    for h in range(8):
        if mspe[h] < mspe_baseline[h]:
            eph = h + 1
            
    print(f"\nConfiguration Finished:")
    print(f"  TVD Score: {tvd:.4f}")
    print(f"  EPH Horizon: {eph} steps")
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
        'mspe_baseline': mspe_baseline
    }

if __name__ == "__main__":
    # Run evaluation harness sequentially
    results_a = train_and_evaluate_config(in_channels=1, use_boundary_penalty=False, use_velocity_decoding=False, steps_train=1000, steps_eval=500)
    results_b = train_and_evaluate_config(in_channels=3, use_boundary_penalty=False, use_velocity_decoding=False, steps_train=1000, steps_eval=500)
    results_c = train_and_evaluate_config(in_channels=3, use_boundary_penalty=True, use_velocity_decoding=True, steps_train=1000, steps_eval=500)
    
    # Save structured test data to disk as compressed npz
    output_path = "test_session_output.npz"
    np.savez_compressed(
        output_path,
        config_a_steps=results_a['steps'],
        config_a_raw_frames=results_a['raw_frames'],
        config_a_true_paths=results_a['true_paths'],
        config_a_pred_paths=results_a['pred_paths'],
        config_a_actions=results_a['actions'],
        config_b_steps=results_b['steps'],
        config_b_raw_frames=results_b['raw_frames'],
        config_b_true_paths=results_b['true_paths'],
        config_b_pred_paths=results_b['pred_paths'],
        config_b_actions=results_b['actions'],
        config_c_steps=results_c['steps'],
        config_c_raw_frames=results_c['raw_frames'],
        config_c_true_paths=results_c['true_paths'],
        config_c_pred_paths=results_c['pred_paths'],
        config_c_actions=results_c['actions']
    )
    print(f"\nSerialized evaluation session output saved to {output_path}")
    
    # Save a copy to the artifacts folder if it exists
    artifacts_dir = r"C:\Users\Yacine\.gemini\antigravity-ide\brain\fc5a9017-606d-4d92-8f88-9056153aec5a"
    if os.path.exists(artifacts_dir):
        artifact_npz_path = os.path.join(artifacts_dir, "test_session_output.npz")
        np.savez_compressed(
            artifact_npz_path,
            config_a_steps=results_a['steps'],
            config_a_raw_frames=results_a['raw_frames'],
            config_a_true_paths=results_a['true_paths'],
            config_a_pred_paths=results_a['pred_paths'],
            config_a_actions=results_a['actions'],
            config_b_steps=results_b['steps'],
            config_b_raw_frames=results_b['raw_frames'],
            config_b_true_paths=results_b['true_paths'],
            config_b_pred_paths=results_b['pred_paths'],
            config_b_actions=results_b['actions'],
            config_c_steps=results_c['steps'],
            config_c_raw_frames=results_c['raw_frames'],
            config_c_true_paths=results_c['true_paths'],
            config_c_pred_paths=results_c['pred_paths'],
            config_c_actions=results_c['actions']
        )
        print(f"Saved duplicate binary npz artifact to {artifact_npz_path}")
        
    # Generate final comparative analysis table
    print("\n" + "="*95)
    print("COMPARATIVE EVALUATION ANALYSIS:")
    print("="*95)
    print(f"{'Metric / Configuration':<30} | {'CONFIG_A (Baseline)':<20} | {'CONFIG_B (Stacking)':<20} | {'CONFIG_C (LeWM Full)':<20}")
    print("-"*100)
    print(f"{'MSPE at step h=1':<30} | {results_a['mspe'][0]:.6f}             | {results_b['mspe'][0]:.6f}             | {results_c['mspe'][0]:.6f}")
    print(f"{'MSPE at step h=4':<30} | {results_a['mspe'][3]:.6f}             | {results_b['mspe'][3]:.6f}             | {results_c['mspe'][3]:.6f}")
    print(f"{'MSPE at step h=8':<30} | {results_a['mspe'][7]:.6f}             | {results_b['mspe'][7]:.6f}             | {results_c['mspe'][7]:.6f}")
    print(f"{'Trajectory Variance Decay (TVD)':<30} | {results_a['tvd']:.6f}             | {results_b['tvd']:.6f}             | {results_c['tvd']:.6f}")
    print(f"{'Effective Prediction Horizon (EPH)':<30} | {results_a['eph']:d} steps             | {results_b['eph']:d} steps             | {results_c['eph']:d} steps")
    print("="*95)
    
    # Save final automated diagnostic line plot comparison
    plt.figure(figsize=(10, 6))
    horizon = np.arange(1, 9)
    plt.plot(horizon, results_a['mspe'], 'ro-', linewidth=2, markersize=6, label="CONFIG_A: Baseline Position-Only")
    plt.plot(horizon, results_b['mspe'], 'bs-', linewidth=2, markersize=6, label="CONFIG_B: Stacking Only")
    plt.plot(horizon, results_c['mspe'], 'g^-', linewidth=2, markersize=6, label="CONFIG_C: Temporal Stacking + BP + Velocity")
    plt.plot(horizon, results_a['mspe_baseline'], 'k--', label="Unmoving Baseline Guess")
    plt.title("Autoregressive Multi-Step Prediction Error (MSPE) comparison", fontsize=12, fontweight='bold')
    plt.xlabel("Horizon Step (h)")
    plt.ylabel("Mean Squared Error (MSPE)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper left")
    
    # Save plot to workspace and artifact directory
    plot_path_ws = "evaluation_comparison.png"
    plt.savefig(plot_path_ws, dpi=150)
    print(f"Saved diagnostic line plot to {plot_path_ws}")
    
    if os.path.exists(artifacts_dir):
        plot_path_art = os.path.join(artifacts_dir, "evaluation_comparison.png")
        plt.savefig(plot_path_art, dpi=150)
        print(f"Saved duplicate plot image artifact to {plot_path_art}")

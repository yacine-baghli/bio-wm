import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import custom modules
from env import ParticleBoxEnv
from encoder import ParticleEncoder, pretrain_encoder
from sigreg import SIGRegLoss
from bio_loop import BioWorldModelLoop, calculate_centroid
import cl_sdk as cl

class ReplayBuffer:
    """
    Replay buffer to store raw noisy environment frames.
    Provides batch sampling for the SIGReg regularizer to maintain stable distribution statistics.
    """
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer = []
        
    def push(self, frame: np.ndarray):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(frame)
        
    def sample(self, batch_size: int) -> np.ndarray:
        indices = np.random.choice(len(self.buffer), min(len(self.buffer), batch_size), replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return np.array(batch, dtype=np.float32)
        
    def __len__(self) -> int:
        return len(self.buffer)

def run_simulation(steps: int = 2000, headless: bool = False, lambda_sigreg: float = 1.0, artifact_dir: str = "."):
    # 1. Hardware/Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # 2. Initialize modules
    env = ParticleBoxEnv()
    encoder = ParticleEncoder().to(device)
    sigreg_loss_fn = SIGRegLoss(channels=64, sketch_dim=64).to(device)
    
    # Optimizer for encoder
    optimizer = optim.Adam(encoder.parameters(), lr=2e-4)
    criterion_pred = nn.MSELoss()
    
    # 3. Pre-train CNN encoder at startup to anchor representation
    pretrain_encoder(encoder, num_samples=800, epochs=15)
    
    # 4. Initialize Replay Buffer
    replay_buffer = ReplayBuffer(capacity=1000)
    
    # 5. Connect to mock BNN via context manager
    with cl.open() as loop:
        bio_loop = BioWorldModelLoop(loop)
        
        # Reset environment
        noisy_frame, clean_frame, pos = env.reset()
        replay_buffer.push(noisy_frame)
        
        # Lists to store metrics for plotting
        steps_history = []
        pred_losses = []
        sigreg_losses = []
        variance_history = []
        actual_straightness_history = []
        pred_straightness_history = []
        
        # Sliding windows for trajectory plotting (last 30 steps)
        actual_path = []
        pred_path = []
        
        # Track persistent action selection (applying thrust for 15 steps to show inertia)
        action = np.random.choice([0, 1, 2, 3])
        
        # Initialize visualization dashboard if not headless
        if not headless:
            plt.ion()
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle("Bio-JEPA LeWorldModel (LeWM) Live Dashboard", fontsize=14, fontweight='bold')
        
        print("\nStarting closed-loop Bio-JEPA LeWM Simulation...")
        print("Step | Pred Loss | SIGReg Loss | Latent Var | Act Straight | Pred Straight")
        print("-" * 80)
        
        start_time = time.time()
        
        for step in range(1, steps + 1):
            # Sample persistent actions (changing thrust direction every 15 steps)
            if step % 15 == 0:
                action = np.random.choice([0, 1, 2, 3])
                
            # Current state encoder forward
            s_curr_t = torch.tensor(noisy_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                z_curr_raw = encoder(s_curr_t).squeeze(0)
                z_curr = torch.softmax(z_curr_raw, dim=0).cpu().numpy()
                
            # Step environment to get next state
            next_noisy, next_clean, next_pos, collision = env.step(action)
            replay_buffer.push(next_noisy)
            
            # Target next state encoder forward (to calculate predictive loss)
            s_next_t = torch.tensor(next_noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # Get actual next representation
            encoder.train()
            optimizer.zero_grad()
            
            z_next_t_raw = encoder(s_next_t)  # [1, 64]
            z_next_t = torch.softmax(z_next_t_raw, dim=1)  # [1, 64]
            z_next = z_next_t.squeeze(0).detach().cpu().numpy()
            
            # Compute actual next state vector (position and velocity scaled to 8x8)
            c_act = calculate_centroid(z_next)
            y_target = np.array([c_act[0], c_act[1], env.dx / 8.0, env.dy / 8.0], dtype=np.float32)
            
            # Predict and Learn online in the BNN
            z_pred = bio_loop.predict_and_learn(z_curr, action, z_next, y_target, boundary_penalty=collision)
            
            # Calculate prediction loss (MSE between encoder output and BNN spike prediction)
            z_pred_t = torch.tensor(z_pred, dtype=torch.float32).unsqueeze(0).to(device)
            loss_pred = criterion_pred(z_next_t, z_pred_t)
            
            # Calculate SIGReg loss on a batch from the replay buffer
            if len(replay_buffer) >= 64:
                batch_frames = replay_buffer.sample(64)
                batch_frames_t = torch.tensor(batch_frames, dtype=torch.float32).unsqueeze(1).to(device)
                z_batch_t = encoder(batch_frames_t)  # [64, 64]
                loss_sigreg = sigreg_loss_fn(z_batch_t)
                latent_variance = torch.var(z_batch_t, dim=0).mean().item()
            else:
                loss_sigreg = torch.tensor(0.0, device=device)
                latent_variance = 0.0
                
            # Joint Two-Term Loss
            loss_total = loss_pred + lambda_sigreg * loss_sigreg
            
            # Optimize digital encoder weights end-to-end
            loss_total.backward()
            optimizer.step()
            
            # Update state
            noisy_frame = next_noisy
            
            # Compute centroids for trajectory tracking (scaled to 8x8 latent dimensions)
            actual_path.append(c_act)
            if len(actual_path) > 30:
                actual_path.pop(0)
                
            # Run Autoregressive Rollout Loop (H=8 steps future forecasting)
            future_actions = [action] * 8
            current_velocity = np.array([env.dx / 8.0, env.dy / 8.0], dtype=np.float32)
            z_pred_trajectory = bio_loop.predict_trajectory(z_curr, future_actions, current_velocity, horizon=8)
            pred_future_centroids = [calculate_centroid(z) for z in z_pred_trajectory]
            
            # Calculate Trajectory Straightness Metric (over the H=8 rollout horizon)
            c_current = calculate_centroid(z_curr)
            full_pred_trajectory = [c_current] + pred_future_centroids
            disp_pred_traj = np.linalg.norm(full_pred_trajectory[-1] - full_pred_trajectory[0])
            path_len_pred_traj = sum(np.linalg.norm(full_pred_trajectory[i] - full_pred_trajectory[i-1]) for i in range(1, len(full_pred_trajectory)))
            straight_pred = disp_pred_traj / (path_len_pred_traj + 1e-6)
            
            if len(actual_path) >= 8:
                disp_act_traj = np.linalg.norm(actual_path[-1] - actual_path[-8])
                path_len_act_traj = sum(np.linalg.norm(actual_path[i] - actual_path[i-1]) for i in range(-7, 0))
                straight_act = disp_act_traj / (path_len_act_traj + 1e-6)
            else:
                straight_act = 1.0
                
            # Log metrics
            steps_history.append(step)
            pred_losses.append(loss_pred.item())
            sigreg_losses.append(loss_sigreg.item())
            variance_history.append(latent_variance)
            actual_straightness_history.append(straight_act)
            pred_straightness_history.append(straight_pred)
            
            # Print printout every 100 steps
            if step % 100 == 0:
                print(f"{step:4d} | {loss_pred.item():.6f}  | {loss_sigreg.item():.6f}    | {latent_variance:.6f}   | {straight_act:.4f}       | {straight_pred:.4f}")
                
            # Live Plotting Update
            if not headless and step % 10 == 0:
                # Panel 1: Noisy actual state
                axs[0].clear()
                axs[0].imshow(noisy_frame, cmap='gray')
                axs[0].set_title(f"Noisy Frame (64x64)\nStep {step}")
                axs[0].axis('off')
                
                # Panel 2: 8x8 Digital Latent Grid (Z_t) Heatmap + History
                axs[1].clear()
                axs[1].imshow(z_next.reshape(8, 8), cmap='hot', interpolation='nearest', extent=[0, 8, 8, 0])
                if len(actual_path) > 0:
                    act_coords = np.array(actual_path)
                    axs[1].plot(act_coords[:, 0], act_coords[:, 1], 'c-', linewidth=1.5, label="History Path")
                    axs[1].scatter(act_coords[-1, 0], act_coords[-1, 1], color='cyan', edgecolors='white', s=50, zorder=5, label="Current Z_t")
                axs[1].set_xlim(0, 8)
                axs[1].set_ylim(0, 8)
                axs[1].invert_yaxis()  # Match image indexing
                axs[1].set_title("Digital Latent Grid (Z_t)\n(8x8 Heatmap + Path)")
                axs[1].legend(loc="upper right", fontsize=8)
                axs[1].grid(True, linestyle='--', alpha=0.3)
                
                # Panel 3: 8x8 Biological World Model H-step Prediction Trajectory
                axs[2].clear()
                pred_coords = np.array(pred_future_centroids)
                current_pos = actual_path[-1] if len(actual_path) > 0 else np.array([4.0, 4.0])
                full_pred_path = np.vstack([current_pos, pred_coords])
                
                # Draw imagined future rollout
                axs[2].plot(full_pred_path[:, 0], full_pred_path[:, 1], 'ro-', linewidth=2.5, markersize=5, label="Imagined Future (H=8)")
                # Overlay actual past trajectory for reference
                if len(actual_path) >= 8:
                    past_coords = np.array(actual_path[-8:])
                    axs[2].plot(past_coords[:, 0], past_coords[:, 1], 'bo--', alpha=0.5, label="Actual Past (H=8)")
                    
                axs[2].set_xlim(0, 8)
                axs[2].set_ylim(0, 8)
                axs[2].invert_yaxis()
                axs[2].set_title("Biological Prediction (Ẑ_t+1:t+H)\n(H=8 Imaginary Rollout)")
                axs[2].legend(loc="upper right", fontsize=8)
                axs[2].grid(True, linestyle='--', alpha=0.5)
                
                plt.draw()
                plt.pause(0.001)
                
        # 6. Save final performance plot to artifact directory
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(pred_losses, 'r-', label="Raw Prediction Loss", alpha=0.3)
        window = 100
        smooth_pred = np.convolve(pred_losses, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(pred_losses)), smooth_pred, 'r-', linewidth=2, label="Prediction Loss (MA100)")
        plt.plot(sigreg_losses, 'g-', label="SIGReg Loss", alpha=0.3)
        smooth_sigreg = np.convolve(sigreg_losses, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(sigreg_losses)), smooth_sigreg, 'g-', linewidth=2, label="SIGReg Loss (MA100)")
        plt.title("LeWM Loss Convergence & Stability")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.subplot(1, 2, 2)
        plt.plot(variance_history, 'm-', label="Latent Channel Variance", alpha=0.8)
        smooth_straight_pred = np.convolve(pred_straightness_history, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(pred_straightness_history)), smooth_straight_pred, 'b-', linewidth=2, label="BNN Straightness (MA100)")
        smooth_straight_act = np.convolve(actual_straightness_history, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(actual_straightness_history)), smooth_straight_act, 'k--', linewidth=1.5, label="Actual Straightness (MA100)")
        plt.title("Latent Dynamics & Path Straightening")
        plt.xlabel("Step")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_path = os.path.join(artifact_dir, "lewm_performance.png")
        plt.savefig(save_path, dpi=150)
        print(f"\nFinal performance plots saved to {save_path}")
        
        elapsed_time = time.time() - start_time
        print(f"Simulation finished in {elapsed_time:.2f} seconds.")
        
        # 7. Print Summary Metrics
        init_pred_loss = np.mean(pred_losses[:100])
        final_pred_loss = np.mean(pred_losses[-100:])
        final_sigreg_loss = np.mean(sigreg_losses[-100:])
        final_variance = np.mean(variance_history[-100:])
        final_straight = np.mean(pred_straightness_history[-100:])
        
        print("\n" + "="*50)
        print("SUMMARY PERFORMANCE METRICS:")
        print(f"Initial Prediction Loss (first 100 steps): {init_pred_loss:.6f}")
        print(f"Final Prediction Loss (last 100 steps):    {final_pred_loss:.6f}")
        print(f"Final SIGReg Loss (last 100 steps):        {final_sigreg_loss:.6f}")
        print(f"Final Latent Variance (last 100 steps):    {final_variance:.6f} (Standard Isotropic Goal: ~1.0)")
        print(f"Final BNN Trajectory Straightness:         {final_straight:.4f}")
        print("="*50)
        
        return {
            'init_pred_loss': init_pred_loss,
            'final_pred_loss': final_pred_loss,
            'final_sigreg_loss': final_sigreg_loss,
            'final_variance': final_variance,
            'final_straightness': final_straight,
            'elapsed_time': elapsed_time
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bio-JEPA LeWorldModel (LeWM) Simulator")
    parser.add_argument("--steps", type=int, default=2000, help="Number of simulation steps")
    parser.add_argument("--headless", action="store_true", help="Run without live graphical display")
    parser.add_argument("--lambda-sigreg", type=float, default=25.0, help="Weight of SIGReg regularization")
    parser.add_argument("--artifact-dir", type=str, default=".", help="Directory to save final metrics plot")
    
    args = parser.parse_args()
    
    # Run simulation
    run_simulation(
        steps=args.steps,
        headless=args.headless,
        lambda_sigreg=args.lambda_sigreg,
        artifact_dir=args.artifact_dir
    )

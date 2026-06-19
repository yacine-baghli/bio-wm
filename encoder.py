import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from env import ParticleBoxEnv
from sigreg import SIGRegLoss

class ParticleEncoder(nn.Module):
    """
    Lightweight CNN Encoder.
    Input: Noisy grayscale 64x64 frame stack of shape [Batch, in_channels, 64, 64].
    Output: 64-dimensional latent vector z_t, which represents an 8x8 spatial grid.
    """
    def __init__(self, latent_dim: int = 64, in_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Raw frame stack of shape [Batch, in_channels, 64, 64]
        Returns:
            torch.Tensor: Latent embedding of shape [Batch, 64]
        """
        feat = self.conv(x)
        z = self.fc(feat)
        return z

def generate_target_grid(pos: np.ndarray, size: int = 64, latent_grid_size: int = 8, sigma: float = 4.0) -> np.ndarray:
    """
    Generate an 8x8 continuous Gaussian target grid centered at the particle's (x, y) coordinates.
    """
    x, y = pos[0], pos[1]
    grid = np.zeros((latent_grid_size, latent_grid_size), dtype=np.float32)
    scale = size / latent_grid_size # 8.0
    
    for r in range(latent_grid_size):
        for c in range(latent_grid_size):
            # Coordinate of center of the latent cell in the 64x64 space
            cell_x = (c + 0.5) * scale
            cell_y = (r + 0.5) * scale
            dist_sq = (cell_x - x)**2 + (cell_y - y)**2
            grid[r, c] = np.exp(-dist_sq / (2 * sigma**2))
            
    # Normalize to form a valid probability distribution (sum to 1)
    grid_sum = grid.sum()
    if grid_sum > 0:
        grid = grid / grid_sum
    return grid.flatten()

def pretrain_encoder(encoder: ParticleEncoder, num_samples: int = 800, epochs: int = 15, batch_size: int = 32, lr: float = 1e-3) -> float:
    """
    Pretrains the CNN encoder to extract the clean particle coordinates from noisy frame stacks.
    This anchors the latent space z_t to the physical spatial position of the particle.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.train()
    
    env = ParticleBoxEnv()
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 1. Generate synthetic sequential dataset with sliding history window
    frames = []
    targets = []
    
    stack = []
    noisy, clean, pos = env.reset()
    for _ in range(encoder.in_channels - 1):
        stack.append(noisy)
        action = np.random.choice([0, 1, 2, 3])
        noisy, clean, pos, _ = env.step(action)
    stack.append(noisy)
    
    for _ in range(num_samples):
        target = generate_target_grid(pos)
        
        frames.append(np.stack(stack, axis=0))
        targets.append(target)
        
        # Advance env
        action = np.random.choice([0, 1, 2, 3])
        noisy, clean, pos, _ = env.step(action)
        stack.pop(0)
        stack.append(noisy)
        
    frames = np.array(frames, dtype=np.float32) # [num_samples, in_channels, 64, 64]
    targets = np.array(targets, dtype=np.float32) # [num_samples, 64]
    
    # Convert to tensors
    frames_t = torch.tensor(frames).to(device) # [num_samples, in_channels, 64, 64]
    targets_t = torch.tensor(targets).to(device) # [num_samples, 64]
    
    print(f"Pretraining CNN Encoder for {epochs} epochs on {device} (channels={encoder.in_channels})...")
    for epoch in range(epochs):
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = frames_t[indices], targets_t[indices]
            
            optimizer.zero_grad()
            pred_y_raw = encoder(batch_x)
            pred_y = torch.softmax(pred_y_raw, dim=1)
            loss = criterion(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(indices)
            
        epoch_loss /= num_samples
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}")
            
    encoder.eval()
    return epoch_loss

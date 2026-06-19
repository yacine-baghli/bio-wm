import torch
import torch.nn as nn

class SIGRegLoss(nn.Module):
    """
    Sketched-Isotropic-Gaussian Regularizer (SIGReg) Loss.
    
    Stabilizes the representation space by forcing the distribution of latent
    embeddings to match a standard multivariate isotropic Gaussian N(0, I)
    using random 1D projections (Cramér-Wold theorem) and comparing their
    Empirical Characteristic Functions (ECFs) with the analytical Gaussian CF.
    """
    def __init__(self, channels: int, sketch_dim: int = 64, num_t: int = 17):
        super().__init__()
        self.channels = channels
        self.sketch_dim = sketch_dim
        
        # Register integration points t between -5 and 5
        self.register_buffer('t', torch.linspace(-5, 5, num_t))
        # Analytical characteristic function values for N(0, 1): ECF(t) = exp(-0.5 * t^2)
        self.register_buffer('exp_f', torch.exp(-0.5 * self.t**2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the SIGReg loss on the batch of latents x.
        
        Args:
            x (torch.Tensor): Latent representation tensor of shape [N, C] where
                              N is the batch size and C is the latent dimension (channels).
                              
        Returns:
            torch.Tensor: Scalar loss term.
        """
        N, C = x.size()
        if N <= 1:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
            
        # 1. Generate random unit projections along the hypersphere
        A = torch.randn(C, self.sketch_dim, device=x.device)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)
        
        # 2. Project x along the random directions
        proj = x @ A  # [N, sketch_dim]
        
        # 3. Compute the Empirical Characteristic Function (ECF) (using the real part)
        # args shape: [N, sketch_dim, num_t]
        args = proj.unsqueeze(2) * self.t.view(1, 1, -1)
        ecf = torch.cos(args).mean(dim=0)  # [sketch_dim, num_t]
        
        # 4. Compute L2-weighted discrepancy between empirical and theoretical CFs
        diff_sq = (ecf - self.exp_f.unsqueeze(0)).square()  # [sketch_dim, num_t]
        err = diff_sq * self.exp_f.unsqueeze(0)
        
        # 5. Integrate using trapezoidal sum
        dt = (self.t[-1] - self.t[0]) / (len(self.t) - 1)
        loss = err.sum(dim=1) * dt
        
        return loss.mean()

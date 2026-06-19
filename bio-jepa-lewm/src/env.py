import numpy as np

class ParticleBoxEnv:
    """
    2D Particle Box environment with physical thrust actions and high visual noise.
    
    The particle is represented as a 3x3 pixel block moving within a 64x64 grid.
    Flicker noise is injected at each frame to test the representation learning.
    """
    def __init__(self, size: int = 64, particle_size: int = 3, noise_level: float = 0.1):
        self.size = size
        self.particle_size = particle_size
        self.noise_level = noise_level
        self.reset()
        
    def reset(self):
        # Position is float to allow smooth continuous updates
        # Start near the center
        self.x = float(self.size // 2)
        self.y = float(self.size // 2)
        
        # Initial velocities
        self.dx = np.random.uniform(0.5, 1.5) * np.random.choice([-1.0, 1.0])
        self.dy = np.random.uniform(0.5, 1.5) * np.random.choice([-1.0, 1.0])
        
        # Speed limits
        self.max_speed = 3.0
        self.thrust_accel = 0.4
        
        return self._get_frame()
        
    def step(self, action: int):
        """
        Step the physics simulation forward.
        
        Actions:
        0: Up    (decreases dy)
        1: Down  (increases dy)
        2: Left  (decreases dx)
        3: Right (increases dx)
        """
        self.collision = False
        
        # Apply action thrust
        if action == 0:
            self.dy -= self.thrust_accel
        elif action == 1:
            self.dy += self.thrust_accel
        elif action == 2:
            self.dx -= self.thrust_accel
        elif action == 3:
            self.dx += self.thrust_accel
            
        # Clip speed to prevent excessive velocities
        self.dx = np.clip(self.dx, -self.max_speed, self.max_speed)
        self.dy = np.clip(self.dy, -self.max_speed, self.max_speed)
        
        # Update positions
        self.x += self.dx
        self.y += self.dy
        
        # Deterministic boundary bouncing (elastic collisions)
        # We account for the particle thickness (particle_size // 2 on each side)
        half_p = self.particle_size / 2.0
        min_pos = half_p
        max_pos = self.size - half_p
        
        # Bounce X
        if self.x <= min_pos:
            self.x = min_pos + (min_pos - self.x)
            self.dx = -self.dx
            self.collision = True
        elif self.x >= max_pos:
            self.x = max_pos - (self.x - max_pos)
            self.dx = -self.dx
            self.collision = True
            
        # Bounce Y
        if self.y <= min_pos:
            self.y = min_pos + (min_pos - self.y)
            self.dy = -self.dy
            self.collision = True
        elif self.y >= max_pos:
            self.y = max_pos - (self.y - max_pos)
            self.dy = -self.dy
            self.collision = True
            
        # Double check bounds constraints in case of high speed
        self.x = np.clip(self.x, min_pos, max_pos)
        self.y = np.clip(self.y, min_pos, max_pos)
        
        noisy, clean, pos = self._get_frame()
        return noisy, clean, pos, self.collision
        
    def _get_frame(self):
        # 1. Generate clean frame
        clean = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Draw 3x3 particle
        px = int(round(self.x))
        py = int(round(self.y))
        
        half_p = self.particle_size // 2
        for dy_idx in range(-half_p, half_p + 1):
            for dx_idx in range(-half_p, half_p + 1):
                ny_idx = py + dy_idx
                nx_idx = px + dx_idx
                if 0 <= ny_idx < self.size and 0 <= nx_idx < self.size:
                    clean[ny_idx, nx_idx] = 1.0
                    
        # 2. Inject structural flicker noise (10% of pixels randomly flipped)
        # Unrelated pixels turn randomly on/off
        noise_mask = np.random.rand(self.size, self.size) < self.noise_level
        noisy = clean.copy()
        noisy[noise_mask] = 1.0 - noisy[noise_mask]
        
        # Keep clean position as coordinate
        pos = np.array([self.x, self.y], dtype=np.float32)
        
        return noisy, clean, pos

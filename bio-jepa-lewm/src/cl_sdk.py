import numpy as np
import contextlib

class ChannelSet:
    def __init__(self, channels):
        self.channels = channels

class StimDesign:
    def __init__(self, phase1_duration, phase1_amplitude, phase2_duration, phase2_amplitude):
        pass

class MockDishBrain:
    """
    Simulated biological neural network (BNN) representing a Cortical Labs DishBrain.
    
    Contains 133 electrodes:
    - 0 to 63: Spatial sensory inputs (8x8 grid mapped to 64 channels).
    - 64 to 67: Motor input channels (4 action channels).
    - 68 to 131: Recording zone (64 channels representing the predicted next-state grid).
    - 132: Boundary penalty input electrode (triggers chaotic Entropy Burst).
    
    Uses a Leaky Integrate-and-Fire (LIF) model and online Hebbian association rules.
    """
    def __init__(self):
        self.num_inputs = 68  # 64 spatial + 4 action
        self.num_outputs = 64  # 64 output prediction electrodes
        self.num_electrodes = 133  # 64 + 4 + 64 + 1
        
        # Initialize synaptic weights connecting inputs to outputs
        self.weights = np.random.uniform(0.1, 0.4, (self.num_inputs, self.num_outputs))
        
        # Normalize weights columns
        self._normalize_weights()
        
        # LIF Neuron state variables
        self.v = np.zeros(self.num_outputs, dtype=np.float32)
        self.v_rest = 0.0
        self.v_reset = 0.0
        self.v_thresh = 1.0
        self.leak = 0.15
        
        # Active stimulation channels
        self.active_position = 0
        self.active_action = 0
        self.active_boundary_penalty = False

    def stimulate(self, position_channel: int, action_channel: int, boundary_penalty: bool = False):
        """
        Deliver stimulation to designated input channels.
        
        Args:
            position_channel (int): Active spatial channel index (0 to 63).
            action_channel (int): Active action channel index (0 to 3).
            boundary_penalty (bool): Trigger boundary penalty burst.
        """
        self.active_position = position_channel
        self.active_action = action_channel
        self.active_boundary_penalty = boundary_penalty

    def read_frames(self, frame_count: int = 5) -> np.ndarray:
        """
        Simulate the continuous-time dynamical system (LIF SNN) for a set number of frames
        and return the firing frequencies of all 133 electrodes.
        """
        micro_steps_per_frame = 10
        frames = []
        
        # If boundary penalty is active, structurally decay the active pathways (LTD Pruning)
        if self.active_boundary_penalty:
            self.weights[self.active_position, :] *= 0.3
            self.weights[64 + self.active_action, :] *= 0.3
            self._normalize_weights()
        
        for _ in range(frame_count):
            frame_data = np.zeros(self.num_electrodes, dtype=np.float32)
            
            # Map input stimulation directly onto sensory/motor electrodes
            frame_data[self.active_position] = 1.0
            frame_data[64 + self.active_action] = 1.0
            if self.active_boundary_penalty:
                frame_data[132] = 1.0  # Activate boundary penalty channel
                
            output_spikes = np.zeros(self.num_outputs, dtype=np.float32)
            
            # Simulate continuous neural dynamics in micro-steps
            for _ in range(micro_steps_per_frame):
                inputs = np.zeros(self.num_inputs, dtype=np.float32)
                inputs[self.active_position] = 1.0
                inputs[64 + self.active_action] = 1.0
                
                # Injected current
                current = np.dot(inputs, self.weights)
                
                # Chaotic Entropy Burst if boundary penalty is triggered
                if self.active_boundary_penalty:
                    # High-frequency, chaotic burst noise
                    noise = np.random.normal(0.0, 0.5, self.num_outputs)
                else:
                    noise = np.random.normal(0.0, 0.08, self.num_outputs)
                
                # Update membrane potential
                self.v = self.v - self.leak * (self.v - self.v_rest) + current + noise
                
                # Determine spikes
                spiked = self.v >= self.v_thresh
                output_spikes[spiked] += 1.0
                self.v[spiked] = self.v_reset
                
            # Compute average firing frequency
            frame_data[68:132] = output_spikes / float(micro_steps_per_frame)
            frames.append(frame_data)
            
        return np.array(frames, dtype=np.float32)

    def update_weights(self, z_curr: np.ndarray, action: int, z_next: np.ndarray, lr: float = 0.08):
        """
        Applies online predictive Hebbian learning to the biological synaptic weights.
        
        Associates the current active spatial input state and action with the observed
        next spatial state.
        """
        # Formulate pre-synaptic input vector
        inputs = np.zeros(self.num_inputs, dtype=np.float32)
        inputs[:64] = z_curr
        inputs[64 + action] = 1.0
        
        # Outer product to find correlation
        dw = lr * np.outer(inputs, z_next)
        self.weights += dw
        
        # Clip weights to prevent negative connections and normalize them
        self.weights = np.clip(self.weights, 0.0, 5.0)
        self._normalize_weights()

    def _normalize_weights(self):
        # L2-normalization along columns (inputs pointing to each output neuron)
        col_norms = np.linalg.norm(self.weights, axis=0, keepdims=True) + 1e-6
        self.weights = self.weights / col_norms

@contextlib.contextmanager
def open():
    """
    Simulate opening the Cortical Labs DishBrain connection.
    """
    brain = MockDishBrain()
    try:
        yield brain
    finally:
        pass

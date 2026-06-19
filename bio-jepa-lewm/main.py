import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# Change working directory to the directory of main.py to resolve relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir:
    os.chdir(script_dir)

from src.eval_harness import train_and_evaluate_config

def load_config(config_path="config/bio_lewm_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_git_init():
    """
    Initializes a local git repository if not already initialized.
    """
    if not os.path.exists(".git"):
        print("\nInitializing local git repository...")
        try:
            subprocess.run(["git", "init"], check=True)
            
            # Create .gitignore
            with open(".gitignore", "w") as f:
                f.write(".venv/\n__pycache__/\n*.pyc\ncheckpoints/\n*.pt\n*.npz\n")
            print("Created .gitignore")
            
            # Create README.md
            with open("README.md", "w") as f:
                f.write("# Bio-JEPA LeWM Production Repository\n\n"
                        "This repository implements the Joint Embedding World Model (LeWM) stabilized by "
                        "Sketched-Isotropic-Gaussian Regularization (SIGReg) and integrated with simulated "
                        "DishBrain BNNs.\n\n"
                        "### Strict Validation Metrics\n"
                        "- **Horizon Failure Rate (HFR)**: % of rollouts where predicted endpoint exceeds 1.5 grid cells error.\n"
                        "- **Directional Failure Rate (DFR)**: % of rollouts with <= 0 cosine similarity to true path vector.\n"
                        "- **Trajectory Magnitude Error (TME)**: Track trajectory scale factor to catch path stunting.\n")
            print("Created README.md")
            
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit: Restructured production repository schema with strict failure metrics"], check=True)
            print("Git repository initialized and initial commit created successfully.")
        except Exception as e:
            print(f"Warning: Git initialization failed: {e}")

def main():
    # Load configuration
    config = load_config()
    
    # Run Git Initialization
    run_git_init()
    
    # Extract config parameters
    bnn_conf = config['bnn']
    train_conf = config['training']
    roll_conf = config['rollout']
    
    # Run CONFIG_A: Baseline Position-Only stack frame 1
    results_a = train_and_evaluate_config(
        in_channels=1,
        use_boundary_penalty=False,
        use_velocity_decoding=False,
        steps_train=train_conf['steps_train_baseline'],
        steps_eval=train_conf['steps_eval'],
        gain_factor=1.0,
        lr_dec_bias_scale=bnn_conf['lr_dec_bias_scale'],
        hebb_lr=bnn_conf['hebb_lr'],
        decay_factor=bnn_conf['decay_factor'],
        lr_dec=bnn_conf['lr_dec']
    )
    
    # Run CONFIG_B: Stacking Only stack frame 3
    results_b = train_and_evaluate_config(
        in_channels=3,
        use_boundary_penalty=False,
        use_velocity_decoding=False,
        steps_train=train_conf['steps_train_baseline'],
        steps_eval=train_conf['steps_eval'],
        gain_factor=1.0,
        lr_dec_bias_scale=bnn_conf['lr_dec_bias_scale'],
        hebb_lr=bnn_conf['hebb_lr'],
        decay_factor=bnn_conf['decay_factor'],
        lr_dec=bnn_conf['lr_dec']
    )
    
    # Run CONFIG_C: Full Stacking + BP + Velocity + Gain calibration (Chronic 20k steps)
    results_c = train_and_evaluate_config(
        in_channels=3,
        use_boundary_penalty=True,
        use_velocity_decoding=True,
        steps_train=train_conf['steps_train_chronic'],
        steps_eval=train_conf['steps_eval'],
        gain_factor=roll_conf['gain_factor'],
        lr_dec_bias_scale=bnn_conf['lr_dec_bias_scale'],
        hebb_lr=bnn_conf['hebb_lr'],
        decay_factor=bnn_conf['decay_factor'],
        lr_dec=bnn_conf['lr_dec'],
        checkpoint_interval=train_conf['checkpoint_interval']
    )
    
    # Serialize outputs
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
    
    # Save a duplicate copy to the artifacts folder
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
    print("\n" + "="*120)
    print("COMPARATIVE EVALUATION ANALYSIS:")
    print("="*120)
    print(f"{'Metric / Configuration':<35} | {'CONFIG_A (Baseline)':<22} | {'CONFIG_B (Stacking)':<22} | {'CONFIG_C (LeWM Full)':<22}")
    print("-"*120)
    print(f"{'MSPE at step h=1':<35} | {results_a['mspe'][0]:.6f}               | {results_b['mspe'][0]:.6f}               | {results_c['mspe'][0]:.6f}")
    print(f"{'MSPE at step h=4':<35} | {results_a['mspe'][3]:.6f}               | {results_b['mspe'][3]:.6f}               | {results_c['mspe'][3]:.6f}")
    print(f"{'MSPE at step h=8':<35} | {results_a['mspe'][7]:.6f}               | {results_b['mspe'][7]:.6f}               | {results_c['mspe'][7]:.6f}")
    print(f"{'Trajectory Variance Decay (TVD)':<35} | {results_a['tvd']:.6f}               | {results_b['tvd']:.6f}               | {results_c['tvd']:.6f}")
    print(f"{'Effective Prediction Horizon (EPH)':<35} | {results_a['eph']:d} steps                | {results_b['eph']:d} steps                | {results_c['eph']:d} steps")
    print(f"{'Horizon Failure Rate (HFR)':<35} | {results_a['hfr']:.2f}%               | {results_b['hfr']:.2f}%               | {results_c['hfr']:.2f}%")
    print(f"{'Directional Failure Rate (DFR)':<35} | {results_a['dfr']:.2f}%               | {results_b['dfr']:.2f}%               | {results_c['dfr']:.2f}%")
    print(f"{'Trajectory Magnitude Ratio (TME)':<35} | {results_a['tme']:.4f}               | {results_b['tme']:.4f}               | {results_c['tme']:.4f}")
    print("="*120)
    
    # Save comparative line plot
    plt.figure(figsize=(10, 6))
    horizon = np.arange(1, 9)
    plt.plot(horizon, results_a['mspe'], 'ro-', linewidth=2, markersize=6, label="CONFIG_A: Baseline Position-Only")
    plt.plot(horizon, results_b['mspe'], 'bs-', linewidth=2, markersize=6, label="CONFIG_B: Stacking Only")
    plt.plot(horizon, results_c['mspe'], 'g^-', linewidth=2, markersize=6, label="CONFIG_C: Temporal Stacking + BP + Velocity + Gain")
    plt.plot(horizon, results_a['mspe_baseline'], 'k--', label="Unmoving Baseline Guess")
    plt.title("Autoregressive Multi-Step Prediction Error (MSPE) Comparison", fontsize=12, fontweight='bold')
    plt.xlabel("Horizon Step (h)")
    plt.ylabel("Mean Squared Error (MSPE)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper left")
    
    plot_path_ws = "evaluation_comparison.png"
    plt.savefig(plot_path_ws, dpi=150)
    print(f"\nSaved diagnostic line plot to {plot_path_ws}")
    
    if os.path.exists(artifacts_dir):
        plot_path_art = os.path.join(artifacts_dir, "evaluation_comparison.png")
        plt.savefig(plot_path_art, dpi=150)
        print(f"Saved duplicate plot image artifact to {plot_path_art}")

if __name__ == "__main__":
    main()

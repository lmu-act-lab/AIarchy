"""
Script to load a checkpoint and continue training or run visualizations.
"""

import argparse
import pickle
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training_environment import TrainingEnvironment
from src.cla import CausalLearningAgent


def load_checkpoint(checkpoint_file_or_dir: str, base_name: str | None = None) -> list[CausalLearningAgent]:
    """
    Load checkpoint(s) from a pickle file or directory.
    
    If a directory is provided, finds the latest checkpoint for each agent.
    If base_name is provided with a directory, uses it to find agent folders.
    
    Parameters
    ----------
    checkpoint_file_or_dir : str
        Path to a checkpoint pickle file OR directory containing agent checkpoints.
    base_name : str | None, optional
        Base name for agent folders (e.g., "Amy"). Only needed when loading from directory.
        
    Returns
    -------
    list[CausalLearningAgent]
        The loaded agent(s).
    """
    if os.path.isdir(checkpoint_file_or_dir):
        # Load from directory - find latest checkpoints for each agent
        if not base_name:
            # Try to infer from parent directory (checkpoint dir is usually named "checkpoints")
            # e.g., pub_sims/Amy/checkpoints -> Amy
            checkpoint_dir = checkpoint_file_or_dir.rstrip(os.sep)
            parent_dir = os.path.dirname(checkpoint_dir)
            if parent_dir and os.path.basename(checkpoint_dir) == "checkpoints":
                base_name = os.path.basename(parent_dir)
            else:
                # Fallback: try to find agent folders and infer from first one
                try:
                    dir_contents = os.listdir(checkpoint_dir)
                    agent_dirs = [d for d in dir_contents 
                                 if os.path.isdir(os.path.join(checkpoint_dir, d)) 
                                 and '_' in d]
                    if agent_dirs:
                        # Extract base name from first agent folder (e.g., "Amy_0000" -> "Amy")
                        base_name = agent_dirs[0].split('_')[0]
                except:
                    pass
        
        if not base_name:
            raise ValueError(f"base_name must be provided when loading from a directory. Could not infer from: {checkpoint_file_or_dir}")
        
        print(f"📂 Loading checkpoints from directory: {checkpoint_file_or_dir}")
        checkpoint_files = TrainingEnvironment.find_latest_checkpoint_dir(checkpoint_file_or_dir, base_name)
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_file_or_dir} with base_name {base_name}")
        
        print(f"   Found {len(checkpoint_files)} agent checkpoint(s)")
        agents = []
        for checkpoint_file in checkpoint_files:
            with open(checkpoint_file, "rb") as f:
                agent = pickle.load(f)
                agents.append(agent)
        
        print(f"✅ Loaded {len(agents)} agents")
        return agents
    else:
        # Load from single file
        if not os.path.exists(checkpoint_file_or_dir):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file_or_dir}")
        
        print(f"📂 Loading checkpoint from: {checkpoint_file_or_dir}")
        with open(checkpoint_file_or_dir, "rb") as f:
            loaded = pickle.load(f)
        
        if isinstance(loaded, list):
            print(f"✅ Loaded {len(loaded)} agents")
            return loaded
        else:
            print("✅ Loaded 1 agent")
            return [loaded]


def continue_training(
    checkpoint_file_or_dir: str,
    additional_iterations: int,
    output_dir: str | None = None,
    checkpoint_interval: int | None = None,
    base_name: str | None = None,
) -> None:
    """
    Load a checkpoint and continue training.
    
    Parameters
    ----------
    checkpoint_file_or_dir : str
        Path to checkpoint file or directory containing checkpoints.
    additional_iterations : int
        Number of additional iterations to train.
    output_dir : str | None, optional
        Directory to save outputs and new checkpoints. If None, uses checkpoint directory.
    checkpoint_interval : int | None, optional
        Save checkpoints every N iterations during continued training.
    base_name : str | None, optional
        Base name for agent folders. Only needed when loading from directory.
    """
    agents = load_checkpoint(checkpoint_file_or_dir, base_name=base_name)
    
    # Determine current iteration from agent memory
    current_iteration = len(agents[0].memory) if agents[0].memory else 0
    final_iteration = current_iteration + additional_iterations
    
    # Determine output directory and base name
    if os.path.isdir(checkpoint_file_or_dir):
        checkpoint_dir = checkpoint_file_or_dir
        if not base_name:
            # Infer from parent directory (checkpoint dir is usually named "checkpoints")
            checkpoint_dir_clean = checkpoint_dir.rstrip(os.sep)
            parent_dir = os.path.dirname(checkpoint_dir_clean)
            if parent_dir and os.path.basename(checkpoint_dir_clean) == "checkpoints":
                base_name = os.path.basename(parent_dir)
            else:
                # Fallback: try to find agent folders and infer from first one
                try:
                    dir_contents = os.listdir(checkpoint_dir)
                    agent_dirs = [d for d in dir_contents 
                                 if os.path.isdir(os.path.join(checkpoint_dir, d)) 
                                 and '_' in d]
                    if agent_dirs:
                        base_name = agent_dirs[0].split('_')[0]
                except:
                    pass
    else:
        checkpoint_dir = os.path.dirname(checkpoint_file_or_dir)
        if not base_name:
            # Extract from filename
            filename = os.path.basename(checkpoint_file_or_dir).replace(".pkl", "")
            # Remove iteration suffix if present
            if "_iter_" in filename:
                base_name = filename.split("_iter_")[0]
            else:
                base_name = filename.split("_")[0] if "_" in filename else filename
    
    if output_dir is None:
        output_dir = checkpoint_dir  # Save in same directory
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎓 Continuing training for {additional_iterations} iterations...")
    print(f"   Starting from iteration: {current_iteration}")
    print(f"   Will train until iteration: {final_iteration}")
    print(f"   Output directory: {output_dir}")
    
    # Create training environment and continue training
    te = TrainingEnvironment()
    te.train(
        mc_rep=additional_iterations,
        style="SA",
        agents=agents,
        checkpoint_dir=output_dir,
        checkpoint_interval=checkpoint_interval,
        agent_name=base_name,
    )
    
    print(f"\n✅ Training completed. Agents saved to: {output_dir}")


def run_visualizations(
    checkpoint_file_or_dir: str,
    output_dir: str | None = None,
    show_params: bool = True,
    base_name: str | None = None,
) -> None:
    """
    Load a checkpoint and run all visualizations.
    
    Parameters
    ----------
    checkpoint_file_or_dir : str
        Path to checkpoint file or directory containing checkpoints.
    output_dir : str | None, optional
        Directory to save visualizations. If None, auto-generates based on checkpoint iteration.
    show_params : bool, optional
        Whether to show parameters in plots. Defaults to True.
    base_name : str | None, optional
        Base name for agent folders. Only needed when loading from directory.
    """
    agents = load_checkpoint(checkpoint_file_or_dir, base_name=base_name)
    
    # Determine current iteration from agent memory
    current_iteration = len(agents[0].memory) if agents[0].memory else 0
    
    # Determine output directory with iteration number
    if output_dir is None:
        if os.path.isdir(checkpoint_file_or_dir):
            checkpoint_dir = checkpoint_file_or_dir
        else:
            checkpoint_dir = os.path.dirname(checkpoint_file_or_dir)
        
        output_dir = os.path.join(checkpoint_dir, f"visualizations_iter_{current_iteration:06d}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Running visualizations...")
    print(f"   Checkpoint iteration: {current_iteration}")
    print(f"   Output directory: {output_dir}")
    
    te = TrainingEnvironment()
    
    # Run all visualizations
    if agents:
        # Use first agent for pre-training viz (may not be accurate if loaded mid-training)
        if hasattr(agents[0], "original_model"):
            te.pre_training_visualization(agents[0], name=output_dir, save=True)
        
        te.post_training_visualization(agents, name=output_dir, save=True)
        te.show_cpt_changes(agents, name=output_dir, save=True)
        te.plot_monte_carlo(agents, show_params=show_params, name=output_dir, save=True)
        te.plot_weighted_rewards(agents, show_params=show_params, name=output_dir, save=True)
        te.plot_u_hat_model_losses(agents, name=output_dir, save=True)
        te.save_reward_csv(agents, name=output_dir)
    
    print(f"\n✅ Visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Load a checkpoint and continue training or run visualizations."
    )
    parser.add_argument(
        "checkpoint_file_or_dir",
        type=str,
        help="Path to checkpoint .pkl file OR directory containing agent checkpoints",
    )
    parser.add_argument(
        "--continue-training",
        type=int,
        metavar="ITERATIONS",
        help="Continue training for N additional iterations",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Run visualizations on the checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (default: checkpoint directory)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        help="Save checkpoints every N iterations during continued training",
    )
    parser.add_argument(
        "--base-name",
        type=str,
        help="Base name for agent folders (e.g., 'Amy'). Only needed when loading from directory.",
    )
    parser.add_argument(
        "--no-params",
        action="store_true",
        help="Don't show parameters in visualizations",
    )
    
    args = parser.parse_args()
    
    if args.continue_training:
        continue_training(
            args.checkpoint_file_or_dir,
            args.continue_training,
            args.output_dir,
            args.checkpoint_interval,
            base_name=args.base_name,
        )
    
    if args.visualize:
        run_visualizations(
            args.checkpoint_file_or_dir,
            args.output_dir,
            show_params=not args.no_params,
            base_name=args.base_name,
        )
    
    if not args.continue_training and not args.visualize:
        print("⚠️  No action specified. Use --continue-training or --visualize")
        print("\nExample usage:")
        print("  python -m src.load_checkpoint checkpoint_dir --visualize")
        print("  python -m src.load_checkpoint checkpoint_dir --continue-training 100")
        print("  python -m src.load_checkpoint checkpoint_dir --continue-training 100 --checkpoint-interval 25")


if __name__ == "__main__":
    main()

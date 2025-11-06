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
from src.hierarchy import Hierarchy
from src.teacher import build_classroom_hierarchy


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


def load_hierarchy_checkpoint(
    checkpoint_dir: str,
    child_names: list[str] | None = None,
    parent_name: str = "Teacher",
) -> Hierarchy:
    """
    Load a hierarchy from checkpoint directory structure.
    
    The checkpoint directory should have the structure:
        checkpoint_dir/
            children/
                {child_name}/
                    checkpoints/
                        {child_name}_{agent_idx:04d}/
                            iter_{iteration:06d}.pkl
            parents/
                {parent_name}/
                    checkpoints/
                        {parent_name}_{agent_idx:04d}/
                            iter_{iteration:06d}.pkl
    
    Parameters
    ----------
    checkpoint_dir : str
        Base checkpoint directory containing children/ and parents/ subdirectories.
    child_names : list[str] | None, optional
        Names of child groups. If None, will try to infer from directory structure.
    parent_name : str, optional
        Name of parent group. Defaults to "Teacher".
        
    Returns
    -------
    Hierarchy
        The loaded hierarchy with all agents restored.
    """
    checkpoint_dir = checkpoint_dir.rstrip(os.sep)
    
    children_dir = os.path.join(checkpoint_dir, "children")
    parents_dir = os.path.join(checkpoint_dir, "parents")
    
    if not os.path.exists(children_dir):
        raise FileNotFoundError(f"Children checkpoint directory not found: {children_dir}")
    if not os.path.exists(parents_dir):
        raise FileNotFoundError(f"Parents checkpoint directory not found: {parents_dir}")
    
    print(f"📂 Loading hierarchy from: {checkpoint_dir}")
    
    # Load children
    if child_names is None:
        # Try to infer child names from directory structure
        child_names = sorted([
            d for d in os.listdir(children_dir)
            if os.path.isdir(os.path.join(children_dir, d))
        ])
    
    print(f"   Found {len(child_names)} child group(s)")
    loaded_children = []
    for child_name in child_names:
        child_checkpoint_dir = os.path.join(children_dir, child_name, "checkpoints")
        if not os.path.exists(child_checkpoint_dir):
            raise FileNotFoundError(f"Child checkpoint directory not found: {child_checkpoint_dir}")
        
        print(f"   Loading child group: {child_name}")
        child_agents = load_checkpoint(child_checkpoint_dir, base_name=child_name)
        loaded_children.append(child_agents)
    
    # Load parents
    print(f"   Loading parent group: {parent_name}")
    parent_checkpoint_dir = os.path.join(parents_dir, parent_name, "checkpoints")
    if not os.path.exists(parent_checkpoint_dir):
        raise FileNotFoundError(f"Parent checkpoint directory not found: {parent_checkpoint_dir}")
    
    parent_agents = load_checkpoint(parent_checkpoint_dir, base_name=parent_name)
    
    # Determine hierarchy configuration from loaded agents
    # We need to infer pooling_vars, glue_vars, and monte_carlo_samples
    if not loaded_children or not parent_agents:
        raise ValueError("Could not load any agents from checkpoint")
    
    # Get configuration from first agents
    first_child = loaded_children[0][0]
    first_parent = parent_agents[0]
    
    # Infer pooling vars (utility vars that are likely pooled)
    pooling_vars = list(first_child.utility_vars)  # Default: all utility vars
    
    # Infer glue vars (reflective vars that are in both parent and child)
    glue_vars = list(
        first_parent.reflective_vars & first_child.glue_vars
    )
    
    # Get monte carlo samples count
    monte_carlo_samples = len(parent_agents)
    
    print(f"   Detected configuration:")
    print(f"     Monte Carlo samples: {monte_carlo_samples}")
    print(f"     Pooling variables: {pooling_vars}")
    print(f"     Glue variables: {glue_vars}")
    
    # Reconstruct the hierarchy structure
    # We need to create a mapping from parent to children
    # Since we only have one parent group, we map it to all child groups
    parent_template = first_parent
    child_templates = [children[0] for children in loaded_children]
    
    # Create a new hierarchy with the same structure
    # Note: We can't directly instantiate Hierarchy with loaded agents, so we need to
    # reconstruct it using the original structure but then replace the agents
    agent_parents = {parent_template: child_templates}
    
    hier = Hierarchy(
        agent_parents,
        pooling_vars=pooling_vars,
        glue_vars=glue_vars,
        monte_carlo_samples=monte_carlo_samples,
    )
    
    # Replace the newly created agents with our loaded ones
    hier.parents = [parent_agents]
    hier.children = loaded_children
    
    print(f"✅ Loaded hierarchy with {len(hier.parents)} parent group(s) and {len(hier.children)} child group(s)")
    return hier


def continue_hierarchical_training(
    checkpoint_dir: str,
    additional_cycles: int,
    parent_iter: int = 1,
    child_iter: int = 5,
    output_dir: str | None = None,
    child_checkpoint_interval: int | None = None,
    parent_checkpoint_interval: int | None = None,
    child_names: list[str] | None = None,
    parent_name: str = "Teacher",
) -> None:
    """
    Load a hierarchy checkpoint and continue training.
    
    Parameters
    ----------
    checkpoint_dir : str
        Base checkpoint directory containing children/ and parents/ subdirectories.
    additional_cycles : int
        Number of additional training cycles to run.
    parent_iter : int, optional
        Number of iterations to train parents per cycle. Defaults to 1.
    child_iter : int, optional
        Number of iterations to train children per cycle. Defaults to 5.
    output_dir : str | None, optional
        Directory to save outputs and new checkpoints. If None, uses checkpoint directory.
    child_checkpoint_interval : int | None, optional
        Save child checkpoints every N iterations during continued training.
    parent_checkpoint_interval : int | None, optional
        Save parent checkpoints every N iterations during continued training.
    child_names : list[str] | None, optional
        Names of child groups. If None, will try to infer from directory structure.
    parent_name : str, optional
        Name of parent group. Defaults to "Teacher".
    """
    hier = load_hierarchy_checkpoint(checkpoint_dir, child_names=child_names, parent_name=parent_name)
    
    # Determine current cycle from agent memory
    # We'll use the first child's memory length as a proxy
    current_cycle = 0
    if hier.children and hier.children[0] and hier.children[0][0].memory:
        # Rough estimate: each cycle has child_iter + parent_iter iterations
        # This is approximate since we don't track cycles directly
        total_iterations = len(hier.children[0][0].memory)
        estimated_cycles = total_iterations // (child_iter + parent_iter)
        current_cycle = estimated_cycles
    
    if output_dir is None:
        output_dir = checkpoint_dir
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎓 Continuing hierarchical training for {additional_cycles} cycles...")
    print(f"   Estimated current cycle: {current_cycle}")
    print(f"   Will train for {additional_cycles} more cycles")
    print(f"   Output directory: {output_dir}")
    
    # Continue training
    hier.train(
        cycles=additional_cycles,
        parent_iter=parent_iter,
        child_iter=child_iter,
        checkpoint_dir=output_dir,
        child_checkpoint_interval=child_checkpoint_interval,
        parent_checkpoint_interval=parent_checkpoint_interval,
        child_names=child_names,
        parent_name=parent_name,
    )
    
    print(f"\n✅ Hierarchical training completed. Agents saved to: {output_dir}")


def run_hierarchical_visualizations(
    checkpoint_dir: str,
    output_dir: str | None = None,
    show_params: bool = True,
    child_names: list[str] | None = None,
    parent_name: str = "Teacher",
) -> None:
    """
    Load a hierarchy checkpoint and run all visualizations.
    
    Parameters
    ----------
    checkpoint_dir : str
        Base checkpoint directory containing children/ and parents/ subdirectories.
    output_dir : str | None, optional
        Directory to save visualizations. If None, auto-generates based on checkpoint iteration.
    show_params : bool, optional
        Whether to show parameters in plots. Defaults to True.
    child_names : list[str] | None, optional
        Names of child groups. If None, will try to infer from directory structure.
    parent_name : str, optional
        Name of parent group. Defaults to "Teacher".
    """
    hier = load_hierarchy_checkpoint(checkpoint_dir, child_names=child_names, parent_name=parent_name)
    
    # Determine current iteration from agent memory
    current_iteration = 0
    if hier.children and hier.children[0] and hier.children[0][0].memory:
        current_iteration = len(hier.children[0][0].memory)
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join(checkpoint_dir, f"visualizations_iter_{current_iteration:06d}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Running hierarchical visualizations...")
    print(f"   Checkpoint iteration: {current_iteration}")
    print(f"   Output directory: {output_dir}")
    
    te = TrainingEnvironment()
    
    # Parent visualizations
    if hier.parents:
        parent_agents = hier.parents[0]
        parent_dir = os.path.join(output_dir, parent_name)
        Path(parent_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"   Generating parent visualizations...")
        if hasattr(parent_agents[0], "original_model"):
            te.pre_training_visualization(parent_agents[0], name=parent_dir, save=True)
        te.post_training_visualization(parent_agents, name=parent_dir, save=True)
        te.show_cpt_changes(parent_agents, name=parent_dir, save=True)
        te.plot_monte_carlo(parent_agents, show_params=show_params, name=parent_dir, save=True)
        te.plot_weighted_rewards(parent_agents, show_params=show_params, name=parent_dir, save=True)
        te.plot_u_hat_model_losses(parent_agents, name=parent_dir, save=True)
        te.save_reward_csv(parent_agents, name=parent_dir)
    
    # Child visualizations
    if hier.children:
        if child_names is None:
            child_names = [f"Child_{i}" for i in range(len(hier.children))]
        
        for i, child_agents in enumerate(hier.children):
            child_name = child_names[i] if i < len(child_names) else f"Child_{i}"
            child_dir = os.path.join(output_dir, child_name)
            Path(child_dir).mkdir(parents=True, exist_ok=True)
            
            print(f"   Generating visualizations for {child_name}...")
            if hasattr(child_agents[0], "original_model"):
                te.pre_training_visualization(child_agents[0], name=child_dir, save=True)
            te.post_training_visualization(child_agents, name=child_dir, save=True)
            te.show_cpt_changes(child_agents, name=child_dir, save=True)
            te.plot_monte_carlo(child_agents, show_params=show_params, name=child_dir, save=True)
            te.plot_weighted_rewards(child_agents, show_params=show_params, name=child_dir, save=True)
            te.plot_u_hat_model_losses(child_agents, name=child_dir, save=True)
            te.save_reward_csv(child_agents, name=child_dir)
    
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
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Load/resume hierarchical training (requires checkpoint directory with children/ and parents/ subdirectories)",
    )
    parser.add_argument(
        "--child-names",
        type=str,
        nargs="+",
        help="Names of child groups (for hierarchical training). If not provided, will infer from directory structure.",
    )
    parser.add_argument(
        "--parent-name",
        type=str,
        default="Teacher",
        help="Name of parent group (for hierarchical training). Defaults to 'Teacher'.",
    )
    parser.add_argument(
        "--parent-iter",
        type=int,
        default=1,
        help="Number of iterations to train parents per cycle (for hierarchical training). Defaults to 1.",
    )
    parser.add_argument(
        "--child-iter",
        type=int,
        default=5,
        help="Number of iterations to train children per cycle (for hierarchical training). Defaults to 5.",
    )
    
    args = parser.parse_args()
    
    if args.hierarchical:
        # Hierarchical training mode
        if args.continue_training:
            continue_hierarchical_training(
                args.checkpoint_file_or_dir,
                args.continue_training,  # This is cycles for hierarchical training
                parent_iter=args.parent_iter,
                child_iter=args.child_iter,
                output_dir=args.output_dir,
                child_checkpoint_interval=args.checkpoint_interval,
                parent_checkpoint_interval=args.checkpoint_interval,
                child_names=args.child_names,
                parent_name=args.parent_name,
            )
        
        if args.visualize:
            run_hierarchical_visualizations(
                args.checkpoint_file_or_dir,
                args.output_dir,
                show_params=not args.no_params,
                child_names=args.child_names,
                parent_name=args.parent_name,
            )
        
        if not args.continue_training and not args.visualize:
            print("⚠️  No action specified. Use --continue-training or --visualize")
            print("\nExample usage (hierarchical):")
            print("  python -m src.load_checkpoint checkpoint_dir --hierarchical --visualize")
            print("  python -m src.load_checkpoint checkpoint_dir --hierarchical --continue-training 10")
            print("  python -m src.load_checkpoint checkpoint_dir --hierarchical --continue-training 10 --checkpoint-interval 5")
    else:
        # Standard (non-hierarchical) training mode
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
            print("\nExample usage (standard):")
            print("  python -m src.load_checkpoint checkpoint_dir --visualize")
            print("  python -m src.load_checkpoint checkpoint_dir --continue-training 100")
            print("  python -m src.load_checkpoint checkpoint_dir --continue-training 100 --checkpoint-interval 25")
            print("\nExample usage (hierarchical):")
            print("  python -m src.load_checkpoint checkpoint_dir --hierarchical --visualize")
            print("  python -m src.load_checkpoint checkpoint_dir --hierarchical --continue-training 10")


if __name__ == "__main__":
    main()

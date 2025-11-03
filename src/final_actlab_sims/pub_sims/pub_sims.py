from src.final_actlab_sims.models import base_structs, students
from src.training_environment import TrainingEnvironment
import copy
import logging
import os
from src.final_actlab_sims.set_seed import set_seed

# Set up environment
set_seed()
logging.disable(logging.WARNING)
testing_environment: TrainingEnvironment = TrainingEnvironment()

# Suppress pgmpy warnings
logging.getLogger("pgmpy").setLevel(logging.ERROR)

def run_simulation(agent, name, iterations, monte_carlo_samples=100, show_params=True, 
                  checkpoint_dir=None, checkpoint_interval=None):
    """Run a complete simulation for a given agent."""
    print(f"Running simulation for {name}...")
    
    # Create output directory
    os.makedirs(f"pub_sims/{name}", exist_ok=True)
    
    # Pre-training visualization
    testing_environment.pre_training_visualization(agent, name=f"pub_sims/{name}", save=True)
    
    # Create Monte Carlo agents
    monte_carlo_agents = [copy.deepcopy(agent) for _ in range(monte_carlo_samples)]
    
    # Train the agents (with checkpointing if enabled)
    testing_environment.train(
        iterations, 
        "SA", 
        monte_carlo_agents,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        agent_name=name
    )
    
    # Post-training visualizations
    testing_environment.post_training_visualization(monte_carlo_agents, name=f"pub_sims/{name}", save=True)
    testing_environment.show_cpt_changes(monte_carlo_agents, name=f"pub_sims/{name}", save=True)
    testing_environment.plot_monte_carlo(monte_carlo_agents, show_params=show_params, name=f"pub_sims/{name}", save=True)
    testing_environment.plot_weighted_rewards(monte_carlo_agents, show_params=show_params, name=f"pub_sims/{name}", save=True)
    testing_environment.plot_u_hat_model_losses(monte_carlo_agents, name=f"pub_sims/{name}", save=True)
    testing_environment.save_reward_csv(monte_carlo_agents, name=f"pub_sims/{name}")
    
    print(f"Completed simulation for {name}")

def main():
    """Run all simulations as specified."""
    
    # Create main output directory
    os.makedirs("pub_sims", exist_ok=True)
    
    print("Starting comprehensive simulation run...")
    print("=" * 50)
    
    # 1. Train base structures for 2000 iterations with 100 Monte Carlo agents
    print("\n1. Training Base Structures (2000 iterations, 100 Monte Carlo agents)")
    print("-" * 60)
    
    base_structures_to_train = {
        "confound_obs": base_structs["confound_obs"],
        "confound_hidden": base_structs["confound_hidden"], 
        "downweigh_struct": base_structs["downweigh_struct"],
        "mediation_obs": base_structs["mediation_obs"],
        "mediation_hidden": base_structs["mediation_hidden"]
    }
    
    # for name, agent in base_structures_to_train.items():
    #     run_simulation(agent, name, iterations=2000, monte_carlo_samples=100, show_params=True)
    
    # 2. Train students (Amy and Leo) for 3000 iterations with 100 Monte Carlo agents
    print("\n2. Training Students (3000 iterations, 100 Monte Carlo agents)")
    print("-" * 60)
    
    students_to_train = {
        "Amy": students["Amy"],
        # "Leo": students["Leo"],
        # "Isla": students["Isla"],
        # "Mei": students["Mei"],
        # "Jonas": students["Jonas"],
        # "Nikhil": students["Nikhil"]
    }
    
    for name, agent in students_to_train.items():
        # Enable checkpointing for Amy (every 500 iterations)
        checkpoint_dir = f"pub_sims/{name}/checkpoints" if name == "Amy" else None
        checkpoint_interval = 500 if name == "Amy" else None
        
        run_simulation(
            agent, 
            name, 
            iterations=3000, 
            monte_carlo_samples=1,  # Restore to 100 for actual runs
            show_params=True,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval
        )
    
    print("\n" + "=" * 50)
    print("All simulations completed successfully!")
    print("Results saved in pub_sims/ directory")

if __name__ == "__main__":
    main()

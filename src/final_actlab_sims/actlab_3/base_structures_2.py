from src.final_actlab_sims.models import all_structs
from src.cla import CausalLearningAgent
from src.training_environment import TrainingEnvironment
from src.final_actlab_sims.set_seed import set_seed
import logging
import os
import copy
logging.disable(logging.WARNING)
set_seed()
testing_environment: TrainingEnvironment = TrainingEnvironment()

# unfinished = ["downweigh_struct_samples32", "downweigh_struct_fastAdapt", "downweigh_struct_highExplore"]
for name, structure in all_structs.items():
    os.makedirs(f"{name}", exist_ok=True)
    # if name not in unfinished:
    #     print(f"Already finished {name}.")
    #     continue
    testing_environment.pre_training_visualization(structure, name=name, save=True)
    structure_agents = [copy.deepcopy(structure) for agent in range(100)]
    testing_environment.train(2000, "SA", structure_agents)
    testing_environment.post_training_visualization(structure_agents, name=name, save=True)
    testing_environment.show_cpt_changes(structure_agents, name=name,  save=True)
    testing_environment.plot_monte_carlo(structure_agents, show_params=True, name=name, save=True)
    testing_environment.plot_weighted_rewards(structure_agents, show_params=True, name=name, save=True)
    testing_environment.plot_u_hat_model_losses(structure_agents, name=name, save=True)
    testing_environment.save_reward_csv(structure_agents, name=name)
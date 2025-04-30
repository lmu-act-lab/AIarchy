from src.final_actlab_sims.models import res2
from src.cla import CausalLearningAgent
from src.training_environment import TrainingEnvironment
import logging
import os
import copy
logging.disable(logging.WARNING)
testing_environment: TrainingEnvironment = TrainingEnvironment()


for name, structure in res2.items():
    os.makedirs(f"{name}", exist_ok=True)
    testing_environment.pre_training_visualization(structure, name=name, save=True)
    structure_agents = [copy.deepcopy(structure) for agent in range(10)]
    testing_environment.train(10, "SA", structure_agents)
    testing_environment.post_training_visualization(structure_agents, name=name, save=True)
    testing_environment.show_cpt_changes(structure_agents, name=name,  save=True)
    testing_environment.plot_monte_carlo(structure_agents, show_params=True, name=name, save=True)
    testing_environment.plot_weighted_rewards(structure_agents, show_params=True, name=name, save=True)
    testing_environment.plot_u_hat_model_losses(structure_agents, name=name, save=True)
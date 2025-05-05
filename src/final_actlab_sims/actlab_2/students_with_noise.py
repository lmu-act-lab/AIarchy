from src.final_actlab_sims.models import students_with_noise
from src.cla import CausalLearningAgent
from src.training_environment import TrainingEnvironment
import logging
import os
import copy
logging.disable(logging.WARNING)
testing_environment: TrainingEnvironment = TrainingEnvironment()


for name, student in students_with_noise.items():
    name += "_noise"
    os.makedirs(f"{name}", exist_ok=True)
    testing_environment.pre_training_visualization(student, name=name, save=True)
    student_agents = [copy.deepcopy(student) for agent in range(100)]
    testing_environment.train(2000, "SA", student_agents)
    testing_environment.post_training_visualization(student_agents, name=name, save=True)
    testing_environment.show_cpt_changes(student_agents, name=name,  save=True)
    testing_environment.plot_monte_carlo(student_agents, show_params=True, name=name, save=True)
    testing_environment.plot_weighted_rewards(student_agents, show_params=True, name=name, save=True)
    testing_environment.plot_u_hat_model_losses(student_agents, name=name, save=True)
    testing_environment.save_reward_csv(student_agents, name=name)

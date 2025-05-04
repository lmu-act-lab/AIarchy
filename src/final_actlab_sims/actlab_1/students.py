from src.final_actlab_sims.models import students, students_with_noise
from src.training_environment import TrainingEnvironment
import copy
import logging
import os
from src.final_actlab_sims.set_seed import set_seed
set_seed()
logging.disable(logging.WARNING)
testing_environment: TrainingEnvironment = TrainingEnvironment()


for name, student_agent in students.items():
    os.makedirs(f"{name}", exist_ok=True)
    testing_environment.pre_training_visualization(student_agent, name=name, save=True)
    student_agents = [copy.deepcopy(student_agent) for agent in range(1)]
    testing_environment.train(1, "SA", student_agents)
    testing_environment.post_training_visualization(student_agents, name=name, save=True)
    testing_environment.show_cpt_changes(student_agents, name=name,  save=True)
    testing_environment.plot_monte_carlo(student_agents, show_params=True, name=name, save=True)
    testing_environment.plot_weighted_rewards(student_agents, show_params=True, name=name, save=True)
    testing_environment.plot_u_hat_model_losses(student_agents, name=name, save=True)
    testing_environment.save_reward_csv(student_agents, name=name)
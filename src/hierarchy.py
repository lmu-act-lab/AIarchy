from src.cla import CausalLearningAgent as CLA
from src.training_environment import TrainingEnvironment
import copy

class Hierarchy:
    def __init__(self, agent_parents: dict[CLA, list[CLA]], pooling_vars: list[str], glue_vars: list[str], monte_carlo_samples: int = 100):
        """
        Initializes the hierarchy with a dictionary of agents and their parents.

        :param agent_parents: A dictionary where keys are parents and values are lists of children.
        """
        for pooling_var in pooling_vars:
            if pooling_var not in agent_parents[next(iter(agent_parents))][0].utility_vars:
                raise ValueError(f"Pooling variable {pooling_var} not found in utility variables of child agent.")
        for glue_var in glue_vars:
            if glue_var not in next(iter(agent_parents)).reflective_vars:
                raise ValueError(f"Glue variable {glue_var} not found in reflective variables of parent agent.")
        self.parents = []
        self.children = []
        for parent, children in agent_parents.items():
            self.parents.append(self.make_monte_carlo(parent, num_samples=monte_carlo_samples))
            for child in children:
                self.children.append(self.make_monte_carlo(child, num_samples=monte_carlo_samples))
        self.TE = TrainingEnvironment()
        self.monte_carlo_samples = monte_carlo_samples
        self.pooling_vars = pooling_vars
        self.glue_vars = glue_vars
        self.pooling_var_vals = {}
        self.glue_var_cpds = {}

    def make_monte_carlo(self, agent: CLA, num_samples: int = 1000):
        """
        Creates a Monte Carlo simulation for the given agent.

        :param agent: The agent to create the Monte Carlo simulation for.
        :param num_samples: The number of samples to generate.
        :return: A list of samples.
        """
        return [copy.deepcopy(agent) for _ in range(num_samples)]

    def pooling_func_update(self):
        for pooling_var in self.pooling_vars:
            aggregated_rewards = []
            for i in range(self.monte_carlo_samples):
                avg_reward = 0
                for children in self.children:
                    for timestep in children[i].memory:
                        avg_reward += timestep.average_reward[pooling_var]
                    avg_reward /= len(children[i].memory)
                avg_reward /= len(self.children)
                aggregated_rewards.append(avg_reward)
            self.pooling_var_vals[pooling_var] = aggregated_rewards

        for parent in self.parents:
            for i in range(self.monte_carlo_samples):
                parent[i].lower_tier_pooled_reward = {}
                for pooling_var in self.pooling_vars:
                    parent[i].lower_tier_pooled_reward[pooling_var] = self.pooling_var_vals[pooling_var][i]


    def glue_var_update(self):
        for glue_var in self.glue_vars:
            cpts = []
            for i in range(self.monte_carlo_samples):
                for parents in self.parents:
                    cpts.append(parents[i].sampling_model.get_cpds(glue_var))
            self.glue_var_cpds[glue_var] = cpts

        for children in self.children:
            for i in range(self.monte_carlo_samples):
                for glue_var in self.glue_vars:
                    children[i].sampling_model.remove_cpds(glue_var)
                    children[i].sampling_model.add_cpds(self.glue_var_cpds[glue_var][i])

    def train(self, cycles, parent_iter, child_iter):
        # First train children for child_iter
        # Pool utilities from memories of children (e.g. grades)
        # Parent must have modified reward function to include pooling function output
        # Then train parents for parent_iter
        # Then update glue variables on children from parent
        for _ in range(cycles):
            for children in self.children:
                self.TE.train(child_iter, "SA", children)
            self.pooling_func_update()
            for parents in self.parents:
                self.TE.train(parent_iter, "SA", parents)
            self.glue_var_update()

# TODO:
# 2. Teacher network (can be simple)
# 4. Testing some basic structures again (confounding, anything new?)
# 6. Organize simulations
# a) Hierarchy on one machine saving plots as it goes
# b) Separate models on separate machines to test different parameters (4 machines)
# c) Make sure to test eta of training on ACT lab machines
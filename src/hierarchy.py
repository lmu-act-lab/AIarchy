from cla import CausalLearningAgent as CLA
from training_environment import TrainingEnvironment

class Hierarchy:
    def __init__(self, agent_parents: dict[CLA, list[CLA]], pooling_vars: list[str], glue_vars: list[str]):
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
        self.agent_parents = agent_parents

    def pooling_func(self):
        pass

    def glue_var_update(self):
        pass

    def train(self, parent_iter, child_iter):
        # First train children for child_iter
        # Pool utilities from memories of children (e.g. grades)
        # Parent must have modified reward function to include pooling function output
        # Then train parents for parent_iter
        # Then update glue variables on children from parent

        pass
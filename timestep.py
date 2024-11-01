import pandas as pd

class TimeStep:
    def __init__(
        self,
        memory: pd.DataFrame,
        sample_vars: set[str],
        reward_vars: set[str],
        weights: dict[str, float],
        tweak_var: str,
    ):
        """
        Custom Timestep class that holds memory, average sample values,
        and average reward values for that timestep.

        Parameters
        ----------
        memory : pd.DataFrame
            All samples and rewards for that timestep.
        sample_vars : set[str]
            Sample variables.
        reward_vars : set[str]
            Utility/reward variables.
        weights : dict[str, float]
            Weights associated with rewards
        """
        self.tweak_var = tweak_var
        self.memory: pd.DataFrame = memory
        self.average_sample: dict[str, int] = {}
        self.average_reward: dict[str, float] = {}
        self.weights = weights

        for var in sample_vars:
            self.average_sample[var] = self.get_rounded_column_average(var)

        for var in reward_vars:
            self.average_reward[var] = self.get_column_average(var)

    def get_rounded_column_average(self, column_name: str) -> int:
        """
        Calculate and round the average of a specified column in the memory DataFrame.

        Parameters
        ----------
        column_name : str
            The name of the column to calculate the average for.
        decimal_places : int, optional
            The number of decimal places to round the average to, by default 2.

        Returns
        -------
        float
            The rounded average value of the specified column.
        """
        average = self.memory[column_name].mean()
        return round(average)

    def get_column_average(self, column_name: str) -> float:
        """
        Calculate and round the average of a specified column in the memory DataFrame.

        Parameters
        ----------
        column_name : str
            The name of the column to calculate the average for.

        Returns
        -------
        float
            The average value of the specified column.
        """
        average = self.memory[column_name].mean()
        return average

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TimeStep):
            return (
                self.memory.all().all() == other.memory.all().all()
                and self.average_sample == other.average_sample
                and self.average_reward == other.average_reward
            )
        return False

    def __hash__(self):
        # Hash sample_vars, reward_vars as sorted tuples to maintain order consistency
        sample_vars_hash = hash(tuple(sorted(self.average_sample.keys())))
        reward_vars_hash = hash(tuple(sorted(self.average_reward.keys())))
        weights_hash = hash(tuple(sorted(self.weights.items())))
        tweak_var_hash = hash(self.tweak_var)

        # Combine the hashes using XOR or a tuple for a unique combination
        return hash((sample_vars_hash, reward_vars_hash, weights_hash, tweak_var_hash))

from cla import CausalLearningAgent
from matplotlib import pyplot as plt
from util import Counter

class TrainingEnvironment():
    def train(
        self, 
        mc_rep: int, style: str, agents: list[CausalLearningAgent]
    ) -> list[CausalLearningAgent]:
        """
        Trains n agents for a specified number of monte carlo repetitions using a specified style.

        Parameters
        ----------
        mc_rep : int
            Monte Carlo Repetitions.
        style : str
            Simulated Annealing/Maximum Exploitation.
        agents : list[CausalLearningAgent]
            List of agents to train.
        """

        match style:
            case "SA":
                for agent in agents:
                    agent.train_SA(mc_rep)
            case "ME":
                for agent in agents:
                    agent.train_ME(mc_rep)
        return agents


    def plot_monte_carlo(self, agents: list[CausalLearningAgent], show_params: bool = False) -> None:
        """
        Plots the average reward across agents.

        Parameters
        ----------
        agents : list[CausalLearningAgent]
            List of agents to plot.
        """
        average_rewards: list[float] = []
        params:  dict[str, UnionType[float, int]] = agents[0].parameters
        for iteration in range(len(agents[0].memory)):
            total_reward = 0
            for agent in agents:
                total_reward += sum(agent.memory[iteration].average_reward.values())
            average_rewards.append(total_reward / len(agents))
        
        if show_params:
            # Adjust bottom space: less space between parameters, enough for plot
            num_params = len(params)
            # Adjust to ensure enough space for the plot and params
            bottom_margin = 0.1 + 0.04 * num_params

            plt.subplots_adjust(bottom=bottom_margin)

            # Dynamically place text under the graph using figure coordinates
            for i, (param, param_val) in enumerate(params.items()):
                # If the parameter is a dictionary, we format it for readability
                if isinstance(param_val, dict):
                    param_val_str = ", ".join([f"{k}: {v}" for k, v in param_val.items()])
                else:
                    param_val_str = str(param_val)

                # Reduced vertical space between parameters
                plt.figtext(
                    0.5,
                    bottom_margin - (0.03 * i) - 0.1,
                    f"{param}: {param_val_str}",
                    ha="center",
                    va="top",
                    wrap=True,
                    fontsize=10,
                    transform=plt.gcf().transFigure,
                )
        plt.plot(average_rewards)
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        plt.title(f"Average Reward Across {len(agents)} Agents")
        plt.show()
        
    def reward(self, 
        sample: dict[str, int], utility_edges: list[tuple[str, str]]
    ) -> dict[str, float]:
        rewards: dict[str, float] = Counter()
        rewards["grades"] += sample["Time studying"] + sample["Tutoring"]
        rewards["social"] += sample["ECs"]
        rewards["health"] += sample["Sleep"] + sample["Exercise"]
        return rewards


    def default_reward(self, 
        sample: dict[str, int], utility_edges: list[tuple[str, str]]
    ) -> dict[str, float]:
        rewards: dict[str, float] = Counter()
        for var, utility in utility_edges:
            rewards[utility] += sample[var]
        return rewards

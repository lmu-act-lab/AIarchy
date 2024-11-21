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


    def plot_monte_carlo(self, agents: list[CausalLearningAgent]) -> None:
        """
        Plots the average reward across agents.

        Parameters
        ----------
        agents : list[CausalLearningAgent]
            List of agents to plot.
        """
        average_rewards: list[float] = []
        for iteration in range(len(agents[0].memory)):
            total_reward = 0
            for agent in agents:
                total_reward += sum(agent.memory[iteration].average_reward.values())
            average_rewards.append(total_reward / len(agents))
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

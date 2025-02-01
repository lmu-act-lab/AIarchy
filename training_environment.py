import networkx as nx
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("TkAgg")  # Use 'TkAgg' backend for interactive plots
from cla import CausalLearningAgent
from util import Counter
import time


class TrainingEnvironment:
    def train(
        self, mc_rep: int, style: str, agents: list[CausalLearningAgent]
    ) -> list[CausalLearningAgent]:
        """
        Trains n agents for a specified number of monte carlo repetitions using a specified style.
        """
        print("=== Training Begin ===")
        start_time = time.time()
        for agent in agents:
            print(f"training agent number {agents.index(agent) + 1}")
            match style:
                case "SA":
                    agent.train_SA(mc_rep)
                case "ME":
                    agent.train_ME(mc_rep)
        end_time = time.time()
        print(f"=== Training End (Elapsed: {end_time - start_time:.2f}s) ===")
        return agents

    def pre_training_visualization(self, agent) -> None:
        """
        Visualize the Bayesian Network structure with nodes and directed edges.
        """
        if hasattr(agent, "parameters"):
            params = agent.parameters

        fig, axs = plt.subplots(1, 2, figsize=(16, 8))

        # Left subplot: Bayesian Network Structure
        G = nx.DiGraph(agent.structural_model.edges())

        pos = nx.spring_layout(G, k=2)

        # Draw nodes in groups based on type and shape
        node_groups = {
            "chance": {"nodes": agent.chance_vars, "color": "blue", "shape": "o"},
            "reflective": {
                "nodes": agent.reflective_vars,
                "color": "green",
                "shape": "s",
            },
            "glue": {
                "nodes": agent.glue_vars,
                "color": "yellow",
                "shape": "h",
            },  # 'h' is hexagon
            "utility": {"nodes": agent.utility_vars, "color": "purple", "shape": "D"},
        }

        for group, attrs in node_groups.items():
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node for node in G.nodes() if node in attrs["nodes"]],
                node_color=attrs["color"],
                node_shape=attrs["shape"],
                node_size=800,
                ax=axs[0],
            )

        # Draw edges with arrows to indicate direction
        nx.draw_networkx_edges(
            G,
            pos,
            ax=axs[0],
            arrows=True,
            arrowsize=20,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",  # Slight curvature for better visibility
        )

        # Draw labels on a separate layer for clarity
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=8,
            font_weight="bold",
            ax=axs[0],
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.7,
            },
        )
        axs[0].set_title("Bayesian Network Structure")
        axs[0].axis("off")  # Hide axes

        # Right subplot: Parameters display
        axs[1].axis("off")
        if hasattr(agent, "parameters"):
            param_lines = [f"{key}: {val}" for key, val in params.items()]
            param_text = "\n".join(param_lines)
            axs[1].text(
                0.5,
                0.5,
                param_text,
                ha="center",
                va="center",
                fontsize=12,
                transform=axs[1].transAxes,
            )
            axs[1].set_title("Parameters")

        plt.tight_layout()
        plt.show()

    def post_training_visualization(self, agents: list["CausalLearningAgent"]) -> None:
        """
        Post-training visualization. Plots parameter evolution if agents store a parameter_history dict.
        Also plots expected rewards across agents.
        """
        # Plot parameter evolution for first agent if available
        if hasattr(agents[0], "parameter_history"):
            history = agents[
                0
            ].parameter_history  # expected to be a dict[str, list[float]]
            plt.figure(figsize=(8, 6))
            for param, values in history.items():
                plt.plot(values, label=param)
            plt.xlabel("Training Iteration")
            plt.ylabel("Parameter Value")
            plt.title("Parameter Evolution Over Training")
            plt.legend()
            plt.show()
        else:
            print("No parameter history available for plotting.")

        # Compare expected rewards across agents
        self.plot_expected_reward(agents, mc_reps=10)

    def plot_cpt_comparison(
        self, agent: "CausalLearningAgent", old_cpds: list, new_cpds: list
    ) -> None:
        """
        Plots side-by-side sampling model graphs annotated with old and new CPTs.
        """

        def draw_graph(cpds, title):
            # Clone the original sampling model
            G = nx.DiGraph(agent.structural_model.edges())
            pos = nx.spring_layout(G)
            labels = {}
            for node in G.nodes():
                # For demonstration, we attach number of rows in the CPT if found
                found = [cpd for cpd in cpds if cpd.variable == node]
                if found:
                    labels[node] = f"{node}\n({len(found[0].values)} rows)"
                else:
                    labels[node] = node
            nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=800)
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
            plt.title(title)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        draw_graph(old_cpds, "Old CPTs")
        plt.subplot(1, 2, 2)
        draw_graph(new_cpds, "New CPTs")
        plt.show()

    def plot_expected_reward(
        self, agents: list["CausalLearningAgent"], mc_reps: int
    ) -> None:
        """
        Plots the average expected reward over a number of monte carlo repetitions across agents.
        """
        average_rewards = []
        # For each agent, simulate evaluation mc_reps times and average the losses
        for agent in agents:
            total_reward = 0
            for _ in range(mc_reps):
                # Assumes agent has a test_dataframe attribute or similar method for evaluation.
                # Here we use a dummy input DataFrame by sampling from agent.memory if available.
                if agent.memory:
                    # Use the last recorded time step's memory as sample
                    last_memory = agent.memory[-1].memory
                    total_reward += agent.evaluate(last_memory)
                else:
                    total_reward += 0
            average_rewards.append(total_reward / mc_reps)

        plt.figure(figsize=(8, 6))
        plt.bar(range(len(agents)), average_rewards)
        plt.xlabel("Agent Index")
        plt.ylabel("Average Expected Reward")
        plt.title(f"Expected Reward over {mc_reps} Monte Carlo Repetitions")
        plt.show()

    def plot_monte_carlo(
        self, agents: list[CausalLearningAgent], show_params: bool = False
    ) -> None:
        """
        Plots the average reward across agents.

        Parameters
        ----------
        agents : list[CausalLearningAgent]
            List of agents to plot.
        """
        average_rewards: list[float] = []
        params: dict[str, UnionType[float, int]] = agents[0].parameters
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
                    param_val_str = ", ".join(
                        [f"{k}: {v}" for k, v in param_val.items()]
                    )
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

    def reward(
        self, sample: dict[str, int], utility_edges: list[tuple[str, str]]
    ) -> dict[str, float]:
        rewards: dict[str, float] = Counter()
        rewards["grades"] += sample["Time studying"] + sample["Tutoring"]
        rewards["social"] += sample["ECs"]
        rewards["health"] += sample["Sleep"] + sample["Exercise"]
        return rewards

    def default_reward(
        self, sample: dict[str, int], utility_edges: list[tuple[str, str]]
    ) -> dict[str, float]:
        rewards: dict[str, float] = Counter()
        for var, utility in utility_edges:
            rewards[utility] += sample[var]
        return rewards

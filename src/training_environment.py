import networkx as nx
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("TkAgg")  # Use 'TkAgg' backend for interactive plots
from src.cla import CausalLearningAgent
from src.util import Counter
import time
import pandas as pd
import random


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

    def pre_training_visualization(self, agent, name="", save = False) -> None:
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

        if save:
            fig.savefig(f"{name}/pre_training_visualization.png")
            plt.close(fig)
        else:
            plt.show()

    def post_training_visualization(self, agents: list["CausalLearningAgent"], name = "", save = False) -> None:
        """
        Post-training visualization. Plots parameter evolution (from ema_history) and 
        average utility weights (from each TimeStep's weights) on the same graph using two y-axes.
        
        The left y-axis corresponds to the parameter evolution, while the right y-axis
        corresponds to the average utility weights. This allows you to compare how both
        evolve over training iterations.
        """
        fig, ax1 = plt.subplots()

        # Plot parameter evolution if ema_history exists.
        if hasattr(agents[0], "ema_history") and agents[0].ema_history:
            history = agents[0].ema_history  # List[dict[str, float]]
            x_params = list(range(len(history)))
            for key in history[0].keys():
                y_params = [d[key] for d in history]
                ax1.plot(x_params, y_params, label=f"EMA: {key}")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("EMA Value")
        else:
            print("No parameter history available for plotting.")
        
        # Plot average weights for each utility using a second y-axis.
        if hasattr(agents[0], "memory") and agents[0].memory:
            utilities = list(agents[0].utility_vars)
            num_iterations = len(agents[0].memory)
            weights_over_time = {util: [] for util in utilities}
            # Compute the average weight per utility for each timestep.
            for t in range(num_iterations):
                for util in utilities:
                    total_weight = 0.0
                    for agent in agents:
                        timestep = agent.memory[t]
                        total_weight += timestep.weights[util]
                    avg_weight = total_weight / len(agents)
                    weights_over_time[util].append(avg_weight)
            x_weights = list(range(num_iterations))
            
            # Create a second y-axis sharing the same x-axis.
            ax2 = ax1.twinx()
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for i, util in enumerate(utilities):
                color = colors[i % len(colors)]
                ax2.plot(x_weights, weights_over_time[util], '-o', color=color, label=f"{util} weight")
            ax2.set_ylabel("Average Utility Weight")
        else:
            print("No memory available for plotting weights.")

        # Combine legends from both axes.
        lines1, labels1 = ax1.get_legend_handles_labels()
        if hasattr(agents[0], "memory") and agents[0].memory:
            lines2, labels2 = ax2.get_legend_handles_labels()
        else:
            lines2, labels2 = [], []
        
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        ax1.set_title("Parameter Evolution Over Iterations")

        if save:
            fig.savefig(f"{name}/post_training_visualization.png")
            plt.close(fig)
        else:
            plt.show()

    def show_cpt_changes(self, agents: list["CausalLearningAgent"], name = "", save = False) -> None:
        """
        Visualize CPD changes by plotting a heatmap for each CPD of the first agent.
        Only CPDs for variables in agent.reflective_vars are plotted.
        The background color intensity is determined by the delta (change from the original CPD),
        while the original CPD values are annotated on the heatmap.
        The evidence configurations are shown in detail (e.g. "diff=0, intel=1").

        The color scale is fixed (vmin=-1 and vmax=1) so that changes are comparable across CPDs.
        """
        # Use only the first agent in the list.
        agent = agents[0]
        cpds = agent.get_cpts()
        
        import seaborn as sns
        from itertools import product
        import pandas as pd
        import numpy as np

        def cpd_to_df(cpd):
            """
            Converts a TabularCPD (or equivalent DiscreteFactor) to a DataFrame with descriptive row labels.
            Assumes the ordering of variables in the CPD is:
                [target, evidence1, evidence2, ...]
            Rows will correspond to each evidence configuration and columns to target states.
            """
            target = cpd.variables[0]
            target_card = cpd.cardinality[0]
            # If there are evidence variables, they appear as the remaining elements in cpd.variables.
            if len(cpd.variables) > 1:
                evidence = cpd.variables[1:]
                # Obtain evidence cardinalities by slicing the overall cardinality array.
                evidence_card = list(cpd.cardinality[1:])
                product_evidence = np.prod(evidence_card)
                # Create descriptive labels for each evidence configuration.
                levels = [list(range(card)) for card in evidence_card]
                index_labels = [
                    ", ".join(f"{var}={val}" for var, val in zip(evidence, tup))
                    for tup in product(*levels)
                ]
                # Reshape values to (target_card, product_evidence) then transpose so that rows are evidence configs.
                reshaped = cpd.values.reshape((target_card, product_evidence)).T
                df = pd.DataFrame(reshaped,
                                index=index_labels,
                                columns=[f"{target}={i}" for i in range(target_card)])
            else:
                # No evidence; create a single-row DataFrame.
                df = pd.DataFrame(cpd.values.reshape((target_card, 1)).T,
                                columns=[f"{target}={i}" for i in range(target_card)],
                                index=["No evidence"])
            return df

        # Loop over each CPD in the agent and only process those whose target is in reflective_vars.
        for cpd in cpds:
            if cpd.variable not in agent.reflective_vars:
                continue

            # Compute the delta CPD (difference from the original CPD)
            delta_cpd = agent.compute_cpd_delta(cpd.variable)
            
            # Convert both the original CPD and the delta CPD to DataFrames.
            df_original = cpd_to_df(cpd)
            df_delta = cpd_to_df(delta_cpd)
            
            # Create a heatmap with an objective color scale.
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(
                df_delta,
                annot=df_original,  # Annotate each cell with the original CPD value.
                fmt=".2f",
                cmap="RdYlGn",
                center=0,      # Zero delta is neutral.
                vmin=-1,       # Fixed minimum for the color scale.
                vmax=1         # Fixed maximum for the color scale.
            )
            plt.title(f"CPD Changes for {cpd.variable}")
            plt.xlabel("Target State")
            plt.ylabel("Evidence Configuration")

            filename = f"{name}/cpt_changes_{cpd.variable}.png"
            if save:
                fig.savefig(filename)
                plt.close(fig)
            else:
                plt.show()

    def plot_cpt_comparison(
        self, agent: "CausalLearningAgent", old_cpds: list, new_cpds: list, name = "", save = False
    ) -> None:
        """
        Plots side-by-side sampling model graphs annotated with old and new CPTs.
        """
        fig = plt.figure(figsize=(12, 5))
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

        if save:
            fig.savefig(f"{name}/cpt_comparison.png")
            plt.close(fig)
        else:
            plt.show()

    def plot_expected_reward(
        self, agents: list["CausalLearningAgent"], mc_reps: int, name="", save=False
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

        fig = plt.figure(figsize=(8, 6))
        plt.bar(range(len(agents)), average_rewards)
        plt.xlabel("Agent Index")
        plt.ylabel("Average Expected Reward")
        plt.title(f"Expected Reward over {mc_reps} Monte Carlo Repetitions")
        if save:
            fig.savefig(f"{name}/expected_reward.png")
            plt.close(fig)
        else:
            plt.show()
    def plot_monte_carlo(
        self, agents: list[CausalLearningAgent], show_params: bool = False, name = "", save = False
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
        fig = plt.figure()
        plt.plot(average_rewards)
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        plt.title(f"Average Reward Across {len(agents)} Agents")

        if save:
            fig.savefig(f"{name}/monte_carlo.png")
            plt.close(fig)
        else:
            plt.show()

    def plot_weighted_rewards(self, agents: list[CausalLearningAgent], show_params: bool = False, name = "", save = False) -> None:
        """
        Plots the weighted rewards for each utility for the first agent only.

        For each utility defined in the agent's `utility_vars`, the method computes:
            - The subjective reward as:
                subjective_reward = average_reward (from timestep) * subjective weight (from timestep)
            - The objective reward as:
                objective_reward = average_reward (from timestep) * objective weight (agent's fixed weight)
        Two lines are plotted for each utility:
            - A solid line (with markers) for subjective rewards.
            - A dashed line for objective rewards.
        Both lines for the same utility share the same color.

        Parameters
        ----------
        agents : list[CausalLearningAgent]
            List of agents (only the first agent is used for plotting).
        show_params : bool, optional
            If True, displays the parameters (from the first agent) below the plot.
        """
        # Use only the first agent.
        agent = agents[0]
        utilities = list(agent.utility_vars)
        num_iterations = len(agent.memory)
        
        # Dictionaries to hold the weighted rewards per iteration.
        subjective_weighted = {util: [] for util in utilities}
        objective_weighted = {util: [] for util in utilities}
        
        # Loop through each timestep in the agent's memory.
        for t in range(num_iterations):
            timestep = agent.memory[t]
            for util in utilities:
                # Compute the weighted subjective reward using the timestep's weight.
                subjective_reward = timestep.average_reward[util] * timestep.weights[util]
                # Compute the weighted objective reward using the agent's fixed objective weight.
                objective_reward = timestep.average_reward[util] * agent.objective_weights[util]
                subjective_weighted[util].append(subjective_reward)
                objective_weighted[util].append(objective_reward)
        
        # Optionally display parameters from the first agent.
        if show_params:
            params = agent.parameters
            num_params = len(params)
            bottom_margin = 0.1 + 0.04 * num_params
            plt.subplots_adjust(bottom=bottom_margin)
            for i, (param, param_val) in enumerate(params.items()):
                if isinstance(param_val, dict):
                    param_val_str = ", ".join([f"{k}: {v}" for k, v in param_val.items()])
                else:
                    param_val_str = str(param_val)
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
        
        iterations = list(range(num_iterations))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        # Plot the lines for each utility.
        fig = plt.figure()
        for i, util in enumerate(utilities):
            color = colors[i % len(colors)]
            plt.plot(iterations, subjective_weighted[util], '-o', color=color, label=f"{util} subjective")
            plt.plot(iterations, objective_weighted[util], '--', color=color, label=f"{util} objective")
        
        plt.xlabel("Iteration")
        plt.ylabel("Weighted Reward")
        plt.title("Subjective & Objective Weighted Rewards for First Agent")
        plt.legend()

        if save:
            fig.savefig(f"{name}/weighted_rewards.png")
            plt.close(fig)
        else:
            plt.show()

    def plot_u_hat_model_losses(self, agents: list["CausalLearningAgent"], name = "", save = False) -> None:
        """
        Plots the loss history for each model in the first agent's u_hat_models.

        Parameters
        ----------
        agents : list[CausalLearningAgent]
            A list of causal learning agents. Only the first agent's u_hat_models are plotted.
        """
        if not agents:
            print("No agents provided.")
            return

        # Use only the first agent.
        agent = agents[0]
        
        # Check if the agent has the u_hat_models attribute and it is not empty.
        if not hasattr(agent, "u_hat_models") or not agent.u_hat_models:
            print("The agent does not have any u_hat_models to plot.")
            return

        fig = plt.figure(figsize=(10, 6))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i, (model_name, model) in enumerate(agent.u_hat_models.items()):
            if hasattr(model, "losses") and model.losses:
                iterations = list(range(len(model.losses)))
                plt.plot(iterations, model.losses, '-o', color=colors[i % len(colors)], label=f"{model_name}")
            else:
                print(f"Model {model_name} does not have a loss history to plot.")

        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss History for each u_hat_model in the First Agent")
        plt.legend(loc='best')
        if save:
            fig.savefig(f"{name}/u_hat_model_losses.png")
            plt.close(fig)
        else:
            plt.show()

    def original_student_reward(
        self, sample: dict[str, int], utility_edges: list[tuple[str, str]], noise: float = 0.0
    ) -> dict[str, float]:
        rewards: dict[str, float] = Counter()
        rewards["grades"] += sample["Time studying"] + sample["Tutoring"]
        rewards["social"] += sample["ECs"]
        rewards["health"] += sample["Sleep"] + sample["Exercise"]
        return rewards

    def test_downweigh_reward(
        self, sample: dict[str, int], utility_edges: list[tuple[str, str]], noise: float = 0.0, lower_tier_pooled_reward: dict[str, float] = None
    ) -> dict[str, float]:

        rewards: dict[str, float] = Counter()
        rewards["util_1"] += 0 if sample["refl_1"] == 1 else 1
        rewards["util_2"] += sample["refl_1"]
        return rewards

    def default_reward(
        self, sample: dict[str, int], utility_edges: list[tuple[str, str]], noise: float = 0.0, lower_tier_pooled_reward: dict[str, float] = None
    ) -> dict[str, float]:
        rewards: dict[str, float] = Counter()
        for var, utility in utility_edges:
            noise_term = random.gauss(0, noise) if noise > 0 else 0.0
            if utility == "teacher_regulation":
                rewards[utility] -= sample["grade_leniency"]
            else:
                rewards[utility] += sample[var] + noise_term



        if lower_tier_pooled_reward:
            for util, reward in lower_tier_pooled_reward.items():
                rewards[util] += reward
        

        return rewards


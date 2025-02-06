from cla import CausalLearningAgent
from pgmpy.factors.discrete import TabularCPD  # type: ignore
from training_environment import TrainingEnvironment
from pgmpy import global_vars
import copy
from approxq_agent import ApproxQComparisonAgent

testing_environment: TrainingEnvironment = TrainingEnvironment()

list_of_cpts = [
    TabularCPD(variable="SES", variable_card=3, values=[[0.29], [0.52], [0.19]]),
    TabularCPD(variable="Motivation", variable_card=2, values=[[0.5], [0.5]]),
    TabularCPD(
        variable="Tutoring",
        variable_card=2,
        values=[[0.9, 0.6, 0.2], [0.1, 0.4, 0.8]],
        evidence=["SES"],
        evidence_card=[3],
    ),
    TabularCPD(
        variable="Time studying",
        variable_card=3,
        values=(
            [
                [0.6, 0.7, 0.3, 0.4, 0.55, 0.65, 0.1, 0.2, 0.4, 0.5, 0.1, 0.2],
                [
                    0.25,
                    0.2,
                    0.5,
                    0.45,
                    0.3,
                    0.25,
                    0.4,
                    0.35,
                    0.4,
                    0.35,
                    0.3,
                    0.25,
                ],
                [
                    0.15,
                    0.1,
                    0.2,
                    0.15,
                    0.15,
                    0.1,
                    0.5,
                    0.45,
                    0.2,
                    0.15,
                    0.6,
                    0.55,
                ],
            ]
        ),
        evidence=["SES", "Motivation", "ECs"],
        evidence_card=[3, 2, 2],
    ),
    TabularCPD(
        variable="Exercise",
        variable_card=2,
        values=[[0.6, 0.4, 0.2], [0.4, 0.6, 0.8]],
        evidence=["SES"],
        evidence_card=[3],
    ),
    TabularCPD(
        variable="ECs",
        variable_card=2,
        values=[[0.7, 0.5, 0.1], [0.3, 0.5, 0.9]],
        evidence=["SES"],
        evidence_card=[3],
    ),
    TabularCPD(
        variable="Sleep",
        variable_card=3,
        values=[[0.1, 0.25, 0.55], [0.3, 0.4, 0.25], [0.6, 0.35, 0.2]],
        evidence=["Time studying"],
        evidence_card=[3],
    ),
]

x = CausalLearningAgent(
    sampling_edges=[
        ("SES", "Tutoring"),
        ("SES", "ECs"),
        ("SES", "Time studying"),
        ("Motivation", "Time studying"),
        ("SES", "Exercise"),
        ("ECs", "Time studying"),
        ("Time studying", "Sleep"),
    ],
    utility_edges=[
        ("Time studying", "grades"),
        ("Tutoring", "grades"),
        ("ECs", "social"),
        ("Sleep", "health"),
        ("Exercise", "health"),
    ],
    cpts=list_of_cpts,
    utility_vars={"grades", "social", "health"},
    reflective_vars={"Time studying", "Exercise", "Sleep", "ECs"},
    chance_vars={"Tutoring", "Motivation", "SES"},
    glue_vars=set(),
    reward_func=testing_environment.default_reward,
    fixed_evidence={"SES": 1},
    hidden_vars=["Tutoring"]
)

y = CausalLearningAgent(
    sampling_edges=[
        ("SES", "Tutoring"),
        ("SES", "ECs"),
        ("SES", "Time studying"),
        ("Motivation", "Time studying"),
        ("SES", "Exercise"),
        ("ECs", "Time studying"),
        ("Time studying", "Sleep"),
    ],
    utility_edges=[
        ("Time studying", "grades"),
        ("Tutoring", "grades"),
        ("ECs", "social"),
        ("Sleep", "health"),
        ("Exercise", "health"),
    ],
    cpts=list_of_cpts,
    utility_vars={"grades", "social", "health"},
    reflective_vars={"Time studying", "Exercise", "Sleep", "ECs"},
    chance_vars={"Tutoring", "Motivation", "SES"},
    glue_vars=set(),
    reward_func=testing_environment.default_reward,
    fixed_evidence={"SES": 0},
)

struct_1: CausalLearningAgent = CausalLearningAgent(
    sampling_edges=[("refl_2", "refl_1")],
    utility_edges=[("refl_2", "util_2"), ("refl_1", "util_1")],
    cpts=[
        TabularCPD(variable="refl_2", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="refl_1",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["refl_2"],
            evidence_card=[2],
        ),
    ],
    utility_vars={"util_1", "util_2"},
    reflective_vars={"refl_1", "refl_2"},
    chance_vars=set(),
    glue_vars=set(),
    reward_func=testing_environment.default_reward,
    fixed_evidence={},
)

struct_2: CausalLearningAgent = CausalLearningAgent(
    sampling_edges=[("chance_1", "refl_1"), ("chance_2", "refl_1")],
    utility_edges=[("refl_1", "util_1")],
    cpts=[
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="chance_2", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="refl_1",
            variable_card=2,
            values=[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
            evidence=["chance_1", "chance_2"],
            evidence_card=[2, 2],
        ),
    ],
    utility_vars={"util_1"},
    reflective_vars={"refl_1"},
    chance_vars={"chance_1", "chance_2"},
    glue_vars=set(),
    reward_func=testing_environment.default_reward,
    fixed_evidence={},
)

struct_3: CausalLearningAgent = CausalLearningAgent(
    sampling_edges=[("refl_1", "chance_1"), ("refl_2", "chance_1")],
    utility_edges=[("chance_1", "util_1")],
    cpts=[
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="refl_2", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="chance_1",
            variable_card=2,
            values=[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
            evidence=["refl_1", "refl_2"],
            evidence_card=[2, 2],
        ),
    ],
    utility_vars={"util_1"},
    reflective_vars={"refl_1", "refl_2"},
    chance_vars={"chance_1"},
    glue_vars=set(),
    reward_func=testing_environment.default_reward,
    fixed_evidence={},
)

struct_4: CausalLearningAgent = CausalLearningAgent(
    [("chance_1", "refl_1")],
    [("refl_1", "util_1")],
    [
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="refl_1",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["chance_1"],
            evidence_card=[2],
        ),
    ],
    {"util_1"},
    {"refl_1"},
    {"chance_1"},
    set(),
    testing_environment.default_reward,
    {},
)

struct_5: CausalLearningAgent = CausalLearningAgent(
    [("refl_1", "chance_1")],
    [("chance_1", "util_1")],
    [
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="chance_1",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["refl_1"],
            evidence_card=[2],
        ),
    ],
    {"util_1"},
    {"refl_1"},
    {"chance_1"},
    set(),
    testing_environment.default_reward,
    {},
)

struct_6 = CausalLearningAgent(
    [("chance_1", "refl_1"), ("chance_1", "refl_2")],
    [("refl_1", "util_1"), ("refl_2", "util_1")],
    [
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="refl_1",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["chance_1"],
            evidence_card=[2],
        ),
        TabularCPD(
            variable="refl_2",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["chance_1"],
            evidence_card=[2],
        ),
    ],
    {"util_1"},
    {"refl_1", "refl_2"},
    {"chance_1"},
    set(),
    testing_environment.default_reward,
    {},
)

struct_7 = CausalLearningAgent(
    [("refl_1", "chance_1"), ("refl_1", "chance_2")],
    [("chance_1", "util_1"), ("chance_2", "util_1")],
    [
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="chance_1",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["refl_1"],
            evidence_card=[2],
        ),
        TabularCPD(
            variable="chance_2",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["refl_1"],
            evidence_card=[2],
        ),
    ],
    {"util_1"},
    {"refl_1"},
    {"chance_1", "chance_2"},
    set(),
    testing_environment.default_reward,
    {},
)

struct_8: CausalLearningAgent = CausalLearningAgent(
    [],
    [("refl_1", "util_1"), ("chance_1", "util_1"), ("chance_2", "util_1")],
    [
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="chance_2", variable_card=2, values=[[0.5], [0.5]]),
    ],
    {"util_1"},
    {"refl_1"},
    {"chance_1", "chance_2"},
    set(),
    testing_environment.default_reward,
    {},
)

struct_9: CausalLearningAgent = CausalLearningAgent(
    [],
    [("refl_1", "util_1"), ("chance_1", "util_1"), ("refl_2", "util_1")],
    [
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="refl_2", variable_card=2, values=[[0.5], [0.5]]),
    ],
    {"util_1"},
    {"refl_1", "refl_2"},
    {"chance_1"},
    set(),
    testing_environment.default_reward,
    {},
)

struct_10: CausalLearningAgent = CausalLearningAgent(
    [("chance_1", "refl_1"), ("refl_1", "chance_2")],
    [("chance_2", "util_1")],
    [
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="refl_1",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["chance_1"],
            evidence_card=[2],
        ),
        TabularCPD(
            variable="chance_2",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["refl_1"],
            evidence_card=[2],
        ),
    ],
    {"util_1"},
    {"refl_1"},
    {"chance_1", "chance_2"},
    set(),
    testing_environment.default_reward,
    {},
)

struct_11: CausalLearningAgent = CausalLearningAgent(
    [],
    [("refl_1", "util_1"), ("chance_1", "util_1")],
    [
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
    ],
    {"util_1"},
    {"refl_1"},
    {"chance_1"},
    set(),
    testing_environment.default_reward,
    {},
)

struct_13: CausalLearningAgent = CausalLearningAgent(
    [],
    [("refl_1", "util_1"), ("refl_2", "util_1")],
    [
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="refl_2", variable_card=2, values=[[0.5], [0.5]]),
    ],
    {"util_1"},
    {"refl_1", "refl_2"},
    set(),
    set(),
    testing_environment.default_reward,
    {},
)

#########################################
##                                     ##
##             TESTING AREA            ##
##                                     ##
#########################################


import logging
logging.disable(logging.WARNING)

testing_environment.pre_training_visualization(x)
testing_environment.pre_training_visualization(y)


# Begin training (here using SA style and 5 monte carlo repetitions for demonstration)
agents = [x, y, struct_1, struct_2, struct_3]
trained_agents = testing_environment.train(5, "SA", agents)

# Post-Training Visualizations:
# For demonstration, we assume that the agents have stored old_cpds and new_cpds (this code is illustrative)
# Here we simply use the same CPTs from x for both old and new
old_cpds = x.get_cpts() if hasattr(x, "get_cpts") else []
new_cpds = x.get_cpts() if hasattr(x, "get_cpts") else []
testing_environment.plot_cpt_comparison(x, old_cpds, new_cpds)

testing_environment.post_training_visualization(trained_agents)


# x_agents = [copy.deepcopy(x) for _ in range(10)]
# print("training x")
# testing_environment.train(5, "SA", x_agents)
# print("x training done")

# y_agents = [copy.deepcopy(y) for _ in range(10)]
# print("training y")
# testing_environment.train(800, "SA", y_agents)
# print("y training done")

# struct_1_agents = [copy.deepcopy(struct_1) for _ in range(15)]
# print("training struct_1")
# testing_environment.train(300, "SA", struct_1_agents)
# print("struct_1 training done")

# struct_2_agents = [copy.deepcopy(struct_2) for _ in range(15)]
# print("training struct_2")
# testing_environment.train(300, "SA", struct_2_agents)
# print("struct_2 training done")

# struct_3_agents = [copy.deepcopy(struct_3) for _ in range(15)]
# print("training struct_3")
# testing_environment.train(300, "SA", struct_3_agents)
# print("struct_3 training done")

# struct_4_agents = [copy.deepcopy(struct_4) for _ in range(15)]
# print("training struct_4")
# testing_environment.train(300, "SA", struct_4_agents)
# print("struct_4 training done")

# struct_5_agents = [copy.deepcopy(struct_5) for _ in range(15)]
# print("training struct_5")
# testing_environment.train(300, "SA", struct_5_agents)
# print("struct_5 training done")

# struct_6_agents = [copy.deepcopy(struct_6) for _ in range(15)]
# print("training struct_6")
# testing_environment.train(300, "SA", struct_6_agents)
# print("struct_6 training done")

# struct_7_agents = [copy.deepcopy(struct_7) for _ in range(15)]
# print("training struct_7")
# testing_environment.train(300, "SA", struct_7_agents)
# print("struct_7 training done")

# struct_8_agents = [copy.deepcopy(struct_8) for _ in range(15)]
# print("training struct_8")
# testing_environment.train(300, "SA", struct_8_agents)
# print("struct_8 training done")

# struct_9_agents = [copy.deepcopy(struct_9) for _ in range(15)]
# print("training struct_9")
# testing_environment.train(300, "SA", struct_9_agents)
# print("struct_9 training done")

# struct_10_agents = [copy.deepcopy(struct_10) for _ in range(15)]
# print("training struct_10")
# testing_environment.train(300, "SA", struct_10_agents)
# print("struct_10 training done")

# struct_11_agents = [copy.deepcopy(struct_11) for _ in range(15)]
# print("training struct_11")
# testing_environment.train(300, "SA", struct_11_agents)
# print("struct_11 training done")

# struct_13_agents = [copy.deepcopy(struct_13) for _ in range(15)]
# print("training struct_13")
# testing_environment.train(300, "SA", struct_13_agents)
# print("struct_13 training done")


# print("x done")
# testing_environment.plot_monte_carlo(x_agents, show_params=True)
# print("y done")
# testing_environment.plot_monte_carlo(y_agents, show_params=True)
# print("struct_1 done")
# testing_environment.plot_monte_carlo(struct_1_agents, show_params=True)
# print("struct_2 done")
# testing_environment.plot_monte_carlo(struct_2_agents, show_params=True)
# print("struct_3 done")
# testing_environment.plot_monte_carlo(struct_3_agents, show_params=True)
# print("struct_4 done")
# testing_environment.plot_monte_carlo(struct_4_agents, show_params=True)
# print("struct_5 done")
# testing_environment.plot_monte_carlo(struct_5_agents, show_params=True)
# print("struct_6 done")
# testing_environment.plot_monte_carlo(struct_6_agents, show_params=True)
# print("struct_7 done")
# testing_environment.plot_monte_carlo(struct_7_agents, show_params=True)
# print("struct_8 done")
# testing_environment.plot_monte_carlo(struct_8_agents, show_params=True)
# print("struct_9 done")
# testing_environment.plot_monte_carlo(struct_9_agents, show_params=True)
# print("struct_10 done")
# testing_environment.plot_monte_carlo(struct_10_agents, show_params=True)
# print("struct_11 done")
# testing_environment.plot_monte_carlo(struct_11_agents, show_params=True)
# print("struct_13 done")
# testing_environment.plot_monte_carlo(struct_13_agents, show_params=True)

# x_agents[0].plot_memory_against(y_agents[0])

new_agent = ApproxQComparisonAgent(actions = [("Time studying", 0), ("Time studying", 1), ("Time studying", 2), ("Exercise", 0), ("Exercise", 1), ("Sleep", 0), ("Sleep", 1), ("Sleep", 2), ("ECs", 0), ("ECs", 1)], gamma = 0.1, alpha = 0.01, cla_agent=x)

iterations = 100
rewards = []
while iterations > 0:
    
    new_agent.choose_action()
    iterations -= 1
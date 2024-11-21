from cla import CausalLearningAgent
from pgmpy.factors.discrete import TabularCPD  # type: ignore
from training_environment import TrainingEnvironment
import copy

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

# Monte carlo stuff
x_agents = [copy.deepcopy(x) for _ in range(10)]
testing_environment.train(100, "SA", x_agents)


y_agents = [copy.deepcopy(y) for _ in range(10)]
testing_environment.train(100, "SA", y_agents)


struct_1_agents = [copy.deepcopy(struct_1) for _ in range(10)]
testing_environment.train(200, "SA", struct_1_agents)


struct_2_agents = [copy.deepcopy(struct_2) for _ in range(10)]
testing_environment.train(50, "SA", struct_2_agents)


struct_3_agents = [copy.deepcopy(struct_3) for _ in range(10)]
testing_environment.train(50, "SA", struct_3_agents)


struct_4_agents = [copy.deepcopy(struct_4) for _ in range(10)]
testing_environment.train(50, "SA", struct_4_agents)


struct_5_agents = [copy.deepcopy(struct_5) for _ in range(10)]
testing_environment.train(50, "SA", struct_5_agents)


struct_6_agents = [copy.deepcopy(struct_6) for _ in range(10)]
testing_environment.train(50, "SA", struct_6_agents)


struct_7_agents = [copy.deepcopy(struct_7) for _ in range(10)]
testing_environment.train(50, "SA", struct_7_agents)


struct_8_agents = [copy.deepcopy(struct_8) for _ in range(10)]
testing_environment.train(50, "SA", struct_8_agents)


struct_9_agents = [copy.deepcopy(struct_9) for _ in range(10)]
testing_environment.train(50, "SA", struct_9_agents)


struct_10_agents = [copy.deepcopy(struct_10) for _ in range(10)]
testing_environment.train(50, "SA", struct_10_agents)


struct_11_agents = [copy.deepcopy(struct_11) for _ in range(10)]
testing_environment.train(50, "SA", struct_11_agents)


struct_13_agents = [copy.deepcopy(struct_13) for _ in range(10)]
testing_environment.train(50, "SA", struct_13_agents)


print("x done")
testing_environment.plot_monte_carlo(x_agents)
print("y done")
testing_environment.plot_monte_carlo(y_agents)
print("struct_1 done")
testing_environment.plot_monte_carlo(struct_1_agents)
print("struct_2 done")
testing_environment.plot_monte_carlo(struct_2_agents)
print("struct_3 done")
testing_environment.plot_monte_carlo(struct_3_agents)
print("struct_4 done")
testing_environment.plot_monte_carlo(struct_4_agents)
print("struct_5 done")
testing_environment.plot_monte_carlo(struct_5_agents)
print("struct_6 done")
testing_environment.plot_monte_carlo(struct_6_agents)
print("struct_7 done")
testing_environment.plot_monte_carlo(struct_7_agents)
print("struct_8 done")
testing_environment.plot_monte_carlo(struct_8_agents)
print("struct_9 done")
testing_environment.plot_monte_carlo(struct_9_agents)
print("struct_10 done")
testing_environment.plot_monte_carlo(struct_10_agents)
print("struct_11 done")
testing_environment.plot_monte_carlo(struct_11_agents)
print("struct_13 done")
testing_environment.plot_monte_carlo(struct_13_agents)
from cla import CausalLearningAgent
from pgmpy.factors.discrete import TabularCPD  # type: ignore
from training_environment import TrainingEnvironment
from pgmpy import global_vars
import copy
from exactq_agent import ExactQComparisonAgent
import matplotlib.pyplot as plt

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

Low_SES = CausalLearningAgent(
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

Mid_SES = CausalLearningAgent(
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

High_SES = CausalLearningAgent(
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
    fixed_evidence={"SES": 2},
)

academic_student = CausalLearningAgent(
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
    fixed_evidence={"SES": 2},
    weights={"grades": 0.5, "health": 0.25, "social": 0.25},
)

healthy_student = CausalLearningAgent(
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
    fixed_evidence={"SES": 2},
    weights={"grades": 0.25, "health": 0.5, "social": 0.25},
)

social_student = CausalLearningAgent(
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
    fixed_evidence={"SES": 2},
    weights={"grades": 0.25, "health": 0.25, "social": 0.5},
)

#########################################
##                                     ##
##             TESTING AREA            ##
##                                     ##
#########################################


import logging

logging.disable(logging.WARNING)

# --- EXACT Q SECTION --- #
# actions = [("Time studying", 0), ("Time studying", 1), ("Time studying", 2), ("Exercise", 0), ("Exercise", 1), ("Sleep", 0), ("Sleep", 1), ("Sleep", 2), ("ECs", 0), ("ECs", 1)]
# struct_2_actions = [("refl_1", 0), ("refl_1", 1)]

# actions = [(action, 0.01) for action in actions[::-1]]
# struct_2_actions = [(action, 0.05) for action in struct_2_actions[::-1]]

# agents = [copy.deepcopy(ExactQComparisonAgent(actions = struct_2_actions, gamma = 0.1, alpha = 0.01, cla_agent=copy.deepcopy(struct_2))) for i in range(15)]

# for exact in agents:
#     print(f"agent: {agents.index(exact)}")
#     exact.train(500)

# average_rewards: list[float] = []
# for iteration in range(len(agents[0].reward_tracker)):
#     total_reward = 0
#     for agent in agents:
#         total_reward += agent.reward_tracker[iteration]
#     average_rewards.append(total_reward / len(agents))

# for cpt in agents[0].agent.get_cpts():
#     print(cpt)

# print(agents[0].qvals)

# max_key = max(agents[0].qvals, key=agents[0].qvals.get)
# print("Key with max value:", max_key)
# print("Max value:", agents[0].qvals[max_key])


# plt.plot(average_rewards)
# plt.xlabel('Iterations')
# plt.ylabel('Reward')
# plt.title('Reward Tracker Over Iterations')
# plt.show()


# -------- DOWNWEIGH TESTING ---------

# downweigh_struct: CausalLearningAgent = CausalLearningAgent(
#     sampling_edges=[],
#     utility_edges=[("refl_1", "util_1"), ("refl_1", "util_2")],
#     cpts=[
#         TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
#     ],
#     utility_vars={"util_1", "util_2"},
#     reflective_vars={"refl_1"},
#     chance_vars=set(),
#     glue_vars=set(),
#     reward_func=testing_environment.test_downweigh_reward,
#     fixed_evidence={},
# )
# # print(downweigh_struct.parent_combinations)
# testing_environment.pre_training_visualization(downweigh_struct)
# monte_carlo_downweigh = [copy.deepcopy(downweigh_struct) for agent in range(3)]
# testing_environment.train(30, "SA", monte_carlo_downweigh)
# testing_environment.post_training_visualization(monte_carlo_downweigh)
# testing_environment.show_cpt_changes(monte_carlo_downweigh)
# testing_environment.plot_monte_carlo(monte_carlo_downweigh, show_params=True)
# testing_environment.plot_weighted_rewards(monte_carlo_downweigh, show_params=True)

### SPRING BREAK BIG TESTING RUN ###
# 250 Monte Carlo Agents
# 1000 iterations for smaller models
# 5000 iterations for student models

# Paraameters of interest
test_temperatures: list[float] = [0.2 * i for i in range(1, 11, 3)]
print(test_temperatures)
test_coolings: list[float] = [0.99**i for i in range(1, 11, 3)]
print(test_coolings)
test_cpt_increase_factors: list[float] = [0.02 * i for i in range(1, 11, 3)]
print(test_cpt_increase_factors)
test_dw_alphas: list[float] = [0.02 * i for i in range(1, 11, 3)]
print(test_dw_alphas)
test_dw_factors: list[float] = [0.015 * i for i in range(1, 11, 3)]
print(test_dw_factors)

# struct 1
# testing_environment.pre_training_visualization(struct_1)
struct_1_agents = [copy.deepcopy(struct_1) for agent in range(500)]
testing_environment.train(1000, "SA", struct_1_agents)


# struct 2
# testing_environment.pre_training_visualization(struct_2)
struct_2_agents = [copy.deepcopy(struct_2) for agent in range(500)]
testing_environment.train(1000, "SA", struct_2_agents)


# struct 3
# testing_environment.pre_training_visualization(struct_3)
struct_3_agents = [copy.deepcopy(struct_3) for agent in range(500)]
testing_environment.train(1000, "SA", struct_3_agents)


# struct 4
# testing_environment.pre_training_visualization(struct_4)
struct_4_agents = [copy.deepcopy(struct_4) for agent in range(500)]
testing_environment.train(1000, "SA", struct_4_agents)


# struct 5
# testing_environment.pre_training_visualization(struct_5)
struct_5_agents = [copy.deepcopy(struct_5) for agent in range(500)]
testing_environment.train(1000, "SA", struct_5_agents)


# struct 6
# testing_environment.pre_training_visualization(struct_6)
struct_6_agents = [copy.deepcopy(struct_6) for agent in range(500)]
testing_environment.train(1000, "SA", struct_6_agents)


# struct 7
# testing_environment.pre_training_visualization(struct_7)
struct_7_agents = [copy.deepcopy(struct_7) for agent in range(500)]
testing_environment.train(1000, "SA", struct_7_agents)


# struct 8
# testing_environment.pre_training_visualization(struct_8)
struct_8_agents = [copy.deepcopy(struct_8) for agent in range(500)]
testing_environment.train(1000, "SA", struct_8_agents)


# struct 9
# testing_environment.pre_training_visualization(struct_9)
struct_9_agents = [copy.deepcopy(struct_9) for agent in range(500)]
testing_environment.train(1000, "SA", struct_9_agents)


# struct 10
# testing_environment.pre_training_visualization(struct_10)
struct_10_agents = [copy.deepcopy(struct_10) for agent in range(500)]
testing_environment.train(1000, "SA", struct_10_agents)


# struct 11
# testing_environment.pre_training_visualization(struct_11)
struct_11_agents = [copy.deepcopy(struct_11) for agent in range(500)]
testing_environment.train(1000, "SA", struct_11_agents)


# struct 13
# testing_environment.pre_training_visualization(struct_13)
struct_13_agents = [copy.deepcopy(struct_13) for agent in range(500)]
testing_environment.train(1000, "SA", struct_13_agents)

Low_SES_agents = [copy.deepcopy(Low_SES) for agent in range(250)]
testing_environment.train(5000, "SA", Low_SES_agents)

Mid_SES_agents = [copy.deepcopy(Mid_SES) for agent in range(250)]
testing_environment.train(5000, "SA", Mid_SES_agents)

High_SES_agents = [copy.deepcopy(High_SES) for agent in range(250)]
testing_environment.train(5000, "SA", High_SES_agents)

academic_student_agents = [copy.deepcopy(academic_student) for agent in range(250)]
testing_environment.train(5000, "SA", academic_student_agents)

healthy_student_agents = [copy.deepcopy(healthy_student) for agent in range(250)]
testing_environment.train(5000, "SA", healthy_student_agents)

social_student_agents = [copy.deepcopy(social_student) for agent in range(250)]
testing_environment.train(5000, "SA", social_student_agents)

temperature_agents = []
for temp in test_temperatures:
    Mid_SES_temperature = CausalLearningAgent(
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
        temperature=temp,
    )
    Mid_SES_agents = [copy.deepcopy(Mid_SES_temperature) for agent in range(250)]
    testing_environment.train(5000, "SA", Mid_SES_agents)
    temperature_agents.append(Mid_SES_agents)

cooling_agents = []
for cooling in test_coolings:
    Mid_SES_cooling = CausalLearningAgent(
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
        cooling_factor=cooling,
    )

    Mid_SES_agents = [copy.deepcopy(Mid_SES_cooling) for agent in range(250)]
    testing_environment.train(5000, "SA", Mid_SES_agents)
    cooling_agents.append(Mid_SES_agents)

cpt_increase_agents = []
for factor in test_cpt_increase_factors:
    Mid_SES_cpt_increase = CausalLearningAgent(
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
        cpt_increase_factor=factor,
    )
    Mid_SES_agents = [copy.deepcopy(Mid_SES_cpt_increase) for agent in range(250)]
    testing_environment.train(5000, "SA", Mid_SES_agents)
    cpt_increase_agents.append(Mid_SES_agents)

dw_alphas_agents = []
for dw_alpha in test_dw_alphas:
    Mid_SES_dw_alpha = CausalLearningAgent(
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
        downweigh_alpha=dw_alpha,
    )
    Mid_SES_agents = [copy.deepcopy(Mid_SES_dw_alpha) for agent in range(250)]
    testing_environment.train(5000, "SA", Mid_SES_agents)
    dw_alphas_agents.append(Mid_SES_agents)

dw_factors_agents = []
for dw_factor in test_dw_factors:
    Mid_SES_dw_factor = CausalLearningAgent(
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
        downweigh_factor=dw_factor,
    )
    Mid_SES_agents = [copy.deepcopy(Mid_SES_dw_factor) for agent in range(250)]
    testing_environment.train(5000, "SA", Mid_SES_agents)
    dw_factors_agents.append(Mid_SES_agents)


### VISUALIZATIONS

# Struct 1
print("Structure 1")
testing_environment.post_training_visualization(struct_1_agents)
testing_environment.show_cpt_changes(struct_1_agents)
testing_environment.plot_monte_carlo(struct_1_agents, show_params=True)
testing_environment.plot_weighted_rewards(struct_1_agents, show_params=True)

# Struct 2
print("Structure 2")
testing_environment.post_training_visualization(struct_2_agents)
testing_environment.show_cpt_changes(struct_2_agents)
testing_environment.plot_monte_carlo(struct_2_agents, show_params=True)
testing_environment.plot_weighted_rewards(struct_2_agents, show_params=True)

# Struct 3
print("Structure 3")
testing_environment.post_training_visualization(struct_3_agents)
testing_environment.show_cpt_changes(struct_3_agents)
testing_environment.plot_monte_carlo(struct_3_agents, show_params=True)
testing_environment.plot_weighted_rewards(struct_3_agents, show_params=True)

# Struct 4
print("Structure 4")
testing_environment.post_training_visualization(struct_4_agents)
testing_environment.show_cpt_changes(struct_4_agents)
testing_environment.plot_monte_carlo(struct_4_agents, show_params=True)
testing_environment.plot_weighted_rewards(struct_4_agents, show_params=True)

# Struct 5
print("Structure 5")
testing_environment.post_training_visualization(struct_5_agents)
testing_environment.show_cpt_changes(struct_5_agents)
testing_environment.plot_monte_carlo(struct_5_agents, show_params=True)
testing_environment.plot_weighted_rewards(struct_5_agents, show_params=True)

# Struct 6
print("Structure 6")
testing_environment.post_training_visualization(struct_6_agents)
testing_environment.show_cpt_changes(struct_6_agents)
testing_environment.plot_monte_carlo(struct_6_agents, show_params=True)
testing_environment.plot_weighted_rewards(struct_6_agents, show_params=True)

# Struct 7
print("Structure 7")
testing_environment.post_training_visualization(struct_7_agents)
testing_environment.show_cpt_changes(struct_7_agents)
testing_environment.plot_monte_carlo(struct_7_agents, show_params=True)
testing_environment.plot_weighted_rewards(struct_7_agents, show_params=True)

# Struct 8
print("Structure 8")
testing_environment.post_training_visualization(struct_8_agents)
testing_environment.show_cpt_changes(struct_8_agents)
testing_environment.plot_monte_carlo(struct_8_agents, show_params=True)
testing_environment.plot_weighted_rewards(struct_8_agents, show_params=True)

# Struct 9
print("Structure 9")
testing_environment.post_training_visualization(struct_9_agents)
testing_environment.show_cpt_changes(struct_9_agents)
testing_environment.plot_monte_carlo(struct_9_agents, show_params=True)
testing_environment.plot_weighted_rewards(struct_9_agents, show_params=True)

# Struct 10
print("Structure 10")
testing_environment.post_training_visualization(struct_10_agents)
testing_environment.show_cpt_changes(struct_10_agents)
testing_environment.plot_monte_carlo(struct_10_agents, show_params=True)
testing_environment.plot_weighted_rewards(struct_10_agents, show_params=True)

# Struct 11
print("Structure 11")
testing_environment.post_training_visualization(struct_11_agents)
testing_environment.show_cpt_changes(struct_11_agents)
testing_environment.plot_monte_carlo(struct_11_agents, show_params=True)
testing_environment.plot_weighted_rewards(struct_11_agents, show_params=True)

# Struct 13
print("Structure 13")
testing_environment.post_training_visualization(struct_13_agents)
testing_environment.show_cpt_changes(struct_13_agents)
testing_environment.plot_monte_carlo(struct_13_agents, show_params=True)
testing_environment.plot_weighted_rewards(struct_13_agents, show_params=True)

# Low SES
print("Low SES")
testing_environment.post_training_visualization(Low_SES_agents)
testing_environment.show_cpt_changes(Low_SES_agents)
testing_environment.plot_monte_carlo(Low_SES_agents, show_params=True)
testing_environment.plot_weighted_rewards(Low_SES_agents, show_params=True)

# Mid SES
print("Mid SES")
testing_environment.post_training_visualization(Mid_SES_agents)
testing_environment.show_cpt_changes(Mid_SES_agents)
testing_environment.plot_monte_carlo(Mid_SES_agents, show_params=True)
testing_environment.plot_weighted_rewards(Mid_SES_agents, show_params=True)

# High SES
print("High SES")
testing_environment.post_training_visualization(High_SES_agents)
testing_environment.show_cpt_changes(High_SES_agents)
testing_environment.plot_monte_carlo(High_SES_agents, show_params=True)
testing_environment.plot_weighted_rewards(High_SES_agents, show_params=True)

# Academic
print("Academic")
testing_environment.post_training_visualization(academic_student_agents)
testing_environment.show_cpt_changes(academic_student_agents)
testing_environment.plot_monte_carlo(academic_student_agents, show_params=True)
testing_environment.plot_weighted_rewards(academic_student_agents, show_params=True)

# Healthy
print("Healthy")
testing_environment.post_training_visualization(healthy_student_agents)
testing_environment.show_cpt_changes(healthy_student_agents)
testing_environment.plot_monte_carlo(healthy_student_agents, show_params=True)
testing_environment.plot_weighted_rewards(healthy_student_agents, show_params=True)

# Social
print("Social")
testing_environment.post_training_visualization(social_student_agents)
testing_environment.show_cpt_changes(social_student_agents)
testing_environment.plot_monte_carlo(social_student_agents, show_params=True)
testing_environment.plot_weighted_rewards(social_student_agents, show_params=True)

# Temperature
for agents in temperature_agents:
    print(f"Agent Temperature: {agents[0].temperature}")
    testing_environment.post_training_visualization(agents)
    testing_environment.show_cpt_changes(agents)
    testing_environment.plot_monte_carlo(agents, show_params=True)
    testing_environment.plot_weighted_rewards(agents, show_params=True)

# Cooling
for agents in cooling_agents:
    print(f"Agent Cooling Factor: {agents[0].cooling_factor}")
    testing_environment.post_training_visualization(agents)
    testing_environment.show_cpt_changes(agents)
    testing_environment.plot_monte_carlo(agents, show_params=True)
    testing_environment.plot_weighted_rewards(agents, show_params=True)

# CPT Increase
for agents in cpt_increase_agents:
    print(f"Agent CPT Increase Factor: {agents[0].cpt_increase_factor}")
    testing_environment.post_training_visualization(agents)
    testing_environment.show_cpt_changes(agents)
    testing_environment.plot_monte_carlo(agents, show_params=True)
    testing_environment.plot_weighted_rewards(agents, show_params=True)

# Downweigh Alpha
for agents in dw_alphas_agents:
    print(f"Agent Downweigh Alpha: {agents[0].downweigh_alpha}")
    testing_environment.post_training_visualization(agents)
    testing_environment.show_cpt_changes(agents)
    testing_environment.plot_monte_carlo(agents, show_params=True)
    testing_environment.plot_weighted_rewards(agents, show_params=True)

# Downweigh Factor
for agents in dw_factors_agents:
    print(f"Agent Downweigh Factor: {agents[0].downweigh_factor}")
    testing_environment.post_training_visualization(agents)
    testing_environment.show_cpt_changes(agents)
    testing_environment.plot_monte_carlo(agents, show_params=True)
    testing_environment.plot_weighted_rewards(agents, show_params=True)

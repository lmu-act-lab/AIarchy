from src.cla import CausalLearningAgent
from pgmpy.factors.discrete import TabularCPD  # type: ignore
from src.training_environment import TrainingEnvironment
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


# struct_1: CausalLearningAgent = CausalLearningAgent(
#     sampling_edges=[("refl_2", "refl_1")],
#     utility_edges=[("refl_2", "util_2"), ("refl_1", "util_1")],
#     cpts=[
#         TabularCPD(variable="refl_2", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(
#             variable="refl_1",
#             variable_card=2,
#             values=[[0.5, 0.5], [0.5, 0.5]],
#             evidence=["refl_2"],
#             evidence_card=[2],
#         ),
#     ],
#     utility_vars={"util_1", "util_2"},
#     reflective_vars={"refl_1", "refl_2"},
#     chance_vars=set(),
#     glue_vars=set(),
#     reward_func=testing_environment.default_reward,
#     fixed_evidence={},
# )

# struct_2: CausalLearningAgent = CausalLearningAgent(
#     sampling_edges=[("chance_1", "refl_1"), ("chance_2", "refl_1")],
#     utility_edges=[("refl_1", "util_1")],
#     cpts=[
#         TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(variable="chance_2", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(
#             variable="refl_1",
#             variable_card=2,
#             values=[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
#             evidence=["chance_1", "chance_2"],
#             evidence_card=[2, 2],
#         ),
#     ],
#     utility_vars={"util_1"},
#     reflective_vars={"refl_1"},
#     chance_vars={"chance_1", "chance_2"},
#     glue_vars=set(),
#     reward_func=testing_environment.default_reward,
#     fixed_evidence={},
# )

# struct_3: CausalLearningAgent = CausalLearningAgent(
#     sampling_edges=[("refl_1", "chance_1"), ("refl_2", "chance_1")],
#     utility_edges=[("chance_1", "util_1")],
#     cpts=[
#         TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(variable="refl_2", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(
#             variable="chance_1",
#             variable_card=2,
#             values=[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
#             evidence=["refl_1", "refl_2"],
#             evidence_card=[2, 2],
#         ),
#     ],
#     utility_vars={"util_1"},
#     reflective_vars={"refl_1", "refl_2"},
#     chance_vars={"chance_1"},
#     glue_vars=set(),
#     reward_func=testing_environment.default_reward,
#     fixed_evidence={},
# )

# struct_4: CausalLearningAgent = CausalLearningAgent(
#     [("chance_1", "refl_1")],
#     [("refl_1", "util_1")],
#     [
#         TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(
#             variable="refl_1",
#             variable_card=2,
#             values=[[0.5, 0.5], [0.5, 0.5]],
#             evidence=["chance_1"],
#             evidence_card=[2],
#         ),
#     ],
#     {"util_1"},
#     {"refl_1"},
#     {"chance_1"},
#     set(),
#     testing_environment.default_reward,
#     {},
# )

# struct_5: CausalLearningAgent = CausalLearningAgent(
#     [("refl_1", "chance_1")],
#     [("chance_1", "util_1")],
#     [
#         TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(
#             variable="chance_1",
#             variable_card=2,
#             values=[[0.5, 0.5], [0.5, 0.5]],
#             evidence=["refl_1"],
#             evidence_card=[2],
#         ),
#     ],
#     {"util_1"},
#     {"refl_1"},
#     {"chance_1"},
#     set(),
#     testing_environment.default_reward,
#     {},
# )

# struct_6 = CausalLearningAgent(
#     [("chance_1", "refl_1"), ("chance_1", "refl_2")],
#     [("refl_1", "util_1"), ("refl_2", "util_1")],
#     [
#         TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(
#             variable="refl_1",
#             variable_card=2,
#             values=[[0.5, 0.5], [0.5, 0.5]],
#             evidence=["chance_1"],
#             evidence_card=[2],
#         ),
#         TabularCPD(
#             variable="refl_2",
#             variable_card=2,
#             values=[[0.5, 0.5], [0.5, 0.5]],
#             evidence=["chance_1"],
#             evidence_card=[2],
#         ),
#     ],
#     {"util_1"},
#     {"refl_1", "refl_2"},
#     {"chance_1"},
#     set(),
#     testing_environment.default_reward,
#     {},
# )

# struct_7 = CausalLearningAgent(
#     [("refl_1", "chance_1"), ("refl_1", "chance_2")],
#     [("chance_1", "util_1"), ("chance_2", "util_1")],
#     [
#         TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(
#             variable="chance_1",
#             variable_card=2,
#             values=[[0.5, 0.5], [0.5, 0.5]],
#             evidence=["refl_1"],
#             evidence_card=[2],
#         ),
#         TabularCPD(
#             variable="chance_2",
#             variable_card=2,
#             values=[[0.5, 0.5], [0.5, 0.5]],
#             evidence=["refl_1"],
#             evidence_card=[2],
#         ),
#     ],
#     {"util_1"},
#     {"refl_1"},
#     {"chance_1", "chance_2"},
#     set(),
#     testing_environment.default_reward,
#     {},
# )

# struct_8: CausalLearningAgent = CausalLearningAgent(
#     [],
#     [("refl_1", "util_1"), ("chance_1", "util_1"), ("chance_2", "util_1")],
#     [
#         TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(variable="chance_2", variable_card=2, values=[[0.5], [0.5]]),
#     ],
#     {"util_1"},
#     {"refl_1"},
#     {"chance_1", "chance_2"},
#     set(),
#     testing_environment.default_reward,
#     {},
# )

# struct_9: CausalLearningAgent = CausalLearningAgent(
#     [],
#     [("refl_1", "util_1"), ("chance_1", "util_1"), ("refl_2", "util_1")],
#     [
#         TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(variable="refl_2", variable_card=2, values=[[0.5], [0.5]]),
#     ],
#     {"util_1"},
#     {"refl_1", "refl_2"},
#     {"chance_1"},
#     set(),
#     testing_environment.default_reward,
#     {},
# )

# struct_10: CausalLearningAgent = CausalLearningAgent(
#     [("chance_1", "refl_1"), ("refl_1", "chance_2")],
#     [("chance_2", "util_1")],
#     [
#         TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(
#             variable="refl_1",
#             variable_card=2,
#             values=[[0.5, 0.5], [0.5, 0.5]],
#             evidence=["chance_1"],
#             evidence_card=[2],
#         ),
#         TabularCPD(
#             variable="chance_2",
#             variable_card=2,
#             values=[[0.5, 0.5], [0.5, 0.5]],
#             evidence=["refl_1"],
#             evidence_card=[2],
#         ),
#     ],
#     {"util_1"},
#     {"refl_1"},
#     {"chance_1", "chance_2"},
#     set(),
#     testing_environment.default_reward,
#     {},
# )

# struct_11: CausalLearningAgent = CausalLearningAgent(
#     [],
#     [("refl_1", "util_1"), ("chance_1", "util_1")],
#     [
#         TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
#     ],
#     {"util_1"},
#     {"refl_1"},
#     {"chance_1"},
#     set(),
#     testing_environment.default_reward,
#     {},
# )

# struct_13: CausalLearningAgent = CausalLearningAgent(
#     [],
#     [("refl_1", "util_1"), ("refl_2", "util_1")],
#     [
#         TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
#         TabularCPD(variable="refl_2", variable_card=2, values=[[0.5], [0.5]]),
#     ],
#     {"util_1"},
#     {"refl_1", "refl_2"},
#     set(),
#     set(),
#     testing_environment.default_reward,
#     {},
# )

downweigh_struct: CausalLearningAgent = CausalLearningAgent(
    sampling_edges=[],
    utility_edges=[("refl_1", "util_1"), ("refl_1", "util_2")],
    cpts=[
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
    ],
    utility_vars={"util_1", "util_2"},
    reflective_vars={"refl_1"},
    chance_vars=set(),
    glue_vars=set(),
    reward_func=testing_environment.test_downweigh_reward,
    fixed_evidence={},
)

confounding_struct_not_hidden: CausalLearningAgent = CausalLearningAgent(
    sampling_edges=[("chance_1", "refl_1")],
    utility_edges=[("refl_1", "util_1"), ("refl_1", "util_2"), ("chance_1", "util_2")],
    cpts=[
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5, 0.5], [0.5, 0.5]], evidence=['chance_1'], evidence_card=[2]),
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]])
    ],
    utility_vars={"util_1", "util_2"},
    reflective_vars={"refl_1"},
    chance_vars={"chance_1"},
    glue_vars=set(),
    reward_func=testing_environment.test_downweigh_reward,
    fixed_evidence={},
)

confounding_struct_hidden: CausalLearningAgent = CausalLearningAgent(
    sampling_edges=[("chance_1", "refl_1")],
    utility_edges=[("refl_1", "util_1"), ("refl_1", "util_2"), ("chance_1", "util_2")],
    cpts=[
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5, 0.5], [0.5, 0.5]], evidence=['chance_1'], evidence_card=[2]),
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]])
    ],
    utility_vars={"util_1", "util_2"},
    reflective_vars={"refl_1"},
    chance_vars={"chance_1"},
    glue_vars=set(),
    reward_func=testing_environment.test_downweigh_reward,
    fixed_evidence={},
    hidden_vars=["chance_1"]
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
    fixed_evidence={"SES": 1},
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
    fixed_evidence={"SES": 1},
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
    fixed_evidence={"SES": 1},
    weights={"grades": 0.25, "health": 0.25, "social": 0.5},
)

sampling_edges = [
    ("SES", "Tutoring"), ("SES", "ECs"), ("SES", "Time studying"),
    ("Motivation", "Time studying"), ("SES", "Exercise"),
    ("ECs", "Time studying"), ("Time studying", "Sleep")
]

utility_edges = [
    ("Time studying", "grades"), ("Tutoring", "grades"),
    ("ECs", "social"), ("Sleep", "health"), ("Exercise", "health"),
]

reflective_vars = {"Time studying", "Exercise", "Sleep", "ECs"}
chance_vars     = {"Tutoring", "Motivation", "SES"}
glue_vars       = set()
utility_vars    = {"grades", "social", "health"}

def make_student(ses_value, weights, T, cool, frustr, reward_noise):
    return CausalLearningAgent(
        sampling_edges=sampling_edges,
        utility_edges=utility_edges,
        cpts=list_of_cpts,                       # includes Grade leniency CPD
        utility_vars=utility_vars,
        reflective_vars=reflective_vars,
        chance_vars=chance_vars,
        glue_vars=glue_vars,                     # teacher can overwrite leniency
        reward_func=testing_environment.default_reward,
        fixed_evidence={"SES": ses_value},
        weights=dict(zip(("grades","social","health"), weights)),
        temperature=T,
        cooling_factor=cool,
        frustration_threshold=frustr,
        reward_noise=reward_noise
    )

# ----------------------------------------
# Helper: generic Bernoulli 0.5 CPT
def bernoulli_cpd(var):
    return TabularCPD(variable=var, variable_card=2, values=[[0.5], [0.5]])

# ----------------------------------------
# C1  Confounder (observed vs. hidden)
confound_obs = CausalLearningAgent(
    sampling_edges=[("chance", "refl")],
    utility_edges=[("refl", "util"), ("chance", "util")],
    cpts=[
        bernoulli_cpd("chance"),
        TabularCPD("refl", 2,
                   [[0.8, 0.2],  # P(refl=0|chance)
                    [0.2, 0.8]], evidence=["chance"], evidence_card=[2]),
    ],
    utility_vars={"util"},
    reflective_vars={"refl"},
    chance_vars={"chance"},
    glue_vars=set(),
    reward_func=testing_environment.default_reward,
)
confound_hidden = copy.deepcopy(confound_obs)
confound_hidden.hidden_vars = ["chance"]  

# ----------------------------------------
# C2  Fork (common cause; refl_B irrelevant)
fork_struct = CausalLearningAgent(
    sampling_edges=[("chance", "refl_A"), ("chance", "refl_B")],
    utility_edges=[("refl_A", "util")],  # refl_B has no path
    cpts=[
        bernoulli_cpd("chance"),
        TabularCPD("refl_A", 2,
                   [[0.9, 0.1], [0.1, 0.9]], evidence=["chance"], evidence_card=[2]),
        TabularCPD("refl_B", 2,
                   [[0.7, 0.3], [0.3, 0.7]], evidence=["chance"], evidence_card=[2]),
    ],
    utility_vars={"util"},
    reflective_vars={"refl_A", "refl_B"},
    chance_vars={"chance"},
    hidden_vars=["chance"],
    glue_vars=set(),
    reward_func=testing_environment.default_reward,
)

# ----------------------------------------
# C3  Mediation chain (observed and hidden mid)
mediation_obs = CausalLearningAgent(
    sampling_edges=[("refl", "mid")],
    utility_edges=[("mid", "util")],
    cpts=[
        bernoulli_cpd("refl"),
        TabularCPD("mid", 2,
                   [[0.8, 0.2], [0.2, 0.8]], evidence=["refl"], evidence_card=[2]),
    ],
    utility_vars={"util"},
    reflective_vars={"refl"},
    chance_vars={"mid"},    
    glue_vars=set(),          # mediator is chance
    reward_func=testing_environment.default_reward,
)

mediation_hidden = copy.deepcopy(mediation_obs)
mediation_hidden.hidden_vars = ["mid"]  # mid is hidden

base_structs = {
    "confound_obs": confound_obs,
    "confound_hidden": confound_hidden,
    "fork": fork_struct,
    "mediation_obs": mediation_obs,
    "mediation_hidden": mediation_hidden,
    "downweigh_struct": downweigh_struct,
}


students = {
    "Amy":    make_student(2, (0.60, 0.20, 0.20), 3.00, 0.995, 0.30, 0),
    "Ben":    make_student(1, (0.45, 0.35, 0.20), 1.00, 0.980, 0.20, 0.25),
    "Carlos": make_student(0, (0.30, 0.55, 0.15), 2.00, 0.970, 0.10, 0.5),
    "Dana":   make_student(1, (0.25, 0.25, 0.50), 0.50, 0.990, 0.28, 0),
    "Ethan":  make_student(2, (0.70, 0.15, 0.15), 0.25, 0.940, 0.22, 0.25),
    "Farah":  make_student(0, (0.40, 0.25, 0.35), 3.00, 0.993, 0.12, 0.5),
    "Grace":  make_student(1, (0.33, 0.33, 0.34), 1.00, 0.985, 0.20, 0),
    "Hiro":   make_student(2, (0.55, 0.10, 0.35), 0.50, 0.970, 0.30, 0.25),
    "Isla":   make_student(0, (0.35, 0.50, 0.15), 3.00, 0.998, 0.18, 0.5),
    "Jonas":  make_student(1, (0.20, 0.30, 0.50), 0.25, 0.920, 0.08, 0),
    "Kiara":  make_student(2, (0.48, 0.40, 0.12), 2.00, 0.990, 0.27, 0.25),
    "Leo":    make_student(0, (0.28, 0.22, 0.50), 0.10, 0.950, 0.19, 0.5),
    "Mei":    make_student(1, (0.50, 0.25, 0.25), 3.00, 0.996, 0.11, 0),
    "Nikhil": make_student(2, (0.37, 0.43, 0.20), 1.00, 0.980, 0.17, 0.25),
    "Olivia": make_student(1, (0.60, 0.10, 0.30), 0.50, 0.930, 0.25, 0.5),
}

param_variants: dict[str, CausalLearningAgent] = {}

for name, agent in base_structs.items():
    # 1. Reward‐noise variant
    a_noise = copy.deepcopy(agent)
    a_noise.reward_noise = 0.2
    param_variants[f"{name}_noise02"] = a_noise

    # 2. High‐sample variant
    a_samples = copy.deepcopy(agent)
    a_samples.sample_num = 32
    param_variants[f"{name}_samples32"] = a_samples

    # 3. Fast‐adaptation variant
    a_fast = copy.deepcopy(agent)
    a_fast.downweigh_alpha = 0.1
    a_fast.cpt_increase_factor = 0.05
    param_variants[f"{name}_fastAdapt"] = a_fast

    # 4. High‐exploration variant
    a_explore = copy.deepcopy(agent)
    a_explore.temperature = 5.0
    a_explore.cooling_factor = 0.999
    param_variants[f"{name}_highExplore"] = a_explore

# Merge base + parameter variants
all_structs = {**base_structs, **param_variants}
print(len(all_structs))

# Using items() + len() + list slicing 
# Split dictionary by half
res1 = dict(list(all_structs.items())[len(all_structs)//2:]) 
res2 = dict(list(all_structs.items())[:len(all_structs)//2]) 



# base_structures = {
#     "struct_1": struct_1,
#     "struct_2": struct_2,
#     "struct_3": struct_3,
#     "struct_4": struct_4,
#     "struct_5": struct_5,
#     "struct_6": struct_6,
#     "struct_7": struct_7,
#     "struct_8": struct_8,
#     "struct_9": struct_9,
#     "struct_10": struct_10,
#     "struct_11": struct_11,
#     "struct_13": struct_13,
# }

# base_models = {
#     "Low SES": Low_SES,
#     "Mid SES": Mid_SES,
#     "High SES": High_SES,
#     "Academic Student": academic_student,
#     "Healthy Student": healthy_student,
#     "Social Student": social_student,
#     "Downweigh Structure": downweigh_struct,
#     "Confounding Structure (Not Hidden)": confounding_struct_not_hidden,
#     "Confounding Structure (Hidden)": confounding_struct_hidden,
# }







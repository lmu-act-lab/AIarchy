from pgmpy.factors.discrete import TabularCPD  # type: ignore
from src.cla import CausalLearningAgent
from src.hierarchy import Hierarchy


def make_teacher() -> CausalLearningAgent:
    """
    Create a teacher agent that:
    - Shares glue var 'grade_leniency' with students
    - Optimizes utilities: 'grades' (student outcomes) and 'teacher_regulation'
    - Has structural edges so that 'grade_leniency' affects both utilities

    Returns
    -------
    CausalLearningAgent
        The teacher agent instance.
    """
    sampling_edges = [
        ("class_composition", "curriculum_complexity"),
        # Note: utilities ('grades', 'teacher_regulation') are NOT part of the sampling graph.
        # They live only in the structural (utility) graph.
    ]

    utility_edges = [
        ("grade_leniency", "class_performance"),
        ("curriculum_complexity", "class_performance"),
    ]

    cpts = [
        TabularCPD("class_composition", 2, [[0.5], [0.5]]),
        TabularCPD(
            variable="curriculum_complexity",
            variable_card=3,
            values=[[0.7, 0.1], [0.2, 0.2], [0.1, 0.7]],
            evidence=["class_composition"],
            evidence_card=[2],
        ),
        TabularCPD(variable="grade_leniency", variable_card=2, values=[[0.5], [0.5]]),
    ]

    teacher = CausalLearningAgent(
        sampling_edges=sampling_edges,
        utility_edges=utility_edges,
        cpts=cpts,
        utility_vars={"class_performance"},
        reflective_vars={"grade_leniency", "curriculum_complexity"},
        chance_vars={"class_composition"},
        glue_vars={"grade_leniency"},
        reward_func=_teacher_reward,
        fixed_evidence={"class_composition": 0},
    )
    return teacher


def build_classroom_hierarchy(
    students: list[CausalLearningAgent],
    pooling_vars: list[str] | None = None,
    glue_vars: list[str] | None = None,
    monte_carlo_samples: int = 100,
) -> Hierarchy:
    """
    Build a Hierarchy that connects a single teacher (parent) to multiple students (children).

    Parameters
    ----------
    students : list[CausalLearningAgent]
        The student agents that will be trained at the lower tier.
    pooling_vars : list[str] | None
        Utility variables to pool from children up to the teacher. Defaults to ["grades"].
    glue_vars : list[str] | None
        Glue variables to propagate from teacher to students. Defaults to ["grade_leniency"].
    monte_carlo_samples : int
        Number of Monte Carlo replicas per agent.

    Returns
    -------
    Hierarchy
        Configured hierarchy ready for `train(cycles, parent_iter, child_iter)`.
    """
    if pooling_vars is None:
        pooling_vars = ["grades"]
    if glue_vars is None:
        glue_vars = ["grade_leniency"]

    teacher = make_teacher()
    mapping: dict[CausalLearningAgent, list[CausalLearningAgent]] = {teacher: students}
    return Hierarchy(
        mapping,
        pooling_vars=pooling_vars,
        glue_vars=glue_vars,
        monte_carlo_samples=monte_carlo_samples,
    )


def _teacher_reward(
    sample: dict[str, int],
    utility_edges: list[tuple[str, str]],
    noise: float = 0.0,
    lower_tier_pooled_reward: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Reward for the teacher that incorporates pooled lower-tier rewards.

    - For 'class_performance', lower leniency is better.
    """
    from src.util import Counter
    import random

    rewards: dict[str, float] = Counter()
    noise_term = random.gauss(0, noise) if noise > 0 else 0.0

    for var, util in utility_edges:
        if util == "class_performance":
            rewards[util] += sample.get("curriculum_complexity", 0)
            if lower_tier_pooled_reward:
                for util_name, pooled_val in lower_tier_pooled_reward.items():
                    rewards[util] += pooled_val
    return rewards

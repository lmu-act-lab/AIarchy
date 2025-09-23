import os
from pathlib import Path
import warnings

from src.teacher import build_classroom_hierarchy
from src.final_actlab_sims import models as M
from src.training_environment import TrainingEnvironment

warnings.filterwarnings("ignore")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Output roots (mirrors act_lab style saving)
    out_root = "src/final_actlab_sims/actlab_main/teacher_hierarchy"
    teacher_dir = f"{out_root}/Teacher"

    for d in (teacher_dir,):
        ensure_dir(d)

    # Build students using the updated make_student (with grade_leniency)
    ben = M.make_student(1, (0.45, 0.35, 0.20), 1.00, 0.980, 0.20, 0)
    dana = M.make_student(1, (0.25, 0.25, 0.50), 0.50, 0.990, 0.28, 0)
    ethan = M.make_student(2, (0.70, 0.15, 0.15), 0.25, 0.940, 0.22, 0)

    # Build hierarchy with pooled 'grades' and glue 'grade_leniency'
    hier = build_classroom_hierarchy(
        [ben, dana, ethan],
        pooling_vars=["grades"],
        glue_vars=["grade_leniency"],
        monte_carlo_samples=100,
    )

    # Train: cycles of child -> pool -> parent -> glue
    hier.train(cycles=52, parent_iter=1, child_iter=7)

    # Visualize/save similar to other sims
    TE = TrainingEnvironment()

    # Teacher visuals (use first parent's MC replica set)
    if hier.parents:
        parent_agents = hier.parents[0]
        TE.pre_training_visualization(parent_agents[0], name=teacher_dir, save=True)
        TE.post_training_visualization(parent_agents, name=teacher_dir, save=True)
        TE.plot_monte_carlo(parent_agents, name=teacher_dir, save=True)
        TE.plot_weighted_rewards(parent_agents, name=teacher_dir, save=True)
        TE.plot_u_hat_model_losses(parent_agents, name=teacher_dir, save=True)
        TE.save_reward_csv(parent_agents, name=teacher_dir)

    # Per-student visuals for each child group
    for idx, student_agents in enumerate(hier.children, start=1):
        student_dir = f"{out_root}/Student_{idx}"
        ensure_dir(student_dir)
        TE.post_training_visualization(student_agents, name=student_dir, save=True)
        TE.plot_monte_carlo(student_agents, name=student_dir, save=True)
        TE.plot_weighted_rewards(student_agents, name=student_dir, save=True)
        TE.plot_u_hat_model_losses(student_agents, name=student_dir, save=True)
        TE.show_cpt_changes(student_agents, name=student_dir, save=True)
        TE.save_reward_csv(student_agents, name=student_dir)


if __name__ == "__main__":
    # Non-interactive plotting backend recommended
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()

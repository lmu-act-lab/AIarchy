import os
from pathlib import Path
import warnings
import logging
import copy

from src.teacher import build_classroom_hierarchy
from src.final_actlab_sims import models as M
from src.training_environment import TrainingEnvironment

from src.final_actlab_sims.set_seed import set_seed

# Set up environment
set_seed()

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress pgmpy logging warnings
logging.getLogger("pgmpy").setLevel(logging.ERROR)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def main() -> None:
    # Output roots (mirrors act_lab style saving)
    out_root = "src/final_actlab_sims/actlab_main/teacher_hierarchy"
    teacher_dir = f"{out_root}/Teacher"

    for d in (teacher_dir,):
        ensure_dir(d)

    # Build students using the updated make_student (with grade_leniency)
    # ben = M.make_student(1, (0.45, 0.35, 0.20), 1.00, 0.980, 0.20, 0)
    # dana = M.make_student(1, (0.25, 0.25, 0.50), 0.50, 0.990, 0.28, 0)
    # ethan = M.make_student(2, (0.70, 0.15, 0.15), 0.25, 0.940, 0.22, 0)
    from src.final_actlab_sims.models import hierarchy_students

    # Build hierarchy with pooled 'grades' and glue 'grade_leniency'
    hier = build_classroom_hierarchy(
        list(hierarchy_students.values()),
        pooling_vars=["grades"],
        glue_vars=["grade_leniency", "curriculum_complexity"],
        monte_carlo_samples=1,
    )

    # Set up checkpoint directory
    checkpoint_dir = f"{out_root}/checkpoints"
    child_checkpoint_interval = 10   # Save child checkpoints every N iterations
    parent_checkpoint_interval = 5   # Save parent checkpoints every N iterations
    
    # Get student names for checkpoint organization
    student_names = list(hierarchy_students.keys())

    # Train: cycles of child -> pool -> parent -> glue
    hier.train(
        cycles=50, 
        parent_iter=1, 
        child_iter=5,
        checkpoint_dir=checkpoint_dir,
        child_checkpoint_interval=child_checkpoint_interval,
        parent_checkpoint_interval=parent_checkpoint_interval,
        child_names=student_names,
        parent_name="Teacher"
    )

    # Visualize/save similar to other sims
    TE = TrainingEnvironment()

    # Teacher visuals (use first parent's MC replica set)
    if hier.parents:
        parent_agents = hier.parents[0]
        TE.pre_training_visualization(parent_agents[0], name=teacher_dir, save=True)
        TE.post_training_visualization(parent_agents, name=teacher_dir, save=True)
        TE.show_cpt_changes(parent_agents, name=teacher_dir, save=True)
        TE.plot_monte_carlo(parent_agents, name=teacher_dir, save=True)
        TE.plot_weighted_rewards(parent_agents, name=teacher_dir, save=True)
        TE.plot_u_hat_model_losses(parent_agents, name=teacher_dir, save=True)
        TE.save_reward_csv(parent_agents, name=teacher_dir)

    # Per-child group visuals (similar to teacher visuals)
    if hier.children:
        student_names = list(hierarchy_students.keys())
        for i, child_agents in enumerate(hier.children):
            student_name = student_names[i]
            child_dir = f"{out_root}/{student_name}"
            ensure_dir(child_dir)
            TE.pre_training_visualization(child_agents[0], name=child_dir, save=True)
            TE.post_training_visualization(child_agents, name=child_dir, save=True)
            TE.show_cpt_changes(child_agents, name=child_dir, save=True)
            TE.plot_monte_carlo(child_agents, name=child_dir, save=True)
            TE.plot_weighted_rewards(child_agents, name=child_dir, save=True)
            TE.plot_u_hat_model_losses(child_agents, name=child_dir, save=True)
            TE.save_reward_csv(child_agents, name=child_dir)


if __name__ == "__main__":
    # Non-interactive plotting backend recommended
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()

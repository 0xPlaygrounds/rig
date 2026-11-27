use super::*;

use Status::{Canceled, Completed, InProgress, Pending};

fn plan() -> Plan {
    Plan::new(vec![
        PlanStep {
            content: "Plan".into(),
            skill: Some("plan".into()),
            command: None,
        },
        PlanStep {
            content: "Review".into(),
            skill: Some("review".into()),
            command: None,
        },
        PlanStep {
            content: "Ship".into(),
            skill: None,
            command: Some("git push".into()),
        },
    ])
}

#[test]
fn steps_advance_monotonically_on_matching_skill_or_command() {
    let mut plan = plan();
    assert!(!plan.read("/x/skills/reviewer/SKILL.md"));
    assert!(!plan.read("/x/skills/review/NOTES.md"));
    assert!(!plan.ran("git status"));
    assert_eq!(plan.statuses(), [Pending, Pending, Pending]);

    assert!(plan.read("review/SKILL.md"));
    assert_eq!(plan.statuses(), [Completed, InProgress, Pending]);
    // Repeating the current or an earlier step is a no-op.
    assert!(!plan.read("/x/skills/review/SKILL.md"));
    assert!(!plan.read("/x/skills/plan/SKILL.md"));

    assert!(plan.ran("cd repo && git push origin main"));
    assert_eq!(plan.statuses(), [Completed, Completed, InProgress]);
    assert!(!plan.read("/x/skills/review/SKILL.md"));

    let mut empty_cmd = Plan::new(vec![PlanStep {
        content: "Any".into(),
        skill: None,
        command: Some(String::new()),
    }]);
    assert!(!empty_cmd.ran("anything"));
    assert_eq!(empty_cmd.statuses(), [Pending]);
}

#[test]
fn ending_a_run_completes_cancels_or_preserves_remaining_steps() {
    let mut active = plan();
    assert!(active.read("/x/skills/review/SKILL.md"));
    let before = active.clone();
    assert!(!active.end(End::NeedsInput));
    assert_eq!(active, before);

    assert!(active.end(End::Failed));
    assert_eq!(active.statuses(), [Completed, Canceled, Canceled]);

    let mut unstarted = plan();
    assert!(unstarted.end(End::Failed));
    assert_eq!(unstarted.statuses(), [Canceled, Canceled, Canceled]);

    let mut finished = plan();
    assert!(finished.end(End::Done));
    assert_eq!(finished.statuses(), [Completed, Completed, Completed]);
    assert!(!finished.end(End::Done));
}

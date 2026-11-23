use super::*;

fn skill(content: &str, skill: &str) -> PlanStep {
    PlanStep {
        content: content.into(),
        skill: Some(skill.into()),
        command: None,
    }
}

fn command(content: &str, command: &str) -> PlanStep {
    PlanStep {
        content: content.into(),
        skill: None,
        command: Some(command.into()),
    }
}

/// Plan, review, ship: the second by skill, the third by command.
fn plan() -> Plan {
    Plan::new(vec![
        skill("Plan", "plan"),
        skill("Review", "review"),
        command("Ship", "git push"),
    ])
}

use Status::{Canceled, Completed, InProgress, Pending};

#[test]
fn a_new_plan_is_all_pending() {
    assert_eq!(plan().statuses(), [Pending, Pending, Pending]);
    assert!(!plan().is_empty());
    assert!(Plan::default().is_empty());
}

#[test]
fn a_plan_is_json_with_content_and_status() {
    assert_eq!(
        plan().json(),
        json!([
            { "content": "Plan", "status": "pending" },
            { "content": "Review", "status": "pending" },
            { "content": "Ship", "status": "pending" },
        ])
    );
}

#[test]
fn an_in_progress_status_serialises_in_camel_case() {
    assert_eq!(json!(Status::InProgress), json!("inProgress"));
    assert_eq!(json!(Status::Completed), json!("completed"));
    assert_eq!(json!(Status::Canceled), json!("canceled"));
}

#[test]
fn a_skill_read_starts_its_step_and_completes_the_ones_before() {
    let mut plan = plan();
    assert!(plan.read("/x/skills/review/SKILL.md"));
    assert_eq!(plan.statuses(), [Completed, InProgress, Pending]);
    assert_eq!(
        plan.json(),
        json!([
            { "content": "Plan", "status": "completed" },
            { "content": "Review", "status": "inProgress" },
            { "content": "Ship", "status": "pending" },
        ])
    );
}

#[test]
fn a_relative_skill_path_starts_its_step_too() {
    let mut plan = plan();
    assert!(plan.read("review/SKILL.md"));
    assert_eq!(plan.statuses(), [Completed, InProgress, Pending]);
}

#[test]
fn a_skill_whose_name_only_ends_the_same_does_not_start_the_step() {
    let mut plan = plan();
    assert!(!plan.read("/x/skills/reviewer/SKILL.md"));
    assert!(!plan.read("/x/skills/review/NOTES.md"));
    assert_eq!(plan.statuses(), [Pending, Pending, Pending]);
}

#[test]
fn a_command_containing_the_steps_command_starts_it() {
    let mut plan = plan();
    assert!(plan.ran("cd repo && git push origin main"));
    assert_eq!(plan.statuses(), [Completed, Completed, InProgress]);
}

#[test]
fn a_command_that_is_not_the_steps_starts_nothing() {
    let mut plan = plan();
    assert!(!plan.ran("git status"));
    assert_eq!(plan.statuses(), [Pending, Pending, Pending]);
}

#[test]
fn a_step_with_an_empty_command_never_starts() {
    let mut plan = Plan::new(vec![command("Any", "")]);
    assert!(!plan.ran("anything"));
    assert_eq!(plan.statuses(), [Pending]);
}

#[test]
fn an_earlier_step_does_not_start_again_once_a_later_one_has() {
    let mut plan = plan();
    assert!(plan.ran("git push"));
    let after = plan.clone();
    assert!(!plan.read("/x/skills/plan/SKILL.md"));
    assert!(!plan.read("/x/skills/review/SKILL.md"));
    assert_eq!(plan, after);
}

#[test]
fn the_step_in_progress_does_not_start_again() {
    let mut plan = plan();
    assert!(plan.read("plan/SKILL.md"));
    assert!(!plan.read("plan/SKILL.md"));
    assert_eq!(plan.statuses(), [InProgress, Pending, Pending]);
}

#[test]
fn a_done_run_completes_every_step() {
    let mut plan = plan();
    assert!(plan.end(End::Done));
    assert_eq!(plan.statuses(), [Completed, Completed, Completed]);
    assert!(!plan.end(End::Done));
}

#[test]
fn a_failed_run_cancels_the_current_step_and_those_after_it() {
    let mut plan = plan();
    assert!(plan.read("review/SKILL.md"));
    assert!(plan.end(End::Failed));
    assert_eq!(plan.statuses(), [Completed, Canceled, Canceled]);
}

#[test]
fn a_failed_run_with_no_step_started_cancels_them_all() {
    let mut plan = plan();
    assert!(plan.end(End::Failed));
    assert_eq!(plan.statuses(), [Canceled, Canceled, Canceled]);
}

#[test]
fn a_failed_empty_plan_does_not_change() {
    assert!(!Plan::default().end(End::Failed));
}

#[test]
fn a_run_waiting_on_a_person_leaves_the_plan_as_it_is() {
    let mut plan = plan();
    assert!(plan.read("review/SKILL.md"));
    let before = plan.clone();
    assert!(!plan.end(End::NeedsInput));
    assert_eq!(plan, before);
}

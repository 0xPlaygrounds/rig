use bevy_ecs::entity::Entity;

use super::{HeldReport, Report, TaskArgs, reaches};
use crate::agent::AgentId;
use crate::inbox::{DeliveryMode, Origin, RequestId};

#[test]
fn reaches_follows_edges_transitively() {
    let edges = [(1, 2), (2, 3), (4, 1)];
    assert!(reaches(&edges, 1, 2));
    assert!(reaches(&edges, 1, 3));
    assert!(reaches(&edges, 4, 3));
    assert!(!reaches(&edges, 3, 1));
    assert!(!reaches(&edges, 2, 4));
}

#[test]
fn reaches_finds_cycles_and_ends_on_them() {
    let edges = [(1, 2), (2, 1), (2, 3)];
    assert!(reaches(&edges, 1, 1));
    assert!(reaches(&edges, 2, 3));
    assert!(!reaches(&edges, 3, 1));
    assert!(!reaches(&[(5, 6)], 1, 1));
}

#[test]
fn report_header_names_the_subagent_and_task_not_the_request() {
    let origin = Origin::agent(
        AgentId("fa7bc7dc-8bed-4938".to_owned()),
        Some(RequestId("call_00_rIBjJGaY".to_owned())),
    )
    .titled("Fix the parser");
    assert_eq!(
        origin.header().as_deref(),
        Some("[Output of agent fa7bc7dc \"Fix the parser\", not the user's words]")
    );
    assert_eq!(
        origin.request,
        Some(RequestId("call_00_rIBjJGaY".to_owned()))
    );
}

#[test]
fn tasks_report_together_unless_asked_alone() {
    let args: TaskArgs = serde_json::from_value(serde_json::json!({
        "description": "d",
        "prompt": "p",
    }))
    .expect("parses");
    assert_eq!(args.report, Report::Together);
    let args: TaskArgs = serde_json::from_value(serde_json::json!({
        "description": "d",
        "prompt": "p",
        "report": "alone",
    }))
    .expect("parses");
    assert_eq!(args.report, Report::Alone);
}

#[test]
fn held_reports_keep_their_delivery_mode() {
    let held = |note| HeldReport {
        text: "answer".to_owned(),
        origin: Origin::user(),
        note,
    };
    assert_eq!(
        held(true).deliver(Entity::PLACEHOLDER).mode,
        DeliveryMode::Note
    );
    assert_eq!(
        held(false).deliver(Entity::PLACEHOLDER).mode,
        DeliveryMode::Queue
    );
}

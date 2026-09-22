use super::*;

#[test]
fn pending_calls_are_reserved_once() {
    let mut world = World::new();
    let entity = world.spawn_empty().id();
    let mut budget = ToolBudget {
        admitted: 59,
        seen: HashSet::new(),
    };
    budget.admit(vec![entity]);
    budget.admit(vec![entity]);
    assert_eq!(budget.admitted, 60);
}

#[test]
#[should_panic(expected = "whole-task tool dispatch budget exhausted")]
fn an_oversized_batch_is_rejected_before_dispatch() {
    let mut world = World::new();
    let entities = vec![world.spawn_empty().id(), world.spawn_empty().id()];
    let mut budget = ToolBudget {
        admitted: 59,
        seen: HashSet::new(),
    };
    budget.admit(entities);
}

#[test]
fn continuation_uses_only_the_remaining_model_budget() {
    assert_eq!(super::super::remaining_turns(17), 13);
}

#[test]
#[should_panic(expected = "whole-task completion budget exhausted")]
fn an_exhausted_task_cannot_start_another_run() {
    super::super::remaining_turns(30);
}

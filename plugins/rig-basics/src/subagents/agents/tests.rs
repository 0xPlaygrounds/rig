use super::working;

#[test]
fn one_agent_at_work_is_counted_in_the_singular() {
    assert_eq!(working("+", "other agent", 0), "");
    assert_eq!(
        working("+", "other agent", 1),
        "+1 other agent working (/agents)"
    );
    assert_eq!(
        working("+", "other agent", 2),
        "+2 other agents working (/agents)"
    );
    assert_eq!(working("", "subagent", 1), "1 subagent working (/agents)");
}

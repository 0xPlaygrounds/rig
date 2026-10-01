use super::*;

/// The generated inputs are fixed text, so a re-recording sends the bytes
/// its cassette holds, and they are the sizes the runs were costed at.
/// Offline because they never reach a provider on their own: the recorded
/// runs replay them.
#[test]
fn generated_inputs_are_deterministic_and_sized() {
    let history = order_history("C001-a").to_string();
    assert_eq!(history, order_history("C001-a").to_string());
    assert_ne!(history, order_history("C001-b").to_string());
    // About a thousand tokens at roughly four bytes a token.
    assert!((3_000..7_000).contains(&history.len()), "{}", history.len());

    let log = shipping_log("TRK00000001");
    assert_eq!(log, shipping_log("TRK00000001"));
    assert!((2_500..6_000).contains(&log.len()), "{}", log.len());

    let policy = policy_text();
    assert_eq!(policy, policy_text());
    // About eight thousand tokens.
    assert!((24_000..40_000).contains(&policy.len()), "{}", policy.len());
}

/// The schedule cycles through its sets every `every` turns. Offline:
/// which set a turn gets is local arithmetic; the recorded dynamic-tools
/// runs show the sets on the wire.
#[test]
fn tool_schedule_switches_every_n_turns() {
    let schedule = ToolSchedule::new(10, vec![vec!["a"], vec!["a", "b"]]);
    assert_eq!(schedule.tools_for(1), ["a"]);
    assert_eq!(schedule.tools_for(10), ["a"]);
    assert_eq!(schedule.tools_for(11), ["a", "b"]);
    assert_eq!(schedule.tools_for(21), ["a"]);
}

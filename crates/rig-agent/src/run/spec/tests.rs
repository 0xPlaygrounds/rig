use super::*;

#[test]
fn spec_round_trips_through_json() {
    let spec = RunSpec {
        preamble: Some("be brief".into()),
        max_turns: Some(3),
        temperature: Some(0.2),
        output_schema: Some(serde_json::json!({"type": "object"})),
        ..RunSpec::new()
    };
    let json = serde_json::to_string(&spec).expect("serialize");
    let back: RunSpec = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, spec);
}

/// `Default` is `new`, and a spec that omits every field deserializes to it.
#[test]
fn default_equals_new() {
    assert_eq!(RunSpec::default(), RunSpec::new());
    assert!(RunSpec::default().augment_output_preamble);
    let spec: RunSpec = serde_json::from_str("{}").expect("deserialize");
    assert_eq!(spec, RunSpec::new());
}

/// The malformed-arguments limit defaults to none, also for a spec that
/// omits it. Both `None` and `Some` survive a round trip through the spec and
/// through a run built from it.
#[test]
fn malformed_call_limit_defaults_to_none_and_round_trips() {
    assert_eq!(
        RunSpec::default().max_consecutive_malformed_tool_calls,
        None
    );
    assert_eq!(RunSpec::new().max_consecutive_malformed_tool_calls, None);
    let spec: RunSpec = serde_json::from_str("{}").expect("deserialize");
    assert_eq!(spec.max_consecutive_malformed_tool_calls, None);

    for limit in [None, Some(5)] {
        let spec = RunSpec {
            max_consecutive_malformed_tool_calls: limit,
            ..RunSpec::new()
        };
        let json = serde_json::to_string(&spec).expect("serialize spec");
        let back: RunSpec = serde_json::from_str(&json).expect("deserialize spec");
        assert_eq!(back.max_consecutive_malformed_tool_calls, limit);

        let run = crate::run::AgentRun::from_spec(&spec, "p", None);
        let json = serde_json::to_value(&run).expect("serialize run");
        assert_eq!(
            json["max_consecutive_malformed_tool_calls"],
            serde_json::json!(limit)
        );
        assert_eq!(json["malformed_tool_call_retries"], 0);
        let back: crate::run::AgentRun = serde_json::from_value(json).expect("deserialize run");
        let json = serde_json::to_value(&back).expect("serialize resumed run");
        assert_eq!(
            json["max_consecutive_malformed_tool_calls"],
            serde_json::json!(limit)
        );
    }
}

/// A run spec and a prepared request stay unwind safe with provider
/// options in their request.
#[test]
fn run_requests_stay_unwind_safe() {
    fn unwind_safe<T: std::panic::UnwindSafe + std::panic::RefUnwindSafe>() {}
    unwind_safe::<crate::run::RunSpec>();
    unwind_safe::<crate::run::PreparedRequest>();
    unwind_safe::<crate::agent::hook::DispatchAction>();
}

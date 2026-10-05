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

/// The malformed-arguments limit defaults to none, also for a spec that
/// omits it. Both `None` and `Some` survive a round trip through the spec and
/// through a run built from it.
#[test]
fn malformed_tool_call_retries_default_to_none_and_round_trip() {
    assert_eq!(RunSpec::default().max_malformed_tool_call_retries, None);
    assert_eq!(RunSpec::new().max_malformed_tool_call_retries, None);
    let spec: RunSpec = serde_json::from_str("{}").expect("deserialize");
    assert_eq!(spec.max_malformed_tool_call_retries, None);

    for limit in [None, Some(5)] {
        let spec = RunSpec {
            max_malformed_tool_call_retries: limit,
            ..RunSpec::new()
        };
        let json = serde_json::to_string(&spec).expect("serialize spec");
        let back: RunSpec = serde_json::from_str(&json).expect("deserialize spec");
        assert_eq!(back.max_malformed_tool_call_retries, limit);

        let run = crate::run::AgentRun::from_spec(&spec, "p", None);
        let json = serde_json::to_value(&run).expect("serialize run");
        assert_eq!(
            json["max_malformed_tool_call_retries"],
            serde_json::json!(limit)
        );
        assert_eq!(json["malformed_tool_call_retries"], 0);
        let back: crate::run::AgentRun = serde_json::from_value(json).expect("deserialize run");
        let json = serde_json::to_value(&back).expect("serialize resumed run");
        assert_eq!(
            json["max_malformed_tool_call_retries"],
            serde_json::json!(limit)
        );
    }
}

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

/// The malformed-arguments limit defaults to three, also for a spec that
/// omits it, and a run built from the spec carries it.
#[test]
fn malformed_tool_call_retries_default_to_three() {
    assert_eq!(RunSpec::default().max_malformed_tool_call_retries, 3);
    assert_eq!(RunSpec::new().max_malformed_tool_call_retries, 3);
    let spec: RunSpec = serde_json::from_str("{}").expect("deserialize");
    assert_eq!(spec.max_malformed_tool_call_retries, 3);

    let spec = RunSpec {
        max_malformed_tool_call_retries: 5,
        ..RunSpec::new()
    };
    let run = crate::run::AgentRun::from_spec(&spec, "p", None);
    let json = serde_json::to_value(&run).expect("serialize run");
    assert_eq!(json["max_malformed_tool_call_retries"], 5);
    assert_eq!(json["malformed_tool_call_retries"], 0);
}

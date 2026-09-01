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

#[test]
fn missing_fields_take_defaults() {
    let spec: RunSpec = serde_json::from_str("{}").expect("deserialize");
    assert_eq!(spec.effective_max_turns(), 1);
    assert!(spec.output_schema.is_none());
}

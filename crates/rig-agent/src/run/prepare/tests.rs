use super::*;

/// A prepared request produced by `prepare_request` survives a JSON round
/// trip — a host caching one in serializable state (a saved world) can
/// restore it losslessly.
#[test]
fn prepared_request_round_trips_through_serde() {
    let spec = RunSpec {
        preamble: Some("be brief".to_string()),
        temperature: Some(0.2),
        ..RunSpec::default()
    };
    let prepared = prepare_request(
        &spec,
        &ProviderCapabilities::default(),
        &[Message::user("hi")],
        vec![ToolDefinition {
            name: rig_core::message::ToolName::new("add").expect("tool name"),
            description: "adds".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }],
        None,
        None,
    )
    .expect("prepare");
    let json = serde_json::to_string(&prepared).expect("serialize");
    let restored: PreparedRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored, prepared);
}

#[test]
fn resolve_output_mode_auto_keeps_native_when_provider_composes() {
    // On providers that compose native structured output with tools (OpenAI,
    // Anthropic), Auto keeps guaranteed native output even with tools present.
    assert_eq!(
        resolve_output_mode(true, true, true, true, &OutputMode::Auto),
        OutputMode::Native,
    );
}

#[test]
fn resolve_output_mode_degrades_to_native_when_output_tool_not_callable() {
    // Tool mode finalizes via the output-tool call; when the tool choice
    // forbids it (None / Specific), structured output must still be enforced
    // via Native rather than silently dropped (#1928 regression guard).
    assert_eq!(
        resolve_output_mode(true, true, false, false, &OutputMode::Auto),
        OutputMode::Native,
    );
    assert_eq!(
        resolve_output_mode(true, true, false, false, &OutputMode::Tool),
        OutputMode::Native,
    );
    // Prompted does not rely on tools, so it is unaffected.
    assert_eq!(
        resolve_output_mode(true, true, false, false, &OutputMode::Prompted),
        OutputMode::Prompted,
    );
}

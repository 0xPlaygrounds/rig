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
        &[Message::user("hi").into()],
        vec![std::sync::Arc::new(ToolDefinition {
            name: rig_core::message::ToolName::new("add").expect("tool name"),
            description: "adds".to_string(),
            parameters: serde_json::json!({"type": "object"}),
            annotations: Default::default(),
        })],
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

/// Hosted tools count as advertised: they satisfy `Required` and steer Auto
/// output away from native on a provider that cannot compose the two. They
/// follow the patched `additional_params.tools` without becoming executable.
#[test]
fn hosted_tools_are_advertised_but_not_executable() {
    use rig_core::completion::ProviderToolDefinition;
    use rig_core::message::ToolChoice;

    let spec = RunSpec {
        additional_params: Some(serde_json::json!({"tools": [{"type": "spec"}]})),
        provider_tools: vec![
            ProviderToolDefinition::new("web_search"),
            ProviderToolDefinition::new("code"),
        ],
        tool_choice: Some(ToolChoice::Required),
        output_schema: Some(
            serde_json::from_value(serde_json::json!({"type": "object"})).expect("schema"),
        ),
        ..RunSpec::default()
    };
    let patch =
        RequestPatch::new().additional_params(serde_json::json!({"tools": [{"type": "patch"}]}));
    let prepared = prepare_request(
        &spec,
        &ProviderCapabilities::default(),
        &[],
        vec![],
        None,
        Some(&patch),
    )
    .expect("hosted tools satisfy Required");

    assert_eq!(prepared.output_mode, OutputMode::Tool);
    assert!(prepared.policy.executable().is_empty());
    let request = prepared.apply(CompletionRequest::new("hi"));
    assert_eq!(
        request.additional_params,
        Some(serde_json::json!({
            "tools": [{"type": "patch"}, {"type": "web_search"}, {"type": "code"}],
        }))
    );
}

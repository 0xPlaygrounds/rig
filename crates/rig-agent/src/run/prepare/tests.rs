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
            annotations: Default::default(),
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

/// A Native turn to a provider that drops the schema marks its policy, which
/// survives serde, so the run checks the answer; a carrying provider's does not.
#[test]
fn a_native_schema_the_provider_drops_marks_the_turn_unenforced() {
    let spec = RunSpec {
        output_schema: Some(serde_json::json!({"type": "object", "required": ["answer"]})),
        ..RunSpec::default()
    };
    let prepare = |capabilities: ProviderCapabilities| {
        prepare_request(
            &spec,
            &capabilities,
            &[Message::user("hi")],
            vec![],
            None,
            None,
        )
        .expect("prepare")
    };

    let dropped = prepare(ProviderCapabilities::default().with_output_schema_dropped(true));
    assert_eq!(dropped.output_mode, OutputMode::Native);
    assert!(dropped.policy.unenforced_schema());
    let json = serde_json::to_string(&dropped).expect("serialize");
    let restored: PreparedRequest = serde_json::from_str(&json).expect("deserialize");
    assert!(restored.policy.unenforced_schema());

    assert!(
        !prepare(ProviderCapabilities::default())
            .policy
            .unenforced_schema()
    );

    // A wire that defers the schema while tools are advertised leaves a turn
    // with tools unenforced, and one without tools enforced.
    let deferring = ProviderCapabilities::default()
        .with_native_output_tool_composition(true)
        .with_output_schema_deferred_with_tools(true);
    assert!(!prepare(deferring).policy.unenforced_schema());
    let with_tool = prepare_request(
        &spec,
        &deferring,
        &[Message::user("hi")],
        vec![ToolDefinition {
            name: rig_core::message::ToolName::new("add").expect("tool name"),
            description: "adds".to_string(),
            parameters: serde_json::json!({"type": "object"}),
            annotations: Default::default(),
        }],
        None,
        None,
    )
    .expect("prepare");
    assert_eq!(with_tool.output_mode, OutputMode::Native);
    assert!(with_tool.policy.unenforced_schema());
    // So does a raw tool in `additional_params`, which the wire also sends.
    let raw = RunSpec {
        additional_params: Some(serde_json::json!({"tools": [{"type": "function"}]})),
        ..spec.clone()
    };
    let with_raw_tool =
        prepare_request(&raw, &deferring, &[Message::user("hi")], vec![], None, None)
            .expect("prepare");
    assert!(with_raw_tool.policy.unenforced_schema());
}

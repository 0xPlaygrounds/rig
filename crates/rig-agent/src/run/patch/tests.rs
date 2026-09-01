use super::*;

/// A patch a host may cache in serializable state round-trips.
#[test]
fn request_patch_round_trips_through_serde() {
    let patch = RequestPatch {
        preamble: Some("p".to_string()),
        temperature: Some(0.5),
        max_tokens: Some(64),
        tool_choice: Some(ToolChoice::Auto),
        active_tools: Some(vec!["add".to_string()]),
        additional_params: Some(serde_json::json!({"k": 1})),
        extra_context: Vec::new(),
        history: Some(vec![Message::user("hi")]),
    };
    let json = serde_json::to_string(&patch).expect("serialize patch");
    assert_eq!(
        serde_json::from_str::<RequestPatch>(&json).expect("deserialize patch"),
        patch
    );
}

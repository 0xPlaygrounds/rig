use super::page;
use serde_json::json;

/// Synthetic entries cover false and malformed flags without changing a live recording.
#[test]
fn tool_call_support_keeps_explicit_values_and_unknown_distinct() {
    let page = page(json!({"data": [
        {"id": "new-model", "capabilities": {"type": "chat", "supports": {"tool_calls": true}}},
        {"id": "no-tools", "capabilities": {"type": "chat", "supports": {"tool_calls": false}}},
        {"id": "missing-flag", "capabilities": {"type": "chat", "supports": {}}},
        {"id": "missing-supports", "capabilities": {"type": "chat"}},
        {"id": "missing-capabilities"},
        {"id": "null-flag", "capabilities": {"supports": {"tool_calls": null}}},
        {"id": "invalid-flag", "capabilities": {"supports": {"tool_calls": "true"}}}
    ]}))
    .expect("decode model listing");
    assert_eq!(
        page.models
            .iter()
            .map(|model| model.supports_tool_calls)
            .collect::<Vec<_>>(),
        [Some(true), Some(false), None, None, None, None, None]
    );
}

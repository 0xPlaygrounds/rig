use super::*;
use crate::wire::Wire;

fn marked(ttl: &str) -> Value {
    json!({"type": "text", "text": "cached", "cache_control": {
        "type": "ephemeral", "ttl": ttl
    }})
}

#[test]
fn final_cache_control_rejects_raw_override_budget_ttl_and_order() {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let cases = [
        (json!({"system": vec![marked("5m"); 5]}), "budget"),
        (
            json!({"messages": [{"role": "user", "content": vec![marked("5m"); 5]}]}),
            "budget",
        ),
        (
            json!({"system": [marked("5m")], "messages": [
                {"role": "user", "content": [marked("1h")]}
            ]}),
            "must appear before",
        ),
        (json!({"system": [marked("1d")]}), "Invalid Anthropic"),
        (
            json!({"messages": [{"role": "user", "content": [marked("1d")]}]}),
            "Invalid Anthropic",
        ),
        (
            json!({"tools": [{"name": "add", "input_schema": {"type": "object"},
            "cache_control": {"type": "ephemeral", "ttl": "1d"}}]}),
            "Invalid Anthropic",
        ),
        (
            json!({"system": vec![marked("1h"); 4],
            "cache_control": {"type": "ephemeral", "ttl": "1h"}}),
            "budget",
        ),
    ];
    for mode in [Mode::Unary, Mode::Streaming] {
        for (params, diagnostic) in &cases {
            let mut request = hello_request();
            request.additional_params = Some(params.clone());
            let error = match wire.encode(request, mode) {
                Ok(_) => panic!("invalid final cache controls must fail: {params}"),
                Err(error) => error,
            };
            assert!(error.to_string().contains(diagnostic), "{mode:?}: {error}");
        }
    }
}

#[test]
fn final_cache_control_accepts_four_ordered_markers_without_scanning_tool_inputs() {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let params = json!({
        "tools": [{"name": "add", "input_schema": {"type": "object"},
            "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        "system": [marked("1h")],
        "messages": [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {
                "cache_control": {"type": "not-a-marker", "ttl": "1d"}
            }}, marked("5m")]}],
        "cache_control": {"type": "ephemeral", "ttl": "5m"}
    });
    for mode in [Mode::Unary, Mode::Streaming] {
        let mut request = hello_request();
        request.additional_params = Some(params.clone());
        let encoded = match wire.encode(request, mode) {
            Ok(encoded) => encoded,
            Err(error) => panic!("ordered final markers must encode: {error}"),
        };
        let body = json_body(&encoded.request);
        for key in ["system", "messages", "cache_control"] {
            assert_eq!(body.get(key), params.get(key), "{mode:?}: {key}");
        }
    }
}

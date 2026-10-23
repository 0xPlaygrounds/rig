use serde_json::json;

use crate::completion::{CompletionRequest, ToolDefinition};
use crate::message::ToolName;
use crate::providers::openai::{OpenAIConfig, wire::DEEPSEEK};
use crate::wire::{Mode, Wire};

/// A request with one tool whose schema uses every bound DeepSeek's strict
/// mode refuses, a `format` it refuses (schemars writes `uint32` for a
/// `u32`) and one it takes, and optional properties.
fn request() -> CompletionRequest {
    CompletionRequest::new("Find the Rust files").tools(vec![ToolDefinition {
        name: ToolName::new("find_files").expect("tool name"),
        description: "Find files".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "extension": {"type": "string", "minLength": 1, "maxLength": 8, "pattern": "^[a-z]+$"},
                "roots": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
                "depth": {"type": "integer", "format": "uint32", "minimum": 0, "maximum": 8},
                "owner": {"type": "string", "format": "email"}
            },
            "required": ["extension"]
        }),
        annotations: Default::default(),
    }])
}

/// Strict function calling is a beta feature: the request goes under `/beta`,
/// every function is `strict`, every object closed with every property
/// required, and the length and item-count bounds and formats DeepSeek
/// refuses are gone while the ones it takes stay. Without strict tools,
/// nothing changes.
#[test]
fn strict_tools_reach_the_beta_endpoint_in_the_schema_deepseek_accepts() {
    let wire = OpenAIConfig::with_key(&DEEPSEEK, "sk-test")
        .chat(super::DEEPSEEK_V4_FLASH)
        .with_strict_tools();
    let encoded = wire.encode(request(), Mode::Unary).expect("encodes");
    assert_eq!(
        encoded.request.uri(),
        "https://api.deepseek.com/beta/chat/completions"
    );
    let body = crate::test_utils::json_body(&encoded.request);
    assert_eq!(
        body.pointer("/tools/0/function"),
        Some(&json!({
            "name": "find_files",
            "description": "Find files",
            "strict": true,
            "parameters": {
                "type": "object",
                "properties": {
                    "extension": {"type": "string", "pattern": "^[a-z]+$"},
                    "roots": {"type": "array", "items": {"type": "string"}},
                    "depth": {"type": "integer", "minimum": 0, "maximum": 8},
                    "owner": {"type": "string", "format": "email"}
                },
                "required": ["extension", "roots", "depth", "owner"],
                "additionalProperties": false
            }
        }))
    );

    // A base URL already at `/beta`, as DeepSeek's guide configures one,
    // gets no second prefix, and one at the `/v1` alias reaches the root's.
    for base in [
        "https://api.deepseek.com/beta/",
        "https://api.deepseek.com/v1",
    ] {
        let wire = OpenAIConfig::with_key(&DEEPSEEK, "sk-test")
            .with_base_url(base)
            .chat(super::DEEPSEEK_V4_FLASH)
            .with_strict_tools();
        let encoded = wire.encode(request(), Mode::Unary).expect("encodes");
        assert_eq!(
            encoded.request.uri(),
            "https://api.deepseek.com/beta/chat/completions",
            "{base}"
        );
    }

    let wire = OpenAIConfig::with_key(&DEEPSEEK, "sk-test").chat(super::DEEPSEEK_V4_FLASH);
    let encoded = wire.encode(request(), Mode::Unary).expect("encodes");
    assert_eq!(
        encoded.request.uri(),
        "https://api.deepseek.com/chat/completions"
    );
    let body = crate::test_utils::json_body(&encoded.request);
    assert_eq!(body.pointer("/tools/0/function/strict"), None);
    assert_eq!(
        body.pointer("/tools/0/function/parameters/properties/extension/minLength"),
        Some(&json!(1))
    );
}

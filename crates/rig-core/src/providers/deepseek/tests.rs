use serde_json::{Value, json};

use crate::completion::{CompletionRequest, ToolDefinition};
use crate::message::ToolName;
use crate::providers::openai::{OpenAIConfig, wire::DEEPSEEK};
use crate::wire::{Mode, Wire};

fn tool(name: &str, parameters: Value) -> ToolDefinition {
    ToolDefinition {
        name: ToolName::new(name).expect("tool name"),
        description: "d".to_owned(),
        parameters,
        annotations: Default::default(),
    }
}

/// The `function` objects a strict DeepSeek wire sends for `tools`, and the
/// URL it sends them to.
fn sent(base: Option<&str>, strict: bool, tools: Vec<ToolDefinition>) -> (String, Vec<Value>) {
    let config = OpenAIConfig::with_key(&DEEPSEEK, "sk-test");
    let config = match base {
        Some(base) => config.with_base_url(base),
        None => config,
    };
    let wire = config.chat(super::DEEPSEEK_V4_FLASH);
    let wire = if strict {
        wire.with_strict_tools()
    } else {
        wire
    };
    let encoded = wire
        .encode(CompletionRequest::new("hi").tools(tools), Mode::Unary)
        .expect("encodes");
    let body = crate::test_utils::json_body(&encoded.request);
    let functions = body
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| tool.get("function").cloned())
        .collect();
    (encoded.request.uri().to_string(), functions)
}

/// Strict function calling is a beta feature at the API root: the default
/// base, one at `/beta` and one at the `/v1` alias all reach
/// `/beta/chat/completions`. Without strict tools nothing changes.
#[test]
fn strict_tools_reach_the_beta_endpoint() {
    let plain = tool(
        "f",
        json!({"type": "object", "properties": {"q": {"type": "string"}}}),
    );
    for base in [
        None,
        Some("https://api.deepseek.com/beta/"),
        Some("https://api.deepseek.com/v1"),
    ] {
        let (uri, _) = sent(base, true, vec![plain.clone()]);
        assert_eq!(
            uri, "https://api.deepseek.com/beta/chat/completions",
            "{base:?}"
        );
    }
    let (uri, functions) = sent(None, false, vec![plain]);
    assert_eq!(uri, "https://api.deepseek.com/chat/completions");
    assert_eq!(functions.first().and_then(|f| f.get("strict")), None);
}

/// The schema shapes schemars writes, rewritten as DeepSeek's strict mode
/// took them live: an `Option` of a struct (a `$ref` in `anyOf`) written
/// out, a one-member `allOf` merged, a string `format` outside its five
/// dropped. Length, item-count and numeric bounds, numeric formats, type
/// lists and `$ref` outside `anyOf` are kept, as it takes them.
#[test]
fn a_strict_schema_takes_the_shape_deepseek_accepts() {
    let (_, functions) = sent(
        None,
        true,
        vec![tool(
            "find",
            json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "when": {"type": "string", "format": "date-time"},
                    "owner_email": {"type": "string", "format": "email"},
                    "limit": {"type": ["integer", "null"], "format": "uint32", "minimum": 0},
                    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "author": {"anyOf": [{"$ref": "#/$defs/Author"}, {"type": "null"}]},
                    "editor": {"allOf": [{"$ref": "#/$defs/Author"}], "description": "who edits"},
                    "first": {"$ref": "#/$defs/Author"}
                },
                "required": ["name"],
                "$defs": {"Author": {"type": "object", "properties": {"name": {"type": "string"}}}}
            }),
        )],
    );
    let author = json!({
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": false
    });
    assert_eq!(
        functions.first(),
        Some(&json!({
            "name": "find",
            "description": "d",
            "strict": true,
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "when": {"type": "string"},
                    "owner_email": {"type": "string", "format": "email"},
                    "limit": {"type": ["integer", "null"], "format": "uint32", "minimum": 0},
                    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "author": {"anyOf": [author, {"type": "null"}]},
                    "editor": {"$ref": "#/$defs/Author", "description": "who edits"},
                    "first": {"$ref": "#/$defs/Author"}
                },
                "required": ["name", "when", "owner_email", "limit", "tags", "author", "editor", "first"],
                "additionalProperties": false,
                "$defs": {"Author": author}
            }
        }))
    );
}

/// A schema strict mode cannot express (a map, a type that holds itself
/// through `anyOf`) sends that tool as written and without `strict`, beside
/// the strict ones: DeepSeek takes the mix.
#[test]
fn a_tool_strict_mode_cannot_express_goes_without_it() {
    let map = json!({"type": "object", "properties": {"labels": {
        "type": "object", "additionalProperties": {"type": "string"}
    }}});
    let tree = json!({
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Tree"}},
        "$defs": {"Tree": {"type": "object", "properties": {
            "next": {"anyOf": [{"$ref": "#/$defs/Tree"}, {"type": "null"}]}
        }}}
    });
    let plain = json!({"type": "object", "properties": {"q": {"type": "string"}}});
    let (_, functions) = sent(
        None,
        true,
        vec![
            tool("map", map.clone()),
            tool("tree", tree.clone()),
            tool("plain", plain),
        ],
    );
    let field = |index: usize, key: &str| functions.get(index).and_then(|f| f.get(key)).cloned();
    assert_eq!(field(0, "strict"), None);
    assert_eq!(field(0, "parameters"), Some(map));
    assert_eq!(field(1, "strict"), None);
    assert_eq!(field(1, "parameters"), Some(tree));
    assert_eq!(field(2, "strict"), Some(json!(true)));
}

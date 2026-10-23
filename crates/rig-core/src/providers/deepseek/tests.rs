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
/// out, one-member `allOf`s merged at any depth, a plain string's `format`
/// outside its five dropped. Length, item-count and numeric bounds, numeric formats, a
/// format on a type list, and `$ref` outside `anyOf` (with a sibling
/// `description`) are kept, as it takes them.
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
                    "since": {"type": ["string", "null"], "format": "date-time"},
                    "owner_email": {"type": "string", "format": "email"},
                    "limit": {"type": ["integer", "null"], "format": "uint32", "minimum": 0},
                    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "author": {"anyOf": [{"$ref": "#/$defs/Author"}, {"type": "null"}]},
                    "editor": {"allOf": [{"$ref": "#/$defs/Author"}], "description": "who edits"},
                    "first": {"$ref": "#/$defs/Author"},
                    "reviewer": {"$ref": "#/$defs/Author", "description": "who reviews"},
                    "nested": {"allOf": [{"allOf": [{"type": "integer", "minimum": 10}]}]}
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
                    "since": {"type": ["string", "null"], "format": "date-time"},
                    "owner_email": {"type": "string", "format": "email"},
                    "limit": {"type": ["integer", "null"], "format": "uint32", "minimum": 0},
                    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "author": {"anyOf": [author, {"type": "null"}]},
                    "editor": {"$ref": "#/$defs/Author", "description": "who edits"},
                    "first": {"$ref": "#/$defs/Author"},
                    "reviewer": {"$ref": "#/$defs/Author", "description": "who reviews"},
                    "nested": {"type": "integer", "minimum": 10}
                },
                "required": [
                    "name", "when", "since", "owner_email", "limit", "tags", "author", "editor",
                    "first", "reviewer", "nested"
                ],
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

/// An `anyOf` member is written out to the end: through a one-member
/// `allOf` around a reference, and through a definition that is itself a
/// reference.
#[test]
fn an_any_of_member_is_written_out_through_all_of_and_references() {
    let (_, functions) = sent(
        None,
        true,
        vec![tool(
            "f",
            json!({
                "type": "object",
                "properties": {
                    "wrapped": {"anyOf": [{"allOf": [{"$ref": "#/$defs/Author"}]}, {"type": "null"}]},
                    "aliased": {"anyOf": [{"$ref": "#/$defs/Writer"}, {"type": "null"}]}
                },
                "$defs": {
                    "Author": {"type": "object", "properties": {"name": {"type": "string"}}},
                    "Writer": {"$ref": "#/$defs/Author"}
                }
            }),
        )],
    );
    let author = json!({
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": false
    });
    let function = functions.first().expect("one tool");
    assert_eq!(function.get("strict"), Some(&json!(true)));
    for property in ["wrapped", "aliased"] {
        assert_eq!(
            function.pointer(&format!("/parameters/properties/{property}/anyOf/0")),
            Some(&author),
            "{property}"
        );
    }
}

/// A one-member `allOf` that restates a bound differently, and a tuple,
/// cannot be strict without loosening the schema: the tool goes without it.
#[test]
fn a_schema_a_merge_would_loosen_goes_without_strict() {
    let restated = json!({"type": "object", "properties": {"n": {
        "type": "integer", "minimum": 0, "allOf": [{"minimum": 10}]
    }}});
    let tuple = json!({"type": "object", "properties": {"pair": {
        "type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}]
    }}});
    let (_, functions) = sent(
        None,
        true,
        vec![tool("restated", restated), tool("tuple", tuple)],
    );
    for function in &functions {
        assert_eq!(function.get("strict"), None, "{function}");
    }
}

/// A type that holds itself through `anyOf` at the root (schemars writes
/// `"$ref": "#"`) goes without strict mode.
#[test]
fn a_root_recursive_type_goes_without_strict() {
    let list = json!({"type": "object", "properties": {
        "next": {"anyOf": [{"$ref": "#"}, {"type": "null"}]}
    }});
    let (_, functions) = sent(None, true, vec![tool("list", list)]);
    assert_eq!(functions.first().and_then(|f| f.get("strict")), None);
}

/// Draft 7's `definitions` and the `$def` DeepSeek's guide spells are
/// closed and written out like `$defs`; a draft 7 tuple goes without strict
/// mode.
#[test]
fn every_definitions_keyword_is_written_out_and_closed() {
    let entry = json!({"type": "object", "properties": {
        "when": {"type": "string", "format": "date-time"}
    }});
    let closed = json!({
        "type": "object",
        "properties": {"when": {"type": "string"}},
        "required": ["when"],
        "additionalProperties": false
    });
    for (key, prefix) in [("definitions", "#/definitions/"), ("$def", "#/$def/")] {
        let reference = format!("{prefix}Entry");
        let schema = json!({
            "type": "object",
            "properties": {
                "optional": {"anyOf": [{"$ref": reference}, {"type": "null"}]},
                "direct": {"$ref": reference}
            },
            key: {"Entry": entry.clone()}
        });
        let (_, functions) = sent(None, true, vec![tool("f", schema)]);
        let function = functions.first().expect("one tool");
        assert_eq!(function.get("strict"), Some(&json!(true)), "{key}");
        assert_eq!(
            function.pointer("/parameters/properties/optional/anyOf/0"),
            Some(&closed),
            "{key}"
        );
        assert_eq!(
            function
                .get("parameters")
                .and_then(|p| p.get(key))
                .and_then(|d| d.get("Entry")),
            Some(&closed),
            "{key}"
        );
    }
    let tuple = json!({"type": "object", "properties": {"pair": {
        "type": "array", "items": [{"type": "string"}, {"type": "integer"}]
    }}});
    let (_, functions) = sent(None, true, vec![tool("tuple", tuple)]);
    assert_eq!(functions.first().and_then(|f| f.get("strict")), None);
}

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
    let parameters = json!({
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
    });
    let required: Vec<_> = parameters
        .get("properties")
        .and_then(Value::as_object)
        .expect("object properties")
        .keys()
        .cloned()
        .collect();
    let (_, functions) = sent(None, true, vec![tool("find", parameters)]);
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
                "required": required,
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
    let map = json!({"type": "object", "properties": {
        "labels": {"type": "object", "additionalProperties": {"type": "string"}},
        "matched": {"type": "object", "patternProperties": {"^x": {"type": "string"}}}
    }});
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

/// Definitions nested in other definitions are closed too, their `oneOf`
/// read as `anyOf`.
#[test]
fn nested_definitions_are_closed() {
    let schema = json!({
        "type": "object",
        "properties": {"inner": {"$ref": "#/definitions/Outer/definitions/Inner"}},
        "definitions": {"Outer": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "definitions": {"Inner": {
                "type": "object",
                "properties": {"v": {"oneOf": [{"type": "string"}, {"type": "integer"}]}}
            }}
        }}
    });
    let (_, functions) = sent(None, true, vec![tool("f", schema)]);
    let function = functions.first().expect("one tool");
    assert_eq!(function.get("strict"), Some(&json!(true)));
    assert_eq!(
        function.pointer("/parameters/definitions/Outer/definitions/Inner"),
        Some(&json!({
            "type": "object",
            "properties": {"v": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
            "required": ["v"],
            "additionalProperties": false
        }))
    );
    assert_eq!(
        function.pointer("/parameters/definitions/Outer/additionalProperties"),
        Some(&json!(false))
    );
}

/// A type list naming `object` or `array` (schemars' `Option<Vec<T>>`),
/// which strict mode refused live while it took the same as an `anyOf`, is
/// split into one: each member keeps its own type's keywords, the parent
/// its description.
#[test]
fn a_type_list_naming_an_array_or_object_becomes_an_any_of() {
    let (_, functions) = sent(
        None,
        true,
        vec![tool(
            "f",
            json!({"type": "object", "properties": {
                "tags": {"type": ["array", "null"], "items": {"type": "string"}, "description": "labels"},
                "meta": {"type": ["object", "null"], "properties": {"k": {"type": "string"}}}
            }}),
        )],
    );
    let function = functions.first().expect("one tool");
    assert_eq!(function.get("strict"), Some(&json!(true)));
    assert_eq!(
        function.pointer("/parameters/properties/tags"),
        Some(&json!({
            "description": "labels",
            "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]
        }))
    );
    assert_eq!(
        function.pointer("/parameters/properties/meta"),
        Some(&json!({"anyOf": [
            {
                "type": "object",
                "properties": {"k": {"type": "string"}},
                "required": ["k"],
                "additionalProperties": false
            },
            {"type": "null"}
        ]}))
    );
}

/// An object with no properties, which strict mode took live only as the
/// whole tool's parameters, sends a tool that holds one deeper without
/// strict mode.
#[test]
fn an_empty_object_below_the_top_goes_without_strict() {
    let empty = json!({"type": "object", "properties": {}});
    let (_, functions) = sent(None, true, vec![tool("none", empty)]);
    assert_eq!(
        functions.first().and_then(|f| f.get("strict")),
        Some(&json!(true))
    );
    for nested in [
        json!({"type": "object", "properties": {"meta": {"type": ["object", "null"]}}}),
        json!({"type": "object", "properties": {"meta": {"allOf": [{"type": "object"}]}}}),
        json!({"type": "object", "properties": {"meta": {"allOf": [{"allOf": [{"type": "object"}]}]}}}),
        json!({"type": "object", "properties": {"list": {
            "type": "array", "items": {"allOf": [{"type": "object"}]}
        }}}),
    ] {
        let (_, functions) = sent(None, true, vec![tool("nested", nested.clone())]);
        let function = functions.first().expect("one tool");
        assert_eq!(function.get("strict"), None, "{nested}");
        assert_eq!(function.get("parameters"), Some(&nested));
    }
}

/// References inside `anyOf` resolve as JSON Pointers: through nested
/// definitions, and with `~1` naming a `/` in a definition's name.
#[test]
fn an_any_of_reference_resolves_as_a_json_pointer() {
    let (_, functions) = sent(
        None,
        true,
        vec![tool(
            "f",
            json!({
                "type": "object",
                "properties": {
                    "nested": {"anyOf": [{"$ref": "#/definitions/Outer/definitions/Inner"}, {"type": "null"}]},
                    "escaped": {"anyOf": [{"$ref": "#/$defs/A~1B"}, {"type": "null"}]}
                },
                "definitions": {"Outer": {"type": "object", "properties": {"x": {"type": "string"}}, "definitions": {
                    "Inner": {"type": "string", "minLength": 2}
                }}},
                "$defs": {"A/B": {"type": "integer"}, "A~1B": {"type": "boolean"}}
            }),
        )],
    );
    let function = functions.first().expect("one tool");
    assert_eq!(function.get("strict"), Some(&json!(true)));
    assert_eq!(
        function.pointer("/parameters/properties/nested/anyOf/0"),
        Some(&json!({"type": "string", "minLength": 2}))
    );
    assert_eq!(
        function.pointer("/parameters/properties/escaped/anyOf/0"),
        Some(&json!({"type": "integer"}))
    );
}

/// Live, strict mode took a `$ref` deeper under an `anyOf` member, as
/// schemars' `Option<Vec<Struct>>` gives one after the split, and refused an
/// untyped schema (`true`, `{}`), which sends the tool without strict mode.
#[test]
fn a_deeper_reference_stays_and_an_untyped_schema_goes_without_strict() {
    let (_, functions) = sent(
        None,
        true,
        vec![tool(
            "f",
            json!({
                "type": "object",
                "properties": {"authors": {"type": ["array", "null"], "items": {"$ref": "#/$defs/Author"}}},
                "$defs": {"Author": {"type": "object", "properties": {"name": {"type": "string"}}}}
            }),
        )],
    );
    let function = functions.first().expect("one tool");
    assert_eq!(function.get("strict"), Some(&json!(true)));
    assert_eq!(
        function.pointer("/parameters/properties/authors/anyOf/0"),
        Some(&json!({"type": "array", "items": {"$ref": "#/$defs/Author"}}))
    );
    for untyped in [
        json!(true),
        json!({}),
        json!({"description": "anything"}),
        json!({"enum": ["a", "b"]}),
    ] {
        let schema = json!({"type": "object", "properties": {"value": untyped}});
        let (_, functions) = sent(None, true, vec![tool("value", schema.clone())]);
        assert_eq!(
            functions.first().and_then(|f| f.get("strict")),
            None,
            "{schema}"
        );
    }
}

/// A `oneOf` beside an `anyOf`, which joining would loosen, sends the tool
/// as written without strict mode, in a property and in a definition.
#[test]
fn a_one_of_beside_an_any_of_goes_without_strict() {
    let both = json!({
        "anyOf": [{"type": "string"}, {"type": "integer"}],
        "oneOf": [{"type": "integer"}, {"type": "boolean"}]
    });
    for schema in [
        json!({"type": "object", "properties": {"v": both.clone()}}),
        json!({
            "type": "object",
            "properties": {"v": {"$ref": "#/$defs/V"}},
            "$defs": {"V": both.clone()}
        }),
    ] {
        let (_, functions) = sent(None, true, vec![tool("f", schema.clone())]);
        let function = functions.first().expect("one tool");
        assert_eq!(function.get("strict"), None, "{schema}");
        assert_eq!(function.get("parameters"), Some(&schema));
    }
}

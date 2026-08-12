//! Live-recorded coverage for the strict-schema transformation matrix.

use rig::completion::{CompletionModel, ToolDefinition};
use rig::message::{AssistantContent, ToolChoice};
use rig::prelude::*;
use rig::providers::anthropic;
use serde_json::{Value, json};

use super::super::support::with_anthropic_cassette;
use super::messages_strict_tools::{assert_strict_tool_call, strict_tool_call_arguments};

async fn assert_strict_schema_rejected(
    client: anthropic::Client,
    tool_name: &str,
    prompt: &str,
    parameters: Value,
) {
    let model = client
        .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
        .with_strict_tools();
    let request = model
        .completion_request(prompt)
        .max_tokens(64)
        .tool_choice(ToolChoice::Required)
        .tool(ToolDefinition {
            name: tool_name.to_string(),
            description: "Exercise a schema the strict compiler rejects.".to_string(),
            parameters,
        })
        .build();
    let error = model
        .completion(request)
        .await
        .expect_err("Anthropic's strict compiler should reject this schema");
    assert_eq!(
        error
            .provider_response_status()
            .expect("provider status should be preserved")
            .as_u16(),
        400
    );
    let body = error
        .provider_response_json()
        .expect("provider error should be JSON")
        .expect("provider error body should be preserved");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn empty_object_schema_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_matrix/empty_object_schema_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "ping_empty",
                "Call ping_empty with an empty object.",
                json!({ "type": "object", "properties": {} }),
                json!({}),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn primitive_types_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_matrix/primitive_types_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_primitives",
                "Record text = alpha, count = 7, ratio = 1.5, enabled = true, and nothing = null exactly.",
                json!({
                    "type": "object",
                    "properties": {
                        "text": { "type": "string" },
                        "count": { "type": "integer" },
                        "ratio": { "type": "number" },
                        "enabled": { "type": "boolean" },
                        "nothing": { "type": "null" }
                    },
                    "required": ["text", "count", "ratio", "enabled", "nothing"]
                }),
                json!({
                    "text": "alpha",
                    "count": 7,
                    "ratio": 1.5,
                    "enabled": true,
                    "nothing": null
                }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn scalar_enum_and_const_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_matrix/scalar_enum_and_const_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_constants",
                "Record mode = beta, version = 3, enabled = true, and marker = null exactly.",
                json!({
                    "type": "object",
                    "properties": {
                        "mode": { "type": "string", "enum": ["alpha", "beta"] },
                        "version": { "type": "integer", "const": 3 },
                        "enabled": { "type": "boolean", "const": true },
                        "marker": { "type": "null", "const": null }
                    },
                    "required": ["mode", "version", "enabled", "marker"]
                }),
                json!({ "mode": "beta", "version": 3, "enabled": true, "marker": null }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn explicit_any_of_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_matrix/explicit_any_of_roundtrip",
        |client| async move {
            let arguments = strict_tool_call_arguments(
                client,
                "record_identifier",
                "Record identifier = 42 as an integer.",
                json!({
                    "type": "object",
                    "properties": {
                        "identifier": {
                            "anyOf": [
                                { "type": "integer" },
                                { "type": "string" }
                            ]
                        }
                    },
                    "required": ["identifier"]
                }),
            )
            .await;
            assert!(
                arguments["identifier"] == json!(42) || arguments["identifier"] == json!("42"),
                "either valid anyOf branch is acceptable, got {arguments:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn root_all_of_object_schema_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_matrix/root_all_of_object_schema_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_root_all_of",
                "Record code = alpha and tenant = acme exactly.",
                json!({
                    "allOf": [
                        {
                            "type": "object",
                            "properties": { "code": { "type": "string" } },
                            "required": ["code"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "code": { "enum": ["alpha", "beta"] },
                                "tenant": { "type": "string" }
                            },
                            "required": ["tenant"]
                        }
                    ]
                }),
                json!({ "code": "alpha", "tenant": "acme" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn root_all_of_local_defs_ref_branch_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_matrix/root_all_of_local_defs_ref_branch_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_root_all_of_ref",
                "Record code = alpha and tenant = acme exactly.",
                json!({
                    "$defs": {
                        "Base": {
                            "type": "object",
                            "properties": { "code": { "type": "string" } },
                            "required": ["code"]
                        }
                    },
                    "allOf": [
                        { "$ref": "#/$defs/Base" },
                        {
                            "type": "object",
                            "properties": { "tenant": { "type": "string" } },
                            "required": ["tenant"]
                        }
                    ]
                }),
                json!({ "code": "alpha", "tenant": "acme" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn root_all_of_local_draft7_ref_branch_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_matrix/root_all_of_local_draft7_ref_branch_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_root_all_of_legacy_ref",
                "Record code = legacy and tenant = acme exactly.",
                json!({
                    "definitions": {
                        "Base": {
                            "type": "object",
                            "properties": { "code": { "type": "string" } },
                            "required": ["code"]
                        }
                    },
                    "allOf": [
                        { "$ref": "#/definitions/Base" },
                        {
                            "type": "object",
                            "properties": { "tenant": { "type": "string" } },
                            "required": ["tenant"]
                        }
                    ]
                }),
                json!({ "code": "legacy", "tenant": "acme" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn top_level_any_of_is_rejected_for_tool_input() {
    with_anthropic_cassette(
        "strict_schema_matrix/top_level_any_of_is_rejected_for_tool_input",
        |client| async move {
            assert_strict_schema_rejected(
                client,
                "record_root_any_of",
                "Call record_root_any_of with value = alpha.",
                json!({
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": { "value": { "type": "string" } },
                            "required": ["value"]
                        },
                        {
                            "type": "object",
                            "properties": { "count": { "type": "integer" } },
                            "required": ["count"]
                        }
                    ]
                }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn top_level_one_of_is_rejected_for_tool_input() {
    with_anthropic_cassette(
        "strict_schema_matrix/top_level_one_of_is_rejected_for_tool_input",
        |client| async move {
            assert_strict_schema_rejected(
                client,
                "record_root_one_of",
                "Call record_root_one_of with value = alpha.",
                json!({
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": { "value": { "type": "string" } },
                            "required": ["value"]
                        },
                        {
                            "type": "object",
                            "properties": { "count": { "type": "integer" } },
                            "required": ["count"]
                        }
                    ]
                }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn scalar_root_schema_is_rejected_for_tool_input() {
    with_anthropic_cassette(
        "strict_schema_matrix/scalar_root_schema_is_rejected_for_tool_input",
        |client| async move {
            assert_strict_schema_rejected(
                client,
                "record_scalar_root",
                "Call record_scalar_root with alpha.",
                json!({ "type": "string" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn boolean_root_schema_is_rejected_for_tool_input() {
    with_anthropic_cassette(
        "strict_schema_matrix/boolean_root_schema_is_rejected_for_tool_input",
        |client| async move {
            assert_strict_schema_rejected(
                client,
                "record_boolean_root",
                "Call record_boolean_root.",
                json!(true),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn root_defs_ref_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_matrix/root_defs_ref_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_root_ref",
                "Record name = Ada and omit nickname.",
                json!({
                    "$ref": "#/$defs/Person",
                    "$defs": {
                        "Person": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "nickname": { "type": "string" }
                            },
                            "required": ["name"]
                        }
                    }
                }),
                json!({ "name": "Ada" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn root_draft7_definitions_ref_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_matrix/root_draft7_definitions_ref_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_legacy_root_ref",
                "Record code = ZX and omit note.",
                json!({
                    "$ref": "#/definitions/Record",
                    "definitions": {
                        "Record": {
                            "type": "object",
                            "properties": {
                                "code": { "type": "string" },
                                "note": { "type": "string" }
                            },
                            "required": ["code"]
                        }
                    }
                }),
                json!({ "code": "ZX" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn root_ref_retains_sibling_definitions() {
    with_anthropic_cassette(
        "strict_schema_matrix/root_ref_retains_sibling_definitions",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_order_ref",
                "Record order_id = 7 and shipping.city = Oslo; omit shipping.line2.",
                json!({
                    "$ref": "#/$defs/Order",
                    "$defs": {
                        "Order": {
                            "type": "object",
                            "properties": {
                                "order_id": { "type": "integer" },
                                "shipping": { "$ref": "#/$defs/Address" }
                            },
                            "required": ["order_id", "shipping"]
                        },
                        "Address": {
                            "type": "object",
                            "properties": {
                                "city": { "type": "string" },
                                "line2": { "type": "string" }
                            },
                            "required": ["city"]
                        }
                    }
                }),
                json!({ "order_id": 7, "shipping": { "city": "Oslo" } }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn root_ref_preserves_sibling_properties_and_required() {
    with_anthropic_cassette(
        "strict_schema_matrix/root_ref_preserves_sibling_properties_and_required",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_tenant_code",
                "Record code = base and tenant = acme exactly.",
                json!({
                    "$ref": "#/$defs/Base",
                    "$defs": {
                        "Base": {
                            "type": "object",
                            "properties": { "code": { "type": "string" } },
                            "required": ["code"]
                        }
                    },
                    "title": "Tenant-qualified code",
                    "description": "Both the referenced and sibling constraints apply.",
                    "properties": {
                        "code": { "const": "base" },
                        "tenant": { "type": "string" }
                    },
                    "required": ["tenant"]
                }),
                json!({ "code": "base", "tenant": "acme" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn root_ref_preserves_sibling_all_of() {
    with_anthropic_cassette(
        "strict_schema_matrix/root_ref_preserves_sibling_all_of",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_all_of_root_ref",
                "Record code = alpha exactly.",
                json!({
                    "$ref": "#/$defs/Base",
                    "$defs": {
                        "Base": {
                            "type": "object",
                            "properties": { "code": { "type": "string" } },
                            "required": ["code"]
                        }
                    },
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {
                                "code": { "enum": ["alpha", "beta"] }
                            },
                            "required": ["code"]
                        }
                    ]
                }),
                json!({ "code": "alpha" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn root_ref_sibling_unions_become_guidance() {
    with_anthropic_cassette(
        "strict_schema_matrix/root_ref_sibling_unions_become_guidance",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_root_ref_unions",
                "Record value as the exact JSON string alpha.",
                json!({
                    "$ref": "#/$defs/Base",
                    "$defs": {
                        "Base": {
                            "type": "object",
                            "properties": { "value": { "type": "string" } },
                            "required": ["value"]
                        }
                    },
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": { "value": { "const": "alpha" } }
                        },
                        {
                            "type": "object",
                            "properties": { "value": { "const": "beta" } }
                        }
                    ],
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": { "value": { "const": "alpha" } }
                        },
                        {
                            "type": "object",
                            "properties": { "value": { "const": "gamma" } }
                        }
                    ]
                }),
                json!({ "value": "alpha" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn root_ref_keeps_document_defs_authoritative_on_collision() {
    with_anthropic_cassette(
        "strict_schema_matrix/root_ref_keeps_document_defs_authoritative_on_collision",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_defs_collision",
                "Record value as the exact JSON string root-string.",
                json!({
                    "$ref": "#/$defs/Root",
                    "$defs": {
                        "Shared": { "type": "string" },
                        "Root": {
                            "type": "object",
                            "$defs": { "Shared": { "type": "integer" } },
                            "properties": {
                                "value": { "$ref": "#/$defs/Shared" }
                            },
                            "required": ["value"]
                        }
                    }
                }),
                json!({ "value": "root-string" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn root_ref_keeps_document_definitions_authoritative_on_collision() {
    with_anthropic_cassette(
        "strict_schema_matrix/root_ref_keeps_document_definitions_authoritative_on_collision",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_definitions_collision",
                "Record value as the exact JSON string legacy-root-string.",
                json!({
                    "$ref": "#/definitions/Root",
                    "definitions": {
                        "Shared": { "type": "string" },
                        "Root": {
                            "type": "object",
                            "definitions": { "Shared": { "type": "integer" } },
                            "properties": {
                                "value": { "$ref": "#/definitions/Shared" }
                            },
                            "required": ["value"]
                        }
                    }
                }),
                json!({ "value": "legacy-root-string" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn chained_root_refs_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_matrix/chained_root_refs_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_chained_ref",
                "Record value as the exact JSON string \"resolved\".",
                json!({
                    "$ref": "#/$defs/Alias",
                    "$defs": {
                        "Alias": { "$ref": "#/$defs/Payload" },
                        "Payload": {
                            "type": "object",
                            "properties": { "value": { "type": "string" } },
                            "required": ["value"]
                        }
                    }
                }),
                json!({ "value": "resolved" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn escaped_json_pointer_root_ref_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_matrix/escaped_json_pointer_root_ref_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_escaped_ref",
                "Record value as the exact JSON string \"escaped\".",
                json!({
                    "$ref": "#/$defs/Person~1Profile",
                    "$defs": {
                        "Person/Profile": {
                            "type": "object",
                            "properties": { "value": { "type": "string" } },
                            "required": ["value"]
                        }
                    }
                }),
                json!({ "value": "escaped" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn dangling_local_root_ref_is_rejected() {
    with_anthropic_cassette(
        "strict_schema_matrix/dangling_local_root_ref_is_rejected",
        |client| async move {
            assert_strict_schema_rejected(
                client,
                "record_dangling_ref",
                "Call record_dangling_ref.",
                json!({
                    "$ref": "#/$defs/Missing",
                    "$defs": {
                        "Present": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn external_root_ref_is_rejected() {
    with_anthropic_cassette(
        "strict_schema_matrix/external_root_ref_is_rejected",
        |client| async move {
            assert_strict_schema_rejected(
                client,
                "record_external_ref",
                "Call record_external_ref.",
                json!({ "$ref": "https://example.com/schemas/input.json" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn scalar_root_ref_is_rejected_for_tool_input() {
    with_anthropic_cassette(
        "strict_schema_matrix/scalar_root_ref_is_rejected_for_tool_input",
        |client| async move {
            assert_strict_schema_rejected(
                client,
                "record_scalar_ref",
                "Call record_scalar_ref.",
                json!({
                    "$ref": "#/$defs/Scalar",
                    "$defs": { "Scalar": { "type": "string" } }
                }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn cyclic_root_ref_aliases_are_rejected() {
    with_anthropic_cassette(
        "strict_schema_matrix/cyclic_root_ref_aliases_are_rejected",
        |client| async move {
            assert_strict_schema_rejected(
                client,
                "record_cyclic_alias",
                "Call record_cyclic_alias.",
                json!({
                    "$ref": "#/$defs/A",
                    "$defs": {
                        "A": { "$ref": "#/$defs/B" },
                        "B": { "$ref": "#/$defs/A" }
                    }
                }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn recursive_ref_is_rejected_by_strict_compiler() {
    with_anthropic_cassette(
        "strict_schema_matrix/recursive_ref_is_rejected_by_strict_compiler",
        |client| async move {
            assert_strict_schema_rejected(
                client,
                "record_linked_node",
                "Record a linked node with value = first and next = null.",
                json!({
                    "$ref": "#/$defs/Node",
                    "$defs": {
                        "Node": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "next": {
                                    "anyOf": [
                                        { "$ref": "#/$defs/Node" },
                                        { "type": "null" }
                                    ]
                                }
                            },
                            "required": ["value", "next"]
                        }
                    }
                }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn supported_array_minimums_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_matrix/supported_array_minimums_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_array_minimums",
                "Record empty = [] and one = [\"only\"] exactly.",
                json!({
                    "type": "object",
                    "properties": {
                        "empty": {
                            "type": "array",
                            "items": { "type": "string" },
                            "minItems": 0
                        },
                        "one": {
                            "type": "array",
                            "items": { "type": "string" },
                            "minItems": 1
                        }
                    },
                    "required": ["empty", "one"]
                }),
                json!({ "empty": [], "one": ["only"] }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn nested_arrays_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_matrix/nested_arrays_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_matrix",
                "Record matrix = [[1, 2], [3, 4]] exactly.",
                json!({
                    "type": "object",
                    "properties": {
                        "matrix": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": { "type": "integer" }
                            }
                        }
                    },
                    "required": ["matrix"]
                }),
                json!({ "matrix": [[1, 2], [3, 4]] }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn every_numeric_constraint_becomes_guidance() {
    with_anthropic_cassette(
        "strict_schema_matrix/every_numeric_constraint_becomes_guidance",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_constrained_number",
                "Record value = 6 exactly.",
                json!({
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "integer",
                            "description": "A constrained test value.",
                            "minimum": 2,
                            "maximum": 10,
                            "exclusiveMinimum": 1,
                            "exclusiveMaximum": 11,
                            "multipleOf": 2
                        }
                    },
                    "required": ["value"]
                }),
                json!({ "value": 6 }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn unsupported_array_constraints_become_guidance() {
    with_anthropic_cassette(
        "strict_schema_matrix/unsupported_array_constraints_become_guidance",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_unique_values",
                "Record values = [1, 2] exactly.",
                json!({
                    "type": "object",
                    "properties": {
                        "values": {
                            "type": "array",
                            "items": { "type": "integer" },
                            "minItems": 2,
                            "maxItems": 4,
                            "uniqueItems": true,
                            "contains": { "const": 2 },
                            "minContains": 1,
                            "maxContains": 1
                        }
                    },
                    "required": ["values"]
                }),
                json!({ "values": [1, 2] }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn unsupported_object_constraints_become_guidance() {
    with_anthropic_cassette(
        "strict_schema_matrix/unsupported_object_constraints_become_guidance",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_labeled_item",
                "Record item.label = ok exactly.",
                json!({
                    "type": "object",
                    "properties": {
                        "item": {
                            "type": "object",
                            "properties": { "label": { "type": "string" } },
                            "required": ["label"],
                            "minProperties": 1,
                            "maxProperties": 1,
                            "propertyNames": { "pattern": "^[a-z]+$" },
                            "patternProperties": { "^x-": { "type": "string" } }
                        }
                    },
                    "required": ["item"]
                }),
                json!({ "item": { "label": "ok" } }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn conditionals_and_dependencies_become_guidance() {
    with_anthropic_cassette(
        "strict_schema_matrix/conditionals_and_dependencies_become_guidance",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_conditional",
                "Record kind = basic and detail = ready exactly.",
                json!({
                    "type": "object",
                    "properties": {
                        "kind": { "type": "string" },
                        "detail": { "type": "string" }
                    },
                    "required": ["kind", "detail"],
                    "if": { "properties": { "kind": { "const": "basic" } } },
                    "then": { "required": ["detail"] },
                    "else": { "not": { "required": ["detail"] } },
                    "dependentRequired": { "kind": ["detail"] },
                    "dependentSchemas": { "kind": { "required": ["detail"] } }
                }),
                json!({ "kind": "basic", "detail": "ready" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn annotations_become_guidance_without_overwriting_description() {
    with_anthropic_cassette(
        "strict_schema_matrix/annotations_become_guidance_without_overwriting_description",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_annotated",
                "Record value = stable exactly.",
                json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "$id": "urn:rig:test:annotated",
                    "type": "object",
                    "title": "Annotated input",
                    "description": "Preserve this description.",
                    "properties": {
                        "value": {
                            "type": "string",
                            "default": "fallback",
                            "examples": ["stable"],
                            "deprecated": false,
                            "readOnly": false,
                            "writeOnly": false
                        }
                    },
                    "required": ["value"]
                }),
                json!({ "value": "stable" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn properties_without_explicit_object_type_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_matrix/properties_without_explicit_object_type_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_inferred_object",
                "Record value as the exact JSON string \"inferred\".",
                json!({
                    "properties": { "value": { "type": "string" } },
                    "required": ["value"]
                }),
                json!({ "value": "inferred" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn boolean_true_subschema_is_rejected_by_strict_compiler() {
    with_anthropic_cassette(
        "strict_schema_matrix/boolean_true_subschema_is_rejected_by_strict_compiler",
        |client| async move {
            assert_strict_schema_rejected(
                client,
                "record_unconstrained_value",
                "Call record_unconstrained_value with value = free.",
                json!({
                        "type": "object",
                        "properties": { "value": true },
                        "required": ["value"]
                }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn required_and_optional_property_order_schema_is_accepted() {
    with_anthropic_cassette(
        "strict_schema_matrix/required_and_optional_property_order_schema_is_accepted",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let request = model
                .completion_request(
                    "Call record_order with required_first = yes and optional_last = included.",
                )
                .preamble("Copy both values exactly into one tool call.".to_string())
                .max_tokens(1024)
                .tool_choice(ToolChoice::Required)
                .tool(ToolDefinition {
                    name: "record_order".to_string(),
                    description: "Record required and optional properties.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "optional_last": { "type": "string" },
                            "required_first": { "type": "string" }
                        },
                        "required": ["required_first"]
                    }),
                })
                .build();

            let response = model
                .completion(request)
                .await
                .expect("strict property-order request should succeed");
            let arguments = response
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(&tool_call.function.arguments),
                    _ => None,
                })
                .expect("response should contain a tool call");
            assert_eq!(arguments["required_first"], "yes");
            if let Some(optional) = arguments.get("optional_last") {
                assert_eq!(optional, "included");
            }
        },
    )
    .await;
}

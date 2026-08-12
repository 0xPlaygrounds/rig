//! Anthropic Messages API strict-tool regression tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::completion::{CompletionModel, ToolDefinition};
use rig::message::{AssistantContent, ToolChoice};
use rig::prelude::*;
use rig::providers::anthropic;
use serde_json::json;

use super::super::support::with_anthropic_cassette;

pub(super) async fn assert_strict_tool_call(
    client: anthropic::Client,
    tool_name: &str,
    prompt: &str,
    parameters: serde_json::Value,
    expected_arguments: serde_json::Value,
) {
    let arguments = strict_tool_call_arguments(client, tool_name, prompt, parameters).await;
    assert_eq!(arguments, expected_arguments);
}

pub(super) async fn strict_tool_call_arguments(
    client: anthropic::Client,
    tool_name: &str,
    prompt: &str,
    parameters: serde_json::Value,
) -> serde_json::Value {
    let model = client
        .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
        .with_strict_tools();
    let request = model
        .completion_request(prompt)
        .preamble(
            "Call the supplied tool exactly once and follow the requested argument shape."
                .to_string(),
        )
        .max_tokens(1024)
        .tool_choice(ToolChoice::Required)
        .tool(ToolDefinition {
            name: tool_name.to_string(),
            description: "Record the exact structured arguments requested by the user.".to_string(),
            parameters,
        })
        .build();

    let response = model
        .completion(request)
        .await
        .expect("strict-tools completion should succeed");
    let tool_calls = response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(
        tool_calls.len(),
        1,
        "exactly one strict tool call is expected"
    );
    assert_eq!(tool_calls[0].function.name, tool_name);
    tool_calls[0].function.arguments.clone()
}

#[tokio::test]
async fn strict_tools_opt_in_roundtrip() {
    with_anthropic_cassette(
        "messages_strict_tools/strict_tools_opt_in_roundtrip",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let request = model
                .completion_request(
                    "Call record_booking exactly once with passengers = 2 and cabin = economy.",
                )
                .preamble("Follow the tool-calling instruction exactly.".to_string())
                .max_tokens(1024)
                .tool_choice(ToolChoice::Required)
                .tool(ToolDefinition {
                    name: "record_booking".to_string(),
                    description: "Record a passenger count and cabin class.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "passengers": { "type": "integer" },
                            "cabin": {
                                "type": "string",
                                "enum": ["economy", "business"]
                            }
                        },
                        "required": ["passengers", "cabin"],
                        "additionalProperties": false
                    }),
                })
                .build();

            let response = model
                .completion(request)
                .await
                .expect("strict-tools completion should succeed");

            let tool_call = response
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call),
                    _ => None,
                })
                .expect("strict tool call should be produced");
            assert_eq!(tool_call.function.name, "record_booking");
            assert_eq!(tool_call.function.arguments["passengers"], json!(2));
            assert_eq!(tool_call.function.arguments["cabin"], json!("economy"));
        },
    )
    .await;
}

#[tokio::test]
async fn optional_scalar_can_be_omitted() {
    with_anthropic_cassette(
        "messages_strict_tools/optional_scalar_can_be_omitted",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_weather_query",
                "Record city = seattle. Omit unit; do not invent any optional field.",
                json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["city"]
                }),
                json!({ "city": "seattle" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn optional_scalar_can_be_included_after_constraint_transform() {
    with_anthropic_cassette(
        "messages_strict_tools/optional_scalar_can_be_included_after_constraint_transform",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_profile",
                "Record name = river and optional nickname = ace. Include both fields exactly.",
                json!({
                    "type": "object",
                    "additionalProperties": true,
                    "properties": {
                        "name": { "type": "string" },
                        "nickname": {
                            "type": "string",
                            "minLength": 3,
                            "maxLength": 12,
                            "pattern": "^[a-z]+$",
                            "format": "slug"
                        }
                    },
                    "required": ["name"]
                }),
                json!({ "name": "river", "nickname": "ace" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn nested_optional_object_can_be_omitted() {
    with_anthropic_cassette(
        "messages_strict_tools/nested_optional_object_can_be_omitted",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_account",
                "Record account_id = acct_demo. Omit the optional contact object entirely.",
                json!({
                    "type": "object",
                    "properties": {
                        "account_id": { "type": "string" },
                        "contact": {
                            "type": "object",
                            "properties": {
                                "email": { "type": "string", "format": "email" },
                                "phone": { "type": "string" }
                            },
                            "required": ["email"]
                        }
                    },
                    "required": ["account_id"]
                }),
                json!({ "account_id": "acct_demo" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn nested_optional_object_can_be_partially_populated() {
    with_anthropic_cassette(
        "messages_strict_tools/nested_optional_object_can_be_partially_populated",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_customer",
                "Record customer name = mira and address city = oslo. Include address, but omit its optional postal_code.",
                json!({
                    "type": "object",
                    "properties": {
                        "customer": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "address": {
                                    "type": "object",
                                    "properties": {
                                        "city": { "type": "string" },
                                        "postal_code": { "type": "string" }
                                    },
                                    "required": ["city"]
                                }
                            },
                            "required": ["name"]
                        }
                    },
                    "required": ["customer"]
                }),
                json!({
                    "customer": {
                        "name": "mira",
                        "address": { "city": "oslo" }
                    }
                }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn nullable_union_can_emit_null() {
    with_anthropic_cassette(
        "messages_strict_tools/nullable_union_can_emit_null",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_resolution",
                "Record ticket = 17 and resolution = null. Omit the optional reason field.",
                json!({
                    "type": "object",
                    "properties": {
                        "ticket": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "format": "uint32"
                        },
                        "resolution": { "type": ["string", "null"] },
                        "reason": { "type": "string" }
                    },
                    "required": ["ticket", "resolution"]
                }),
                json!({ "ticket": 17, "resolution": null }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn discriminated_any_of_preserves_const() {
    with_anthropic_cassette(
        "messages_strict_tools/discriminated_any_of_preserves_const",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_notification",
                "Record an email notification with kind = email and address = ops@example.com. Omit the optional subject.",
                json!({
                    "type": "object",
                    "properties": {
                        "notification": {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "kind": { "type": "string", "const": "email" },
                                        "address": { "type": "string", "format": "email" },
                                        "subject": { "type": "string" }
                                    },
                                    "required": ["kind", "address"]
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "kind": { "type": "string", "const": "sms" },
                                        "number": { "type": "string" }
                                    },
                                    "required": ["kind", "number"]
                                }
                            ]
                        }
                    },
                    "required": ["notification"]
                }),
                json!({
                    "notification": {
                        "kind": "email",
                        "address": "ops@example.com"
                    }
                }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn all_of_composition_roundtrip() {
    with_anthropic_cassette(
        "messages_strict_tools/all_of_composition_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_code",
                "Record code = alpha. Omit the optional label.",
                json!({
                    "type": "object",
                    "properties": {
                        "code": {
                            "allOf": [
                                { "type": "string" },
                                { "enum": ["alpha", "beta"] }
                            ]
                        },
                        "label": { "type": "string" }
                    },
                    "required": ["code"]
                }),
                json!({ "code": "alpha" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn defs_ref_preserves_optional_member() {
    with_anthropic_cassette(
        "messages_strict_tools/defs_ref_preserves_optional_member",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_destination",
                "Record destination city = kyoto. Omit the optional country member.",
                json!({
                    "$defs": {
                        "Place": {
                            "type": "object",
                            "properties": {
                                "city": { "type": "string" },
                                "country": { "type": "string" }
                            },
                            "required": ["city"]
                        }
                    },
                    "type": "object",
                    "properties": {
                        "destination": { "$ref": "#/$defs/Place" }
                    },
                    "required": ["destination"]
                }),
                json!({ "destination": { "city": "kyoto" } }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn draft7_definitions_ref_preserves_optional_member() {
    with_anthropic_cassette(
        "messages_strict_tools/draft7_definitions_ref_preserves_optional_member",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_coordinates",
                "Record point latitude = 59 and longitude = 18. Omit the optional label.",
                json!({
                    "definitions": {
                        "Point": {
                            "type": "object",
                            "properties": {
                                "latitude": { "type": "integer" },
                                "longitude": { "type": "integer" },
                                "label": { "type": "string" }
                            },
                            "required": ["latitude", "longitude"]
                        }
                    },
                    "type": "object",
                    "properties": {
                        "point": { "$ref": "#/definitions/Point" }
                    },
                    "required": ["point"]
                }),
                json!({ "point": { "latitude": 59, "longitude": 18 } }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn array_items_and_formats_roundtrip() {
    with_anthropic_cassette(
        "messages_strict_tools/array_items_and_formats_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_batch",
                "Record batch_id = 123e4567-e89b-12d3-a456-426614174000, due_date = 2026-09-21, and exactly two items: sku = red with quantity = 2, then sku = blue with quantity = 3. Omit every optional note.",
                json!({
                    "type": "object",
                    "properties": {
                        "batch_id": { "type": "string", "format": "uuid" },
                        "due_date": { "type": "string", "format": "date" },
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 3,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sku": { "type": "string" },
                                    "quantity": { "type": "integer" },
                                    "note": { "type": "string" }
                                },
                                "required": ["sku", "quantity"]
                            }
                        }
                    },
                    "required": ["batch_id", "due_date", "items"]
                }),
                json!({
                    "batch_id": "123e4567-e89b-12d3-a456-426614174000",
                    "due_date": "2026-09-21",
                    "items": [
                        { "sku": "red", "quantity": 2 },
                        { "sku": "blue", "quantity": 3 }
                    ]
                }),
            )
            .await;
        },
    )
    .await;
}

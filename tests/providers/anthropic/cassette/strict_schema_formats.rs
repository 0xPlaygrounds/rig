//! Live-recorded coverage for every string format Anthropic strict schemas support.

use serde_json::json;

use super::super::support::with_anthropic_cassette;
use super::messages_strict_tools::assert_strict_tool_call;

#[tokio::test]
async fn supported_date_time_format_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_formats/supported_date_time_format_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_date_time",
                "Record value = 2026-09-21T14:30:00Z exactly.",
                json!({
                    "type": "object",
                    "properties": { "value": { "type": "string", "format": "date-time" } },
                    "required": ["value"]
                }),
                json!({ "value": "2026-09-21T14:30:00Z" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn supported_time_format_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_formats/supported_time_format_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_time",
                "Record value = 14:30:00Z exactly.",
                json!({
                    "type": "object",
                    "properties": { "value": { "type": "string", "format": "time" } },
                    "required": ["value"]
                }),
                json!({ "value": "14:30:00Z" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn supported_date_format_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_formats/supported_date_format_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_date",
                "Record value = 2026-09-21 exactly.",
                json!({
                    "type": "object",
                    "properties": { "value": { "type": "string", "format": "date" } },
                    "required": ["value"]
                }),
                json!({ "value": "2026-09-21" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn supported_duration_format_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_formats/supported_duration_format_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_duration",
                "Record value = P3D exactly.",
                json!({
                    "type": "object",
                    "properties": { "value": { "type": "string", "format": "duration" } },
                    "required": ["value"]
                }),
                json!({ "value": "P3D" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn supported_email_format_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_formats/supported_email_format_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_email",
                "Record value = ops@example.com exactly.",
                json!({
                    "type": "object",
                    "properties": { "value": { "type": "string", "format": "email" } },
                    "required": ["value"]
                }),
                json!({ "value": "ops@example.com" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn supported_hostname_format_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_formats/supported_hostname_format_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_hostname",
                "Record value = api.example.com exactly.",
                json!({
                    "type": "object",
                    "properties": { "value": { "type": "string", "format": "hostname" } },
                    "required": ["value"]
                }),
                json!({ "value": "api.example.com" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn supported_uri_format_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_formats/supported_uri_format_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_uri",
                "Record value = https://example.com/path exactly.",
                json!({
                    "type": "object",
                    "properties": { "value": { "type": "string", "format": "uri" } },
                    "required": ["value"]
                }),
                json!({ "value": "https://example.com/path" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn supported_ipv4_format_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_formats/supported_ipv4_format_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_ipv4",
                "Record value = 192.0.2.1 exactly.",
                json!({
                    "type": "object",
                    "properties": { "value": { "type": "string", "format": "ipv4" } },
                    "required": ["value"]
                }),
                json!({ "value": "192.0.2.1" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn supported_ipv6_format_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_formats/supported_ipv6_format_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_ipv6",
                "Record value = 2001:db8::1 exactly.",
                json!({
                    "type": "object",
                    "properties": { "value": { "type": "string", "format": "ipv6" } },
                    "required": ["value"]
                }),
                json!({ "value": "2001:db8::1" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn supported_uuid_format_roundtrip() {
    with_anthropic_cassette(
        "strict_schema_formats/supported_uuid_format_roundtrip",
        |client| async move {
            assert_strict_tool_call(
                client,
                "record_uuid",
                "Record value = 123e4567-e89b-12d3-a456-426614174000 exactly.",
                json!({
                    "type": "object",
                    "properties": { "value": { "type": "string", "format": "uuid" } },
                    "required": ["value"]
                }),
                json!({ "value": "123e4567-e89b-12d3-a456-426614174000" }),
            )
            .await;
        },
    )
    .await;
}

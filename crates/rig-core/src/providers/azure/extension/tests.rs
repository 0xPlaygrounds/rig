//! Azure's Chat options as the bodies they encode to, and its extras. The
//! encoder tests are unit tests because no recording sends typed provider
//! options. No Azure recording exists, so the extras test decodes
//! Copilot's recorded reply, which carries the same content-filter fields,
//! through the Azure wire.

use serde_json::{Value, json};

use super::*;
use crate::providers::openai::wire::{AZURE, Chat, OpenAIConfig};
use crate::test_utils::provider_extensions::{
    assert_no_reserved_leaf, body_with, recorded_reply, reply_of,
};

fn chat_wire(api_version: &str) -> Chat {
    OpenAIConfig::with_key(&AZURE, "key")
        .with_base_url("https://example.openai.azure.com")
        .with_api_version(api_version)
        .chat(crate::providers::azure::GPT_4O_MINI)
}

fn body(options: &AzureOptions) -> Value {
    body_with::<AzureExt, _>(&chat_wire("2025-04-01-preview"), options)
}

#[test]
fn data_sources_land_at_top_level() {
    let source = json!({
        "type": "azure_search",
        "parameters": {"endpoint": "https://search.example", "index_name": "docs"}
    });
    let body = body(&AzureOptions::new().data_source(source.clone()));
    assert_eq!(body["data_sources"], json!([source]));
}

#[test]
fn openai_chat_fields_land_on_azure() {
    let body = body(&AzureOptions::new().chat(ChatOptions::new().logprobs(true).top_logprobs(2)));
    assert_eq!(body["logprobs"], true);
    assert_eq!(body["top_logprobs"], 2);
}

#[test]
fn no_option_writes_a_leaf_the_request_or_a_mapped_option_owns() {
    let options = AzureOptions::new()
        .chat(ChatOptions::new().logit_bias(1, 1).presence_penalty(0.2))
        .data_source(json!({"type": "azure_search"}));
    assert_no_reserved_leaf::<AzureExt, _>(
        &[chat_wire("2024-10-21"), chat_wire("2025-04-01-preview")],
        &options,
    );
}

#[tokio::test]
async fn extras_from_a_recorded_content_filtered_reply() {
    let reply = reply_of(
        chat_wire("2025-04-01-preview"),
        recorded_reply("copilot", "agent/completion_smoke", 0),
    )
    .await;
    let extras = reply
        .extras::<AzureExt>()
        .unwrap_or_else(|| panic!("an Azure reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    let prompt_filter = extras.prompt_filter_results.unwrap_or_default();
    assert_eq!(
        prompt_filter.first().map(|result| &result["prompt_index"]),
        Some(&json!(0))
    );
    assert_eq!(
        extras
            .content_filter_results
            .as_ref()
            .and_then(|results| results.pointer("/hate/filtered")),
        Some(&json!(false))
    );
    assert_eq!(extras.service_tier.as_deref(), Some("default"));
    assert_eq!(
        extras
            .completion_tokens_details
            .and_then(|details| details.rejected_prediction_tokens),
        Some(0)
    );
}

/// Each `AzureOptions` Chat field setter equals the `chat` form, as a value
/// and as the JSON it serializes to.
#[test]
fn field_setters_equal_the_section_form() {
    let chat = |section: ChatOptions| AzureOptions::new().chat(section);
    let pairs = [
        (
            AzureOptions::new().logprobs(true),
            chat(ChatOptions::new().logprobs(true)),
        ),
        (
            AzureOptions::new().top_logprobs(3),
            chat(ChatOptions::new().top_logprobs(3)),
        ),
        (
            AzureOptions::new().frequency_penalty(0.5),
            chat(ChatOptions::new().frequency_penalty(0.5)),
        ),
        (
            AzureOptions::new().presence_penalty(0.25),
            chat(ChatOptions::new().presence_penalty(0.25)),
        ),
        (
            AzureOptions::new().logit_bias(42, -100),
            chat(ChatOptions::new().logit_bias(42, -100)),
        ),
        (
            AzureOptions::new().prediction("draft"),
            chat(ChatOptions::new().prediction("draft")),
        ),
        (
            AzureOptions::new()
                .data_source(json!({"type": "azure_search"}))
                .logprobs(true)
                .top_logprobs(2),
            AzureOptions::new()
                .data_source(json!({"type": "azure_search"}))
                .chat(ChatOptions::new().logprobs(true).top_logprobs(2)),
        ),
    ];
    for (short, long) in pairs {
        assert_eq!(short, long);
        let short = serde_json::to_value(&short).expect("options serialize");
        assert_eq!(
            short,
            serde_json::to_value(&long).expect("options serialize")
        );
        assert_ne!(short["openai.chat"], json!({}));
    }
}

//! OpenRouter's typed options as the bodies they encode to, and its extras
//! read from recorded replies. The encoder tests are unit tests because no
//! recording sends typed provider options; the extras tests decode recorded
//! unary replies.

use serde_json::{Value, json};

use super::*;
use crate::completion::{CompletionRequest, Effort, ProviderOptions};
use crate::providers::deepseek::extension::DeepSeekExt;
use crate::providers::openai::wire::{Chat, OPENROUTER, OpenAIConfig};
use crate::test_utils::provider_extensions::{
    assert_no_reserved_leaf, body_with, encoded_body, recorded_reply, reply_of, request_with,
};
use crate::wire::Mode;

const MODEL: &str = "openai/gpt-4o-mini";

fn chat(model: &str) -> Chat {
    OpenAIConfig::with_key(&OPENROUTER, "key").chat(model)
}

fn body(options: &OpenRouterOptions) -> Value {
    body_with::<OpenRouterExt, _>(&chat(MODEL), options)
}

fn fallbacks(models: &[&str]) -> ModelFallbacks {
    ModelFallbacks::new(models.iter().copied()).unwrap_or_else(|error| panic!("{error}"))
}

#[test]
fn provider_preferences_land_under_provider() {
    let preferences = ProviderPreferences::new()
        .order(["anthropic", "openai"])
        .only(["anthropic"])
        .ignore(["deepinfra"])
        .allow_fallbacks(false)
        .require_parameters(true)
        .data_collection(DataCollection::Deny)
        .zdr(true)
        .sort(
            ProviderSortConfig::new(ProviderSortStrategy::Throughput)
                .partition(SortPartition::None),
        )
        .preferred_min_throughput(ThroughputThreshold::Percentile(
            PercentileThresholds::new()
                .p50(10.0)
                .p75(20.0)
                .p90(30.0)
                .p99(40.0),
        ))
        .preferred_max_latency(LatencyThreshold::Simple(2.5))
        .max_price(
            MaxPrice::new()
                .prompt(1.0)
                .completion(2.0)
                .request(0.5)
                .image(0.25),
        )
        .quantizations([Quantization::Int4, Quantization::Fp8]);
    let body = body(&OpenRouterOptions::new().provider(preferences));
    assert_eq!(
        body["provider"],
        json!({
            "order": ["anthropic", "openai"],
            "only": ["anthropic"],
            "ignore": ["deepinfra"],
            "allow_fallbacks": false,
            "require_parameters": true,
            "data_collection": "deny",
            "zdr": true,
            "sort": {"by": "throughput", "partition": "none"},
            "preferred_min_throughput": {"p50": 10.0, "p75": 20.0, "p90": 30.0, "p99": 40.0},
            "preferred_max_latency": 2.5,
            "max_price": {"prompt": 1.0, "completion": 2.0, "request": 0.5, "image": 0.25},
            "quantizations": ["int4", "fp8"]
        })
    );
}

#[test]
fn a_simple_sort_is_its_strategy() {
    let body = body(&OpenRouterOptions::new().provider(ProviderPreferences::new().cheapest()));
    assert_eq!(body["provider"], json!({"sort": "price"}));
}

#[test]
fn models_land_at_top_level() {
    let body = body(&OpenRouterOptions::new().models(fallbacks(&["anthropic/claude-sonnet-4.6"])));
    assert_eq!(body["models"], json!(["anthropic/claude-sonnet-4.6"]));
}

#[test]
fn plugins_land_at_top_level() {
    let body = body(&OpenRouterOptions::new().plugin(json!({"id": "web", "max_results": 3})));
    assert_eq!(body["plugins"], json!([{"id": "web", "max_results": 3}]));
}

#[test]
fn session_id_lands_at_top_level() {
    let body = body(&OpenRouterOptions::new().session_id("session-1"));
    assert_eq!(body["session_id"], "session-1");
}

#[test]
fn metadata_lands_under_metadata() {
    let body = body(&OpenRouterOptions::new().metadata("tenant", "acme"));
    assert_eq!(body["metadata"], json!({"tenant": "acme"}));
}

#[test]
fn reasoning_exclude_joins_the_mapped_effort() {
    let request = request_with::<OpenRouterExt>(&OpenRouterOptions::new().reasoning_exclude(true))
        .reasoning(Effort::High);
    let body =
        encoded_body(&chat(MODEL), request, Mode::Unary).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        body["reasoning"],
        json!({"effort": "high", "exclude": true})
    );
}

#[test]
fn reasoning_summary_lands_under_reasoning() {
    let body = body(&OpenRouterOptions::new().reasoning_summary(ReasoningSummary::Detailed));
    assert_eq!(body["reasoning"], json!({"summary": "detailed"}));
}

#[test]
fn top_k_lands_at_top_level() {
    assert_eq!(body(&OpenRouterOptions::new().top_k(40))["top_k"], 40);
}

#[test]
fn min_p_lands_at_top_level() {
    assert_eq!(body(&OpenRouterOptions::new().min_p(0.05))["min_p"], 0.05);
}

#[test]
fn top_a_lands_at_top_level() {
    assert_eq!(body(&OpenRouterOptions::new().top_a(0.2))["top_a"], 0.2);
}

#[test]
fn repetition_penalty_lands_at_top_level() {
    assert_eq!(
        body(&OpenRouterOptions::new().repetition_penalty(1.1))["repetition_penalty"],
        1.1
    );
}

#[test]
fn user_lands_at_top_level() {
    assert_eq!(
        body(&OpenRouterOptions::new().user("user-1"))["user"],
        "user-1"
    );
}

/// One entry serves both routes: the shared section reaches the Responses
/// body too.
#[test]
fn options_reach_the_responses_route() {
    let wire = OpenAIConfig::with_key(&OPENROUTER, "key").responses(MODEL);
    let options = OpenRouterOptions::new()
        .provider(ProviderPreferences::new().only(["openai"]))
        .top_k(5);
    let body = body_with::<OpenRouterExt, _>(&wire, &options);
    assert_eq!(body["provider"], json!({"only": ["openai"]}));
    assert_eq!(body["top_k"], 5);
}

#[test]
fn no_option_writes_a_leaf_the_request_or_a_mapped_option_owns() {
    let options = OpenRouterOptions::new()
        .provider(ProviderPreferences::new().zdr(true))
        .models(fallbacks(&["a/b"]))
        .plugin(json!({"id": "web"}))
        .session_id("s")
        .metadata("k", "v")
        .reasoning_exclude(true)
        .reasoning_summary(ReasoningSummary::Auto)
        .top_k(1)
        .min_p(0.1)
        .top_a(0.1)
        .repetition_penalty(1.0)
        .user("u");
    assert_no_reserved_leaf::<OpenRouterExt, _>(
        &[chat(MODEL), chat("anthropic/claude-sonnet-4.6")],
        &options,
    );
}

// Model fallbacks: the tests of the typed `ModelFallbacks` builder this
// options type replaces, against the encoded body.

#[test]
fn model_fallbacks_serialize_model_and_models() {
    let body = body_with::<OpenRouterExt, _>(
        &chat("openai/gpt-4o"),
        &OpenRouterOptions::new().models(fallbacks(&[
            "anthropic/claude-sonnet-4.6",
            "gryphe/mythomax-l2-13b",
        ])),
    );
    assert_eq!(body["model"], "openai/gpt-4o");
    assert_eq!(
        body["models"],
        json!(["anthropic/claude-sonnet-4.6", "gryphe/mythomax-l2-13b"])
    );
}

#[test]
fn model_fallbacks_keep_the_request_model_override_as_primary() {
    let mut request = request_with::<OpenRouterExt>(
        &OpenRouterOptions::new().models(fallbacks(&["anthropic/claude-sonnet-4.6"])),
    );
    request.model = Some("google/gemini-2.5-flash".to_owned());
    let body = encoded_body(&chat("openai/gpt-4o"), request, Mode::Unary)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(body["model"], "google/gemini-2.5-flash");
    assert_eq!(body["models"], json!(["anthropic/claude-sonnet-4.6"]));
}

#[test]
fn model_fallbacks_merge_with_provider_preferences() {
    let body = body(
        &OpenRouterOptions::new()
            .provider(ProviderPreferences::new().require_parameters(true))
            .models(fallbacks(&["anthropic/claude-sonnet-4.6"])),
    );
    assert_eq!(body["provider"]["require_parameters"], true);
    assert_eq!(body["models"], json!(["anthropic/claude-sonnet-4.6"]));
}

#[test]
fn model_fallbacks_reject_an_empty_list() {
    assert_eq!(ModelFallbacks::new(Vec::<&str>::new()), Err(EmptyFallbacks));
}

/// Raw `additional_params` rank above typed options, so the raw list wins.
#[test]
fn raw_models_beat_typed_fallbacks() {
    let request = request_with::<OpenRouterExt>(
        &OpenRouterOptions::new().models(fallbacks(&["typed-fallback"])),
    )
    .additional_params(json!({"models": ["from-additional-params"]}));
    let body = encoded_body(&chat("openai/gpt-4o"), request, Mode::Unary)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(body["models"], json!(["from-additional-params"]));
}

#[test]
fn model_fallbacks_are_omitted_when_unset() {
    let request = CompletionRequest::new("hi").provider_options(ProviderOptions::new());
    let body =
        encoded_body(&chat(MODEL), request, Mode::Unary).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(body["model"], MODEL);
    assert!(body.get("models").is_none(), "{body}");
}

#[test]
fn model_fallbacks_survive_a_streaming_encode() {
    let request = request_with::<OpenRouterExt>(
        &OpenRouterOptions::new().models(fallbacks(&["anthropic/claude-sonnet-4.6"])),
    );
    let body = encoded_body(&chat("openai/gpt-4o"), request, Mode::Streaming)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(body["model"], "openai/gpt-4o");
    assert_eq!(body["models"], json!(["anthropic/claude-sonnet-4.6"]));
    assert_eq!(body["stream"], true);
}

#[tokio::test]
async fn chat_extras_from_a_unary_recording() {
    let reply = reply_of(
        chat(MODEL),
        recorded_reply(
            "openrouter",
            "raw_capture_matrix/raw_round_trips_openrouter_type",
            0,
        ),
    )
    .await;
    let extras = reply
        .extras::<OpenRouterExt>()
        .unwrap_or_else(|| panic!("an OpenRouter reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(extras.provider.as_deref(), Some("Azure"));
    assert_eq!(extras.cost, Some(2.7e-6));
    assert_eq!(extras.native_finish_reason.as_deref(), Some("stop"));
    assert_eq!(extras.system_fingerprint.as_deref(), Some("fp_369e662417"));
    assert_eq!(extras.service_tier, None);
    assert_eq!(extras.is_byok, Some(false));
    assert_eq!(
        extras
            .cost_details
            .as_ref()
            .and_then(|details| details.get("upstream_inference_cost")),
        Some(&json!(2.7e-6))
    );
    assert_eq!(
        extras
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.get("cached_tokens")),
        Some(&json!(0))
    );
    assert_eq!(extras.annotations, None);
    assert!(reply.extras::<DeepSeekExt>().is_none());
}

#[tokio::test]
async fn responses_extras_from_a_unary_recording() {
    let reply = reply_of(
        OpenAIConfig::with_key(&OPENROUTER, "key").responses("google/gemini-3-flash-preview"),
        recorded_reply(
            "openrouter",
            "openai_responses_compat/openai_responses_raw_response_accepts_service_tier_metadata",
            0,
        ),
    )
    .await;
    let extras = reply
        .extras::<OpenRouterExt>()
        .unwrap_or_else(|| panic!("an OpenRouter reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(extras.cost, Some(2.85e-5));
    assert_eq!(extras.service_tier.as_deref(), Some("default"));
    assert_eq!(extras.is_byok, Some(false));
    assert_eq!(extras.provider, None);
    assert_eq!(extras.native_finish_reason, None);
    assert_eq!(
        extras
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.get("cache_write_tokens")),
        Some(&json!(0))
    );
}

/// A stream's `raw` is the unary document, so the extras of the recorded
/// stream equal those of the recorded unary answer of the same prompt,
/// `native_finish_reason` included.
#[tokio::test]
async fn chat_extras_read_alike_from_a_recorded_stream() {
    use crate::test_utils::provider_extensions::{recorded_stream, streamed_reply_of};

    let read = |reply: crate::completion::CompletionResponse| {
        reply
            .extras::<OpenRouterExt>()
            .unwrap_or_else(|| panic!("an OpenRouter reply"))
            .unwrap_or_else(|error| panic!("{error}"))
    };
    let streamed = read(
        streamed_reply_of(
            chat(MODEL),
            recorded_stream(
                "openrouter",
                "raw_stream_capture_matrix/stream_raw_exposes_terminal_cost_and_provider",
                0,
            ),
        )
        .await,
    );
    let unary = read(
        reply_of(
            chat(MODEL),
            recorded_reply(
                "openrouter",
                "raw_capture_matrix/raw_round_trips_openrouter_type",
                0,
            ),
        )
        .await,
    );
    assert_eq!(streamed, unary);
    assert_eq!(streamed.provider.as_deref(), Some("Azure"));
    assert_eq!(streamed.cost, Some(2.7e-6));
    assert_eq!(streamed.native_finish_reason.as_deref(), Some("stop"));
}

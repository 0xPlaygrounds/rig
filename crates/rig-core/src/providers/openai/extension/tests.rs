//! OpenAI's Chat options as the bodies they encode to, its sections as they
//! serialize, and its extras read from a recorded Chat reply and from built
//! Responses bodies. The encoder tests are unit tests because no recording
//! sends typed provider options; the recorded Responses replies are read in
//! the Responses wire's tests.

use serde_json::{Value, json};

use super::*;
use crate::completion::{GenerationOptions, OnUnsupported, ProviderOptions, Reasoning};
use crate::providers::openai::completion::{GPT_4_1_MINI, GPT_6_SOL};
use crate::providers::openai::wire::{OPENAI, OpenAIConfig};
use crate::providers::openrouter::extension::OpenRouterExt;
use crate::test_utils::TraceCapture;
use crate::test_utils::provider_extensions::{
    assert_no_reserved_leaf, body_with, encoded_body, recorded_reply, reply_of, request_with,
};
use crate::wire::Mode;

fn chat_wire(model: &str) -> crate::providers::openai::wire::Chat {
    OpenAIConfig::with_key(&OPENAI, "key").chat(model)
}

fn body(chat: ChatOptions) -> Value {
    body_with::<OpenAiExt, _>(&chat_wire(GPT_4_1_MINI), &OpenAiOptions::new().chat(chat))
}

#[test]
fn logit_bias_lands_under_logit_bias() {
    let body = body(ChatOptions::new().logit_bias(50256, -100).logit_bias(13, 5));
    assert_eq!(body["logit_bias"], json!({"13": 5, "50256": -100}));
}

#[test]
fn prediction_lands_as_content_prediction() {
    let body = body(ChatOptions::new().prediction("fn main() {}"));
    assert_eq!(
        body["prediction"],
        json!({"type": "content", "content": "fn main() {}"})
    );
}

#[test]
fn logprobs_lands_at_top_level() {
    assert_eq!(body(ChatOptions::new().logprobs(true))["logprobs"], true);
}

#[test]
fn top_logprobs_lands_at_top_level() {
    assert_eq!(body(ChatOptions::new().top_logprobs(5))["top_logprobs"], 5);
}

#[test]
fn frequency_penalty_lands_at_top_level() {
    assert_eq!(
        body(ChatOptions::new().frequency_penalty(0.5))["frequency_penalty"],
        0.5
    );
}

#[test]
fn presence_penalty_lands_at_top_level() {
    assert_eq!(
        body(ChatOptions::new().presence_penalty(-0.5))["presence_penalty"],
        -0.5
    );
}

#[test]
fn modalities_land_at_top_level() {
    let body = body(ChatOptions::new().modalities([Modality::Text, Modality::Audio]));
    assert_eq!(body["modalities"], json!(["text", "audio"]));
}

#[test]
fn audio_lands_at_top_level() {
    let body = body(ChatOptions::new().audio("alloy", AudioFormat::Pcm16));
    assert_eq!(body["audio"], json!({"voice": "alloy", "format": "pcm16"}));
}

#[test]
fn web_search_options_land_at_top_level() {
    let options = WebSearchOptions::new()
        .search_context_size(SearchContextSize::Low)
        .user_location(
            ApproximateLocation::new()
                .city("Paris")
                .country("FR")
                .region("Ile-de-France")
                .timezone("Europe/Paris"),
        );
    let body = body(ChatOptions::new().web_search_options(options));
    assert_eq!(
        body["web_search_options"],
        json!({
            "search_context_size": "low",
            "user_location": {
                "type": "approximate",
                "approximate": {
                    "city": "Paris",
                    "country": "FR",
                    "region": "Ile-de-France",
                    "timezone": "Europe/Paris"
                }
            }
        })
    );
}

/// OpenAI defaults to Responses: the Chat section is skipped there, with a
/// debug event and no warning.
#[test]
fn the_chat_section_is_not_sent_on_responses() {
    let wire = OpenAIConfig::with_key(&OPENAI, "key").responses(GPT_4_1_MINI);
    let options = OpenAiOptions::new().chat(ChatOptions::new().logprobs(true));
    let capture = TraceCapture::default();
    let body = tracing::subscriber::with_default(capture.subscriber(), || {
        body_with::<OpenAiExt, _>(&wire, &options)
    });
    assert!(body.get("logprobs").is_none(), "{body}");
    assert!(capture.warnings().is_empty(), "{:?}", capture.warnings());
}

/// Raw `additional_params` rank above typed options.
#[test]
fn raw_additional_params_beat_typed_options() {
    let request = request_with::<OpenAiExt>(
        &OpenAiOptions::new().chat(ChatOptions::new().frequency_penalty(0.5)),
    )
    .additional_params(json!({"frequency_penalty": 1.0}));
    let body = encoded_body(&chat_wire(GPT_4_1_MINI), request, Mode::Unary)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(body["frequency_penalty"], 1.0);
}

/// The final-body check sees typed fields too: GPT-6 Sol reasons by
/// default and then takes no `logprobs`, until reasoning is off.
#[test]
fn typed_logprobs_on_gpt_6_while_reasoning_fail_the_body_check() {
    let options = OpenAiOptions::new().chat(ChatOptions::new().logprobs(true));
    let strict = request_with::<OpenAiExt>(&options)
        .options(GenerationOptions::default().reasoning(crate::completion::Effort::Medium));
    let error = encoded_body(&chat_wire(GPT_6_SOL), strict, Mode::Unary)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(error.contains("`logprobs`"), "{error}");
    let lenient = request_with::<OpenAiExt>(&options)
        .options(GenerationOptions::default().on_unsupported(OnUnsupported::Ignore));
    let body = encoded_body(&chat_wire(GPT_6_SOL), lenient, Mode::Unary)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        body.get("logprobs"),
        None,
        "a typed field is left out: {body}"
    );
    let unset = encoded_body(
        &chat_wire(GPT_6_SOL),
        request_with::<OpenAiExt>(&options),
        Mode::Unary,
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(unset["logprobs"], true, "no option set: sent as built");
    let off = request_with::<OpenAiExt>(&options).options(
        GenerationOptions::default()
            .reasoning(Reasoning::Off)
            .on_unsupported(OnUnsupported::Error),
    );
    let body = encoded_body(&chat_wire(GPT_6_SOL), off, Mode::Unary)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(body["logprobs"], true);
}

#[test]
fn no_option_writes_a_leaf_the_request_or_a_mapped_option_owns() {
    let chat = ChatOptions::new()
        .logit_bias(1, 1)
        .prediction("p")
        .logprobs(true)
        .top_logprobs(1)
        .frequency_penalty(0.1)
        .presence_penalty(0.1)
        .modalities([Modality::Text])
        .audio("alloy", AudioFormat::Mp3)
        .web_search_options(WebSearchOptions::new().search_context_size(SearchContextSize::High));
    assert_no_reserved_leaf::<OpenAiExt, _>(
        &[
            chat_wire(GPT_4_1_MINI),
            chat_wire("gpt-5.6"),
            chat_wire(GPT_6_SOL),
        ],
        &OpenAiOptions::new().chat(chat),
    );
}

#[tokio::test]
async fn chat_extras_from_a_unary_recording() {
    let reply = reply_of(
        chat_wire("gpt-5-mini"),
        recorded_reply("openai", "corpus_matrix_chat/output_tool_thinking", 0),
    )
    .await;
    let extras = reply
        .extras::<OpenAiExt>()
        .unwrap_or_else(|| panic!("an OpenAI reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(extras.service_tier.as_deref(), Some("default"));
    assert_eq!(extras.system_fingerprint, None);
    let completion = extras.completion_tokens_details.unwrap_or_default();
    assert_eq!(completion.reasoning_tokens, Some(64));
    assert_eq!(completion.accepted_prediction_tokens, Some(0));
    let prompt = extras.prompt_tokens_details.unwrap_or_default();
    assert_eq!(prompt.cached_tokens, Some(0));
    assert_eq!(prompt.audio_tokens, Some(0));
    assert!(reply.extras::<OpenRouterExt>().is_none());
}

/// A fully set shared section serializes under `"*"`, and no section is
/// sent for a route the options leave empty.
#[test]
fn the_shared_section_serializes_under_the_shared_key() {
    let options = OpenAiOptions::default().shared(
        OpenAiShared::default()
            .store(false)
            .metadata("tenant", "acme")
            .prompt_cache_key("k1")
            .safety_identifier("u-1"),
    );
    let entry = ProviderOptions::new()
        .with::<OpenAiExt>(&options)
        .expect("the options are sections");
    assert_eq!(
        serde_json::to_value(entry.get::<OpenAiExt>()).expect("the sections serialize"),
        json!({"*": {
            "store": false,
            "metadata": {"tenant": "acme"},
            "prompt_cache_key": "k1",
            "safety_identifier": "u-1"
        }})
    );
}

/// Options that set nothing leave no entry.
#[test]
fn empty_options_leave_no_entry() {
    let entry = ProviderOptions::new()
        .with::<OpenAiExt>(&OpenAiOptions::default())
        .expect("the options are sections");
    assert!(entry.is_empty());
}

/// A compatible server's text `reasoning` gives no reasoning fields rather
/// than failing the whole view, and a reply with no envelope fields reads
/// all `None`.
#[test]
fn a_text_reasoning_reads_as_no_reasoning() {
    let api = Api::from_static("openai.responses");
    let extras = OpenAiExtras::from_reply(
        &api,
        &json!({"reasoning": "thought about it", "service_tier": "flex", "output": []}),
    )
    .expect("the view reads");
    assert_eq!(
        extras,
        OpenAiExtras {
            service_tier: Some("flex".to_owned()),
            ..OpenAiExtras::default()
        }
    );
    assert_eq!(
        OpenAiExtras::from_reply(&api, &json!({})).expect("the view reads"),
        OpenAiExtras::default()
    );
}

/// A stream's `raw` is the unary document, so the extras of the recorded
/// stream equal those of the recorded unary answer of the same prompt.
#[tokio::test]
async fn chat_extras_read_alike_from_a_recorded_stream() {
    use crate::test_utils::provider_extensions::{recorded_stream, streamed_reply_of};

    let read = |reply: crate::completion::CompletionResponse| {
        reply
            .extras::<OpenAiExt>()
            .unwrap_or_else(|| panic!("an OpenAI reply"))
            .unwrap_or_else(|error| panic!("{error}"))
    };
    let streamed = read(
        streamed_reply_of(
            chat_wire("gpt-4.1-nano"),
            recorded_stream(
                "openai",
                "raw_stream_capture_matrix/chat_stream_raw_round_trips_typed",
                0,
            ),
        )
        .await,
    );
    let unary = read(
        reply_of(
            chat_wire("gpt-4.1-nano"),
            recorded_reply("openai", "raw_capture_matrix/chat_raw_round_trips_typed", 0),
        )
        .await,
    );
    assert_eq!(streamed, unary);
    assert_eq!(streamed.service_tier.as_deref(), Some("default"));
}

/// Each `OpenAiOptions` field setter equals its section form, as a value
/// and as the JSON it serializes to.
#[test]
fn field_setters_equal_the_section_form() {
    let shared = |section: OpenAiShared| OpenAiOptions::new().shared(section);
    let chat = |section: ChatOptions| OpenAiOptions::new().chat(section);
    let responses = |section: OpenAiResponsesOptions| OpenAiOptions::new().responses(section);
    let pairs = [
        (
            OpenAiOptions::new().store(false),
            shared(OpenAiShared::default().store(false)),
        ),
        (
            OpenAiOptions::new().metadata("k", "v"),
            shared(OpenAiShared::default().metadata("k", "v")),
        ),
        (
            OpenAiOptions::new().prompt_cache_key("key"),
            shared(OpenAiShared::default().prompt_cache_key("key")),
        ),
        (
            OpenAiOptions::new().safety_identifier("user"),
            shared(OpenAiShared::default().safety_identifier("user")),
        ),
        (
            OpenAiOptions::new().reasoning_summary(ReasoningSummary::Auto),
            responses(OpenAiResponsesOptions::default().reasoning_summary(ReasoningSummary::Auto)),
        ),
        (
            OpenAiOptions::new().include([Include::ReasoningEncryptedContent]),
            responses(
                OpenAiResponsesOptions::default().include([Include::ReasoningEncryptedContent]),
            ),
        ),
        (
            OpenAiOptions::new().conversation("conv"),
            responses(OpenAiResponsesOptions::default().conversation("conv")),
        ),
        (
            OpenAiOptions::new().truncation(Truncation::Auto),
            responses(OpenAiResponsesOptions::default().truncation(Truncation::Auto)),
        ),
        (
            OpenAiOptions::new().background(true),
            responses(OpenAiResponsesOptions::default().background(true)),
        ),
        (
            OpenAiOptions::new().max_tool_calls(3),
            responses(OpenAiResponsesOptions::default().max_tool_calls(3)),
        ),
        (
            OpenAiOptions::new().logit_bias(42, -100),
            chat(ChatOptions::new().logit_bias(42, -100)),
        ),
        (
            OpenAiOptions::new().prediction("draft"),
            chat(ChatOptions::new().prediction("draft")),
        ),
        (
            OpenAiOptions::new().frequency_penalty(0.5),
            chat(ChatOptions::new().frequency_penalty(0.5)),
        ),
        (
            OpenAiOptions::new().presence_penalty(0.25),
            chat(ChatOptions::new().presence_penalty(0.25)),
        ),
        (
            OpenAiOptions::new()
                .store(false)
                .metadata("k", "v")
                .reasoning_summary(ReasoningSummary::Detailed)
                .background(true)
                .frequency_penalty(0.5),
            OpenAiOptions::new()
                .shared(OpenAiShared::default().store(false).metadata("k", "v"))
                .responses(
                    OpenAiResponsesOptions::default()
                        .reasoning_summary(ReasoningSummary::Detailed)
                        .background(true),
                )
                .chat(ChatOptions::new().frequency_penalty(0.5)),
        ),
    ];
    for (short, long) in pairs {
        assert_eq!(short, long);
        let short = serde_json::to_value(&short).expect("options serialize");
        assert_eq!(
            short,
            serde_json::to_value(&long).expect("options serialize")
        );
        assert_ne!(
            short,
            serde_json::to_value(OpenAiOptions::new()).expect("options serialize")
        );
    }
}

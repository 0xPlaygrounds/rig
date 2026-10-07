//! Typed provider options on the Responses wire, and the typed extras of its
//! replies. Each encoder test sets one field and checks the body is the
//! baseline body with that field's JSON merged in. The extras are read from
//! recorded replies, and read the same from a stream of the turn.

use super::*;
use crate::completion::options::{CacheRetention, Effort, GenerationOptions, OnUnsupported};
use crate::completion::{ProviderOptions, ReplayTarget};
use crate::providers::chatgpt::extension::{ChatGptExt, ChatGptExtras, ChatGptOptions};
use crate::providers::openai::extension::{
    AccessPrograms, ContextManagement, CyberAccess, Include, ItemPhase, OpenAiExt, OpenAiExtras,
    OpenAiOptions, OpenAiResponsesOptions, OpenAiShared, ReasoningContext, ReasoningMode,
    ReasoningSummary, Truncation,
};
use crate::providers::openai::responses_api::Delivery;
use crate::providers::openai::{GPT_5_6, GPT_5_6_SOL, GPT_6_SOL};
use serde_json::{Value, json};

fn gpt_6() -> Responses {
    OpenAIConfig::new("test-key").responses(GPT_6_SOL)
}

/// `options` as the request's only provider entry, for `P`.
fn entry<P: crate::completion::ProviderExtension>(options: &P::Options) -> ProviderOptions {
    ProviderOptions::new()
        .with::<P>(options)
        .expect("the options are an object of sections")
}

fn shared(shared: OpenAiShared) -> ProviderOptions {
    entry::<OpenAiExt>(&OpenAiOptions::default().shared(shared))
}

fn responses(responses: OpenAiResponsesOptions) -> ProviderOptions {
    entry::<OpenAiExt>(&OpenAiOptions::default().responses(responses))
}

/// The unary body `wire` sends for the bare prompt with `generation` and
/// `provider`.
fn body(wire: &Responses, generation: GenerationOptions, provider: ProviderOptions) -> Value {
    prepared_body_of(
        wire,
        prompt().options(generation).provider_options(provider),
    )
}

/// `base` with `patch` merged in, objects key by key.
fn merged(mut base: Value, patch: Value) -> Value {
    fn merge(base: &mut Value, patch: Value) {
        match (base, patch) {
            (Value::Object(base), Value::Object(patch)) => {
                for (key, value) in patch {
                    merge(base.entry(key).or_insert(Value::Null), value);
                }
            }
            (base, patch) => *base = patch,
        }
    }
    merge(&mut base, patch);
    base
}

/// The body for `provider` on `wire` is the body without it, with `patch`
/// merged in.
#[track_caller]
fn assert_adds(
    wire: &Responses,
    generation: GenerationOptions,
    provider: ProviderOptions,
    patch: Value,
) {
    let baseline = body(wire, generation.clone(), ProviderOptions::new());
    assert_eq!(body(wire, generation, provider), merged(baseline, patch));
}

// ── the shared section ─────────────────────────────────────────────────

/// `store: false` reaches the body, asks for the reasoning ciphertext, and
/// leaves out a prior reasoning item the provider would have to resolve
/// from stored state.
#[test]
fn shared_store_reaches_the_body_and_shapes_history_stateless() {
    let wire = OpenAIConfig::new("test-key").responses("gpt-5.4");
    let reply = json!({
        "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-5.4",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "answer", "annotations": []}]}
        ]
    });
    let reply = crate::test_utils::history::decode(
        &wire,
        Mode::Unary,
        [crate::wire::WireFrame::Text(reply.to_string())],
    )
    .expect("the reply decodes");
    let history = vec![
        Message::user("q"),
        reply.message().expect("the reply is a turn"),
        Message::user("again"),
    ];
    let reasoning_ids = |body: &Value| -> Vec<Value> {
        body["input"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|item| item["type"] == "reasoning")
            .map(|item| item["id"].clone())
            .collect()
    };

    let stored = prepared_body_of(&wire, turn(history.clone()));
    let stateless = prepared_body_of(
        &wire,
        turn(history).provider_options(shared(OpenAiShared::default().store(false))),
    );

    assert_eq!(reasoning_ids(&stored), vec![json!("rs_1")]);
    assert_eq!(stateless["store"], json!(false));
    assert_eq!(stateless["include"], json!(["reasoning.encrypted_content"]));
    assert!(reasoning_ids(&stateless).is_empty(), "{stateless}");
}

#[test]
fn shared_metadata_reaches_the_body() {
    assert_adds(
        &gpt_6(),
        GenerationOptions::default(),
        shared(OpenAiShared::default().metadata("tenant", "acme")),
        json!({"metadata": {"tenant": "acme"}}),
    );
}

#[test]
fn shared_prompt_cache_key_reaches_the_body() {
    assert_adds(
        &gpt_6(),
        GenerationOptions::default(),
        shared(OpenAiShared::default().prompt_cache_key("k1")),
        json!({"prompt_cache_key": "k1"}),
    );
}

#[test]
fn shared_safety_identifier_reaches_the_body() {
    assert_adds(
        &gpt_6(),
        GenerationOptions::default(),
        shared(OpenAiShared::default().safety_identifier("u-1")),
        json!({"safety_identifier": "u-1"}),
    );
}

// ── the Responses section ──────────────────────────────────────────────

#[test]
fn reasoning_summary_joins_the_mapped_effort() {
    assert_adds(
        &gpt_6(),
        GenerationOptions::default().reasoning(Effort::High),
        responses(OpenAiResponsesOptions::default().reasoning_summary(ReasoningSummary::Auto)),
        json!({"reasoning": {"effort": "high", "summary": "auto"}}),
    );
}

/// The typed mode beside the mapped effort sends the body recorded in
/// `openai/gpt_5_6_reasoning/mode_pro_with_independent_effort.yaml`, which
/// the cassette test replays.
#[test]
fn reasoning_mode_matches_the_recorded_request() {
    let wire = OpenAIConfig::new("test-key").responses(GPT_5_6_SOL);
    let body = prepared_body_of(
        &wire,
        CompletionRequest::new("Reply with exactly: OK")
            .reasoning(Effort::High)
            .provider_options(responses(
                OpenAiResponsesOptions::default().reasoning_mode(ReasoningMode::Pro),
            )),
    );
    assert_eq!(
        body,
        json!({
            "include": ["reasoning.encrypted_content"],
            "input": [{
                "content": [{"text": "Reply with exactly: OK", "type": "input_text"}],
                "role": "user",
                "type": "message"
            }],
            "model": "gpt-5.6-sol",
            "reasoning": {"effort": "high", "mode": "pro"}
        })
    );
}

/// With no mapped effort, the typed context alone makes the `reasoning`
/// object, which also asks for the ciphertext.
#[test]
fn reasoning_context_reaches_the_body() {
    assert_adds(
        &gpt_6(),
        GenerationOptions::default(),
        responses(
            OpenAiResponsesOptions::default().reasoning_context(ReasoningContext::CurrentTurn),
        ),
        json!({
            "reasoning": {"context": "current_turn"},
            "include": ["reasoning.encrypted_content"]
        }),
    );
}

#[test]
fn include_is_unioned_with_the_ciphertext() {
    assert_adds(
        &gpt_6(),
        GenerationOptions::default().reasoning(Effort::Low),
        responses(OpenAiResponsesOptions::default().include([Include::FileSearchCallResults])),
        json!({"include": ["file_search_call.results", "reasoning.encrypted_content"]}),
    );
}

#[test]
fn conversation_reaches_the_body_and_continues_stored() {
    let options = responses(OpenAiResponsesOptions::default().conversation("conv_1"));
    let wire = gpt_6();
    assert!(wire.continues_stored(&prompt().provider_options(options.clone())));
    assert!(!wire.continues_stored(&prompt()));
    assert_adds(
        &wire,
        GenerationOptions::default(),
        options,
        json!({"conversation": "conv_1"}),
    );
}

#[test]
fn truncation_reaches_the_body() {
    assert_adds(
        &gpt_6(),
        GenerationOptions::default(),
        responses(OpenAiResponsesOptions::default().truncation(Truncation::Auto)),
        json!({"truncation": "auto"}),
    );
}

#[test]
fn context_management_reaches_the_body() {
    assert_adds(
        &gpt_6(),
        GenerationOptions::default(),
        responses(
            OpenAiResponsesOptions::default()
                .context_management([ContextManagement::compaction(Some(1000))]),
        ),
        json!({"context_management": [{"type": "compaction", "compact_threshold": 1000}]}),
    );
}

/// The typed comparison id joins the `ttl` the mapped cache retention
/// sends.
#[test]
fn prompt_cache_comparison_joins_the_mapped_ttl() {
    assert_adds(
        &gpt_6(),
        GenerationOptions::default().cache(CacheRetention::Long),
        responses(OpenAiResponsesOptions::default().prompt_cache_comparison("resp_0")),
        json!({"prompt_cache_options": {"ttl": "30m", "comparison_response_id": "resp_0"}}),
    );
}

#[test]
fn background_reaches_the_http_body() {
    assert_adds(
        &gpt_6(),
        GenerationOptions::default(),
        responses(OpenAiResponsesOptions::default().background(true)),
        json!({"background": true}),
    );
}

/// A WebSocket session takes no `background`: a typed one is refused under
/// the default policy and left out with a warning under `Ignore`, while a
/// raw one is dropped as before.
#[test]
fn background_is_refused_on_a_websocket_session() {
    let wire = gpt_6();
    let typed = responses(OpenAiResponsesOptions::default().background(true));

    let error = wire
        .responses_request(
            &prompt().provider_options(typed.clone()),
            Delivery::WebSocket,
        )
        .expect_err("a typed background is refused");
    assert!(
        error
            .to_string()
            .contains("openai.openai.responses.background"),
        "{error}"
    );

    let ignored = wire
        .responses_request(
            &prompt()
                .on_unsupported(OnUnsupported::Ignore)
                .provider_options(typed),
            Delivery::WebSocket,
        )
        .expect("an ignored refusal encodes");
    assert!(ignored.get("background").is_none());

    let raw = wire
        .responses_request(
            &prompt().additional_params(json!({"background": true})),
            Delivery::WebSocket,
        )
        .expect("a raw background is dropped");
    assert!(raw.get("background").is_none());
}

#[test]
fn max_tool_calls_reaches_the_body() {
    assert_adds(
        &gpt_6(),
        GenerationOptions::default(),
        responses(OpenAiResponsesOptions::default().max_tool_calls(3)),
        json!({"max_tool_calls": 3}),
    );
}

/// GPT-6 takes `top_logprobs` only at effort `none`: the final-body check
/// refuses a typed one as it refuses a raw one.
#[test]
fn top_logprobs_is_checked_like_a_raw_key() {
    let wire = gpt_6();
    let typed = responses(OpenAiResponsesOptions::default().top_logprobs(2));
    let strict = prompt()
        .provider_options(typed.clone())
        .reasoning(crate::completion::Effort::Medium);
    let error = wire
        .encode(strict, Mode::Unary)
        .expect_err("gpt-6 reasons by default");
    assert_eq!(
        error
            .unsupported_option()
            .map(|refused| refused.option.as_ref()),
        Some("top_logprobs"),
        "{error}"
    );
    assert_adds(
        &wire,
        GenerationOptions::default(),
        typed.clone(),
        json!({"top_logprobs": 2}),
    );

    assert_adds(
        &wire,
        GenerationOptions::default().reasoning(crate::completion::options::Reasoning::Off),
        typed,
        json!({"top_logprobs": 2}),
    );
}

#[test]
fn access_programs_reaches_the_body() {
    assert_adds(
        &gpt_6(),
        GenerationOptions::default(),
        responses(
            OpenAiResponsesOptions::default()
                .access_programs(AccessPrograms::cyber(CyberAccess::Standard)),
        ),
        json!({"access_programs": {"cyber": "standard"}}),
    );
}

/// `additional_params` stays the top layer: a raw key beats the typed one.
#[test]
fn raw_additional_params_beat_typed_provider_options() {
    let body = prepared_body_of(
        &gpt_6(),
        prompt()
            .provider_options(shared(OpenAiShared::default().store(true)))
            .additional_params(json!({"store": false})),
    );
    assert_eq!(body["store"], json!(false));
}

/// Another provider's entry is not read: an `openai` entry on the ChatGPT
/// wire sends nothing.
#[test]
fn an_openai_entry_is_not_read_by_chatgpt() {
    assert_adds(
        &chatgpt(),
        GenerationOptions::default(),
        shared(OpenAiShared::default().prompt_cache_key("k")),
        json!({}),
    );
}

// ── ChatGPT ────────────────────────────────────────────────────────────

/// The typed key reaches the body, and the backend still gets
/// `store: false`.
#[test]
fn chatgpt_prompt_cache_key_reaches_the_body() {
    let options = entry::<ChatGptExt>(&ChatGptOptions::default().prompt_cache_key("k"));
    let wire = chatgpt();
    assert_adds(
        &wire,
        GenerationOptions::default(),
        options.clone(),
        json!({"prompt_cache_key": "k"}),
    );
    assert_eq!(
        body(&wire, GenerationOptions::default(), options)["store"],
        json!(false)
    );
}

#[test]
fn chatgpt_client_metadata_reaches_the_body() {
    assert_adds(
        &chatgpt(),
        GenerationOptions::default(),
        entry::<ChatGptExt>(&ChatGptOptions::default().client_metadata("x", "y")),
        json!({"client_metadata": {"x": "y"}}),
    );
}

#[test]
fn chatgpt_access_programs_reaches_the_body() {
    assert_adds(
        &chatgpt(),
        GenerationOptions::default(),
        entry::<ChatGptExt>(
            &ChatGptOptions::default()
                .access_programs(AccessPrograms::cyber(CyberAccess::Standard)),
        ),
        json!({"access_programs": {"cyber": "standard"}}),
    );
}

// ── ported from the pull requests these options supersede ──────────────

/// The typed `prompt_cache_options` field and the mapped retention build
/// the object raw `additional_params` would, and send no
/// `prompt_cache_retention`. The typed `mode` and `prewarm` of the original
/// are not options: `mode` belongs to the cache retention and the WebSocket
/// warm-up owns `prewarm`.
#[test]
fn prompt_cache_options_reach_the_request_typed_and_through_additional_params() {
    let wire = gpt_6();
    let typed = body(
        &wire,
        GenerationOptions::default().cache(CacheRetention::Long),
        responses(OpenAiResponsesOptions::default().prompt_cache_comparison("resp_0")),
    );
    let raw = prepared_body_of(
        &wire,
        prompt().additional_params(json!({
            "prompt_cache_options": {"ttl": "30m", "comparison_response_id": "resp_0"}
        })),
    );
    assert_eq!(typed, raw);
    assert_eq!(
        typed["prompt_cache_options"],
        json!({"ttl": "30m", "comparison_response_id": "resp_0"})
    );
    assert!(typed.get("prompt_cache_retention").is_none());
}

/// The cache key and the cache options sit at the top level of the body.
#[test]
fn prompt_cache_options_serialize_at_request_top_level() {
    let body = body(
        &OpenAIConfig::new("test-key").responses(GPT_5_6),
        GenerationOptions::default().cache(CacheRetention::Long),
        shared(OpenAiShared::default().prompt_cache_key("tenant:acme:v1")),
    );
    assert_eq!(body["prompt_cache_key"], json!("tenant:acme:v1"));
    assert_eq!(body["prompt_cache_options"]["ttl"], json!("30m"));
}

// ── extras from recorded replies ───────────────────────────────────────

/// The recorded pro-mode reply: its tier, effective reasoning, retention,
/// message phase and payer. Another provider's extras are not read from it.
#[tokio::test]
async fn openai_extras_from_a_unary_recording() {
    let body = cassette_body("openai/gpt_5_6_reasoning/mode_pro_with_independent_effort.yaml");
    let response = folded_unary(OpenAIConfig::new("test-key").responses(GPT_5_6_SOL), &body).await;

    let extras = response
        .extras::<OpenAiExt>()
        .expect("the reply is OpenAI's")
        .expect("the reply holds the extras");
    assert_eq!(
        extras,
        OpenAiExtras {
            service_tier: Some("default".to_owned()),
            reasoning_effort: Some("high".to_owned()),
            reasoning_summary: None,
            reasoning_mode: Some("pro".to_owned()),
            reasoning_context: Some("all_turns".to_owned()),
            prompt_cache_retention: Some("24h".to_owned()),
            incomplete_reason: None,
            phases: Some(vec![ItemPhase {
                id: "msg_01482f6e798b81cc006ab396ad536087d0bbd9124b114f5411".to_owned(),
                phase: Some("final_answer".to_owned()),
            }]),
            billing_payer: Some("developer".to_owned()),
            ..OpenAiExtras::default()
        }
    );
    assert!(response.extras::<ChatGptExt>().is_none());
}

/// A reply capped by its output limit, whose only item is reasoning.
#[tokio::test]
async fn openai_extras_of_an_incomplete_unary_recording() {
    let body = cassette_body("openai/reasoning_matrix_responses/capped.yaml");
    let response = folded_unary(gpt_6(), &body).await;

    let extras = response
        .extras::<OpenAiExt>()
        .expect("the reply is OpenAI's")
        .expect("the reply holds the extras");
    assert_eq!(
        extras.incomplete_reason.as_deref(),
        Some("max_output_tokens")
    );
    assert_eq!(extras.reasoning_effort.as_deref(), Some("low"));
    assert_eq!(extras.reasoning_summary.as_deref(), Some("detailed"));
    assert_eq!(extras.reasoning_mode.as_deref(), Some("standard"));
    assert_eq!(extras.reasoning_context.as_deref(), Some("current_turn"));
    assert_eq!(extras.prompt_cache_retention.as_deref(), Some("in_memory"));
    assert_eq!(extras.phases, None);
    assert_eq!(extras.billing_payer.as_deref(), Some("developer"));
}

/// ChatGPT answers a unary request with an event stream; its terminal
/// response object carries the envelope and an empty `output`, which the
/// items the stream finished fill.
#[tokio::test]
async fn chatgpt_extras_from_a_unary_recording() {
    let sse = cassette_body("chatgpt/codex_sessions/long_history_replay_nonstreaming.yaml");
    let response = folded_unary(chatgpt(), &sse).await;

    let extras = response
        .extras::<ChatGptExt>()
        .expect("the reply is ChatGPT's")
        .expect("the reply holds the extras");
    assert_eq!(
        extras,
        ChatGptExtras {
            service_tier: Some("default".to_owned()),
            reasoning_effort: Some("none".to_owned()),
            reasoning_summary: None,
            reasoning_mode: Some("standard".to_owned()),
            reasoning_context: Some("current_turn".to_owned()),
            prompt_cache_retention: Some("24h".to_owned()),
            incomplete_reason: None,
            phases: None,
        }
    );
    assert!(response.extras::<OpenAiExt>().is_none());
}

/// One prompt answered unary and streamed: every field reads the same from
/// both, except the payer, which only a unary body states.
#[tokio::test]
async fn openai_extras_read_the_same_from_streamed_and_unary_recordings() {
    let wire = || OpenAIConfig::new("test-key").responses("gpt-4.1-nano");
    let unary = cassette_body(
        "openai/raw_capture_matrix/responses_raw_exposes_service_tier_and_store.yaml",
    );
    let streamed =
        cassette_body("openai/raw_stream_capture_matrix/responses_stream_raw_exposes_status.yaml");
    let extras = |response: completion::CompletionResponse| {
        response
            .extras::<OpenAiExt>()
            .expect("the reply is OpenAI's")
            .expect("the reply holds the extras")
    };
    let from_body = extras(folded_unary(wire(), &unary).await);
    let from_stream = extras(folded_stream(wire(), &streamed).await);

    assert_eq!(from_body.billing_payer.as_deref(), Some("developer"));
    assert_eq!(from_stream.billing_payer, None);
    assert_eq!(from_stream.service_tier.as_deref(), Some("default"));
    assert_eq!(
        from_stream.prompt_cache_retention.as_deref(),
        Some("in_memory")
    );
    let phases = |extras: &OpenAiExtras| extras.phases.as_ref().map(Vec::len);
    assert_eq!(phases(&from_stream), Some(1));
    assert_eq!(phases(&from_body), Some(1));
    assert_eq!(
        OpenAiExtras {
            billing_payer: None,
            phases: None,
            ..from_body
        },
        OpenAiExtras {
            phases: None,
            ..from_stream
        }
    );
}

/// The ChatGPT backend's terminal states no output: the message's phase
/// is read from the items the stream finished, on the unary path and the
/// streamed one alike.
#[tokio::test]
async fn chatgpt_extras_read_the_same_phases_from_both_paths() {
    let sse = cassette_body("chatgpt/codex_sessions/streamed_phase_round_trips_on_follow_up.yaml");
    let extras = |response: completion::CompletionResponse| {
        response
            .extras::<ChatGptExt>()
            .expect("the reply is ChatGPT's")
            .expect("the reply holds the extras")
    };
    let from_body = extras(folded_unary(chatgpt(), &sse).await);
    let from_stream = extras(folded_stream(chatgpt(), &sse).await);

    assert_eq!(
        from_stream.phases,
        Some(vec![ItemPhase {
            id: "msg_0f892fbd3994996e016abc8bbc3e3c87d2913ed196f7d239c2".to_owned(),
            phase: Some("final_answer".to_owned()),
        }])
    );
    assert_eq!(from_stream, from_body);
}

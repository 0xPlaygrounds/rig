//! Each Gemini provider option reaches the body of the route that reads it.
//! These are unit tests: the request side is deterministic encoding, and a
//! recorded request with each option set does not exist. The extras are
//! read from recorded replies in the cassette targets.

use serde_json::{Map, Value, json};

use super::*;
use crate::completion::{
    CompletionRequest, GenerationOptions, OnUnsupported, ProviderOptions, Reasoning, ReplayTarget,
    ServiceTier, ToolDefinition,
};
use crate::error::ProviderError;
use crate::message::{ToolChoice, ToolName};
use crate::operation::Completion;
use crate::providers::gemini::GeminiConfig;
use crate::providers::gemini::completion::GenerateContent;
use crate::providers::gemini::interactions_api::{InteractionResume, Interactions};
use crate::wire::{Body, Mode, Operation, Wire};

/// The body `wire` sends for `request`, prepared as a model call prepares
/// it.
fn sent<W>(wire: &W, request: CompletionRequest) -> Result<Value, ProviderError>
where
    W: Wire<Op = Completion, Payload = crate::wire::Encoded>,
{
    let request = Completion::prepare(request, &wire.describe())?;
    let encoded = wire.encode(request, Mode::Unary)?;
    let Body::Bytes(bytes) = encoded.request.body() else {
        return Err(ProviderError::request("a Gemini body is JSON"));
    };
    Ok(serde_json::from_slice(bytes)?)
}

fn rest() -> GenerateContent {
    GenerateContent::new(GeminiConfig::new("k"), "gemini-2.5-flash")
}

fn interactions() -> Interactions {
    Interactions::new(GeminiConfig::new("k"), "gemini-2.5-flash")
}

/// A request carrying `options` as the Gemini entry.
fn with(options: &GeminiOptions) -> CompletionRequest {
    CompletionRequest::new("hi").provider_options(
        ProviderOptions::new()
            .with::<GeminiExt>(options)
            .expect("Gemini options are sections"),
    )
}

/// The GenerateContent body for `config`.
fn generate(config: GenerationConfig) -> Value {
    let options = GeminiOptions::new()
        .generate_content(GenerateContentOptions::new().generation_config(config));
    sent(&rest(), with(&options)).expect("the body builds")
}

/// The Interactions body for `section`.
fn interact(section: InteractionsOptions) -> Value {
    sent(
        &interactions(),
        with(&GeminiOptions::new().interactions(section)),
    )
    .expect("the body builds")
}

#[test]
fn store_reaches_generate_content_and_interactions() {
    let options = GeminiOptions::new().store(false);
    let body = sent(&rest(), with(&options)).expect("the body builds");
    assert_eq!(body["store"], json!(false));
    let body = sent(&interactions(), with(&options)).expect("the body builds");
    assert_eq!(body["store"], json!(false));
}

#[test]
fn labels_reach_generate_content_and_interactions() {
    let options = GeminiOptions::new().label("team", "rig");
    let body = sent(&rest(), with(&options)).expect("the body builds");
    assert_eq!(body["labels"], json!({"team": "rig"}));
    let body = sent(&interactions(), with(&options)).expect("the body builds");
    assert_eq!(body["labels"], json!({"team": "rig"}));
}

#[test]
fn include_thoughts_joins_the_mapped_thinking_config() {
    let options = GeminiOptions::new().generate_content(
        GenerateContentOptions::new()
            .generation_config(GenerationConfig::new().include_thoughts(true)),
    );
    let request = with(&options)
        .options(GenerationOptions::default().reasoning(Reasoning::Budget { tokens: 1024 }));
    let body = sent(&rest(), request).expect("the body builds");
    assert_eq!(
        body["generationConfig"]["thinkingConfig"],
        json!({"thinkingBudget": 1024, "includeThoughts": true})
    );
}

#[test]
fn top_k_lands_in_generation_config() {
    let body = generate(GenerationConfig::new().top_k(40));
    assert_eq!(body["generationConfig"]["topK"], json!(40));
}

#[test]
fn presence_penalty_lands_in_generation_config() {
    let body = generate(GenerationConfig::new().presence_penalty(0.5));
    assert_eq!(body["generationConfig"]["presencePenalty"], json!(0.5));
}

#[test]
fn frequency_penalty_lands_in_generation_config() {
    let body = generate(GenerationConfig::new().frequency_penalty(-0.5));
    assert_eq!(body["generationConfig"]["frequencyPenalty"], json!(-0.5));
}

#[test]
fn response_logprobs_lands_in_generation_config() {
    let body = generate(GenerationConfig::new().response_logprobs(true));
    assert_eq!(body["generationConfig"]["responseLogprobs"], json!(true));
}

#[test]
fn logprobs_lands_in_generation_config() {
    let body = generate(GenerationConfig::new().logprobs(3));
    assert_eq!(body["generationConfig"]["logprobs"], json!(3));
}

#[test]
fn candidate_count_is_one() {
    let body = generate(GenerationConfig::new().candidate_count(CandidateCount::One));
    assert_eq!(body["generationConfig"]["candidateCount"], json!(1));
}

#[test]
fn response_modalities_land_in_generation_config() {
    let body = generate(
        GenerationConfig::new()
            .response_modalities([ResponseModality::Text, ResponseModality::Image]),
    );
    assert_eq!(
        body["generationConfig"]["responseModalities"],
        json!(["TEXT", "IMAGE"])
    );
}

#[test]
fn image_config_lands_in_generation_config() {
    let body = generate(
        GenerationConfig::new()
            .image_config(ImageConfig::new().aspect_ratio("16:9").image_size("2K")),
    );
    assert_eq!(
        body["generationConfig"]["imageConfig"],
        json!({"aspectRatio": "16:9", "imageSize": "2K"})
    );
}

#[test]
fn speech_config_lands_in_generation_config() {
    let body = generate(
        GenerationConfig::new().speech_config(SpeechConfig::voice("Kore").language_code("en-US")),
    );
    assert_eq!(
        body["generationConfig"]["speechConfig"],
        json!({
            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}},
            "languageCode": "en-US"
        })
    );
    let body = generate(
        GenerationConfig::new()
            .speech_config(SpeechConfig::speakers([("Joe", "Kore"), ("Jane", "Puck")])),
    );
    assert_eq!(
        body["generationConfig"]["speechConfig"],
        json!({"multiSpeakerVoiceConfig": {"speakerVoiceConfigs": [
            {"speaker": "Joe", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}},
            {"speaker": "Jane", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}}
        ]}})
    );
}

#[test]
fn media_resolution_lands_in_generation_config() {
    let body = generate(GenerationConfig::new().media_resolution(MediaResolution::Low));
    assert_eq!(
        body["generationConfig"]["mediaResolution"],
        json!("MEDIA_RESOLUTION_LOW")
    );
}

#[test]
fn enable_enhanced_civic_answers_lands_in_generation_config() {
    let options = GeminiOptions::new()
        .generate_content(GenerateContentOptions::new().enable_enhanced_civic_answers(true));
    let body = sent(&rest(), with(&options)).expect("the body builds");
    assert_eq!(
        body["generationConfig"]["enableEnhancedCivicAnswers"],
        json!(true)
    );
}

#[test]
fn generation_config_merges_with_the_request_fields() {
    let options = GeminiOptions::new().generate_content(
        GenerateContentOptions::new().generation_config(GenerationConfig::new().top_k(40)),
    );
    let body = sent(&rest(), with(&options).temperature(0.0)).expect("the body builds");
    assert_eq!(
        body["generationConfig"],
        json!({"temperature": 0.0, "topK": 40})
    );
}

#[test]
fn safety_settings_replace_the_null() {
    let body = sent(&rest(), CompletionRequest::new("hi")).expect("the body builds");
    assert_eq!(body["safetySettings"], Value::Null);
    let options = GeminiOptions::new().generate_content(
        GenerateContentOptions::new()
            .safety_setting(HarmCategory::Harassment, HarmBlockThreshold::BlockOnlyHigh)
            .safety_setting(HarmCategory::Jailbreak, HarmBlockThreshold::Off),
    );
    let body = sent(&rest(), with(&options)).expect("the body builds");
    assert_eq!(
        body["safetySettings"],
        json!([
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_JAILBREAK", "threshold": "OFF"}
        ])
    );
}

#[test]
fn agent_replaces_model() {
    let body = interact(InteractionsOptions::new().agent("deep-research-pro-preview-12-2025"));
    assert_eq!(body["agent"], json!("deep-research-pro-preview-12-2025"));
    assert!(body.get("model").is_none(), "{body}");
}

#[test]
fn agent_config_lands_at_the_top() {
    let body = interact(
        InteractionsOptions::new().agent_config(AgentConfig::DeepResearch {
            thinking_summaries: Some(ThinkingSummaries::Auto),
        }),
    );
    assert_eq!(
        body["agent_config"],
        json!({"type": "deep-research", "thinking_summaries": "auto"})
    );
    let body = interact(InteractionsOptions::new().agent_config(AgentConfig::Dynamic));
    assert_eq!(body["agent_config"], json!({"type": "dynamic"}));
}

#[test]
fn background_lands_at_the_top() {
    let body = interact(InteractionsOptions::new().background(true));
    assert_eq!(body["background"], json!(true));
}

#[test]
fn previous_interaction_id_lands_and_continues_stored() {
    let options = GeminiOptions::new()
        .interactions(InteractionsOptions::new().previous_interaction_id("v1_abc"));
    let request = with(&options);
    assert!(interactions().continues_stored(&request));
    let body = sent(&interactions(), request).expect("the body builds");
    assert_eq!(body["previous_interaction_id"], json!("v1_abc"));
}

#[test]
fn safety_settings_land_snake_case() {
    let body = interact(InteractionsOptions::new().safety_setting(
        HarmCategory::HateSpeech,
        HarmBlockThreshold::BlockLowAndAbove,
    ));
    assert_eq!(
        body["safety_settings"],
        json!([{"type": "hate_speech", "threshold": "block_low_and_above"}])
    );
}

#[test]
fn thinking_summaries_land_in_generation_config() {
    let section = InteractionsOptions::new().thinking_summaries(ThinkingSummaries::None);
    let request = with(&GeminiOptions::new().interactions(section)).temperature(0.0);
    let body = sent(&interactions(), request).expect("the body builds");
    assert_eq!(
        body["generation_config"],
        json!({"temperature": 0.0, "thinking_summaries": "none"})
    );
}

#[test]
fn speech_lands_in_generation_config() {
    let body = interact(
        InteractionsOptions::new().speech(
            InteractionSpeech::voice("Kore")
                .language("en-US")
                .speaker("Joe"),
        ),
    );
    assert_eq!(
        body["generation_config"]["speech_config"],
        json!([{"voice": "Kore", "language": "en-US", "speaker": "Joe"}])
    );
}

/// Options that set every field of every section.
fn every_field() -> GeminiOptions {
    let config = GenerationConfig::new()
        .include_thoughts(true)
        .top_k(40)
        .presence_penalty(0.5)
        .frequency_penalty(0.5)
        .response_logprobs(true)
        .logprobs(2)
        .candidate_count(CandidateCount::One)
        .response_modalities([ResponseModality::Text])
        .image_config(ImageConfig::new().aspect_ratio("1:1"))
        .speech_config(SpeechConfig::voice("Kore"))
        .media_resolution(MediaResolution::High);
    GeminiOptions::new()
        .store(true)
        .label("team", "rig")
        .generate_content(
            GenerateContentOptions::new()
                .generation_config(config)
                .enable_enhanced_civic_answers(true)
                .safety_setting(HarmCategory::Harassment, HarmBlockThreshold::BlockNone),
        )
        .interactions(
            InteractionsOptions::new()
                .agent("deep-research-pro-preview-12-2025")
                .agent_config(AgentConfig::Dynamic)
                .background(true)
                .previous_interaction_id("v1_abc")
                .safety_setting(HarmCategory::Harassment, HarmBlockThreshold::BlockNone)
                .thinking_summaries(ThinkingSummaries::Auto)
                .speech(InteractionSpeech::voice("Kore")),
        )
}

#[test]
fn generate_content_fields_never_reach_interactions() {
    let options = GeminiOptions::new().generate_content(
        GenerateContentOptions::new()
            .generation_config(GenerationConfig::new().top_k(40))
            .safety_setting(HarmCategory::Harassment, HarmBlockThreshold::BlockNone),
    );
    let with_options = sent(&interactions(), with(&options)).expect("the body builds");
    let without = sent(&interactions(), CompletionRequest::new("hi")).expect("the body builds");
    assert_eq!(with_options, without);
}

#[test]
fn interactions_fields_never_reach_generate_content() {
    let options = GeminiOptions::new().interactions(
        InteractionsOptions::new()
            .background(true)
            .thinking_summaries(ThinkingSummaries::Auto),
    );
    let with_options = sent(&rest(), with(&options)).expect("the body builds");
    let without = sent(&rest(), CompletionRequest::new("hi")).expect("the body builds");
    assert_eq!(with_options, without);
}

#[test]
fn a_resumed_interaction_refuses_provider_options() {
    let wire = InteractionResume::new(GeminiConfig::new("k"), "v1_abc");
    let error = sent(&wire, with(&GeminiOptions::new().store(true)))
        .expect_err("a resumed interaction sends no body");
    assert!(error.to_string().contains("provider options"), "{error}");
}

#[test]
fn raw_generation_config_beats_typed() {
    let options = GeminiOptions::new().generate_content(
        GenerateContentOptions::new()
            .generation_config(GenerationConfig::new().top_k(40).presence_penalty(0.5)),
    );
    let request = with(&options).additional_params(json!({"generationConfig": {"topK": 3}}));
    let body = sent(&rest(), request).expect("the body builds");
    assert_eq!(body["generationConfig"]["topK"], json!(3));
    assert_eq!(body["generationConfig"]["presencePenalty"], json!(0.5));
}

#[test]
fn unset_options_leave_the_body_alone() {
    let request = CompletionRequest::new("hi").provider_options(
        ProviderOptions::new()
            .with::<GeminiExt>(&GeminiOptions::new())
            .expect("sections"),
    );
    assert!(request.provider_options.is_empty());
}

/// Every JSON pointer to a non-null leaf of `value`; an array is a leaf.
fn leaves(value: &Value, at: String, into: &mut Vec<String>) {
    match value {
        Value::Object(fields) => {
            for (key, field) in fields {
                leaves(field, format!("{at}/{key}"), into);
            }
        }
        Value::Null => {}
        _ => into.push(at),
    }
}

/// Whether leaf pointers `a` and `b` name the same leaf, or one lies under
/// the other.
fn overlap(a: &str, b: &str) -> bool {
    a == b || a.starts_with(&format!("{b}/")) || b.starts_with(&format!("{a}/"))
}

/// The leaves `target`'s body gets from a request setting every field the
/// request and its generation options own.
fn reserved<W>(wire: &W) -> Vec<String>
where
    W: Wire<Op = Completion, Payload = crate::wire::Encoded>,
{
    let tool = ToolDefinition::new(
        ToolName::new("add").expect("tool name"),
        "Add",
        json!({"type": "object"}),
    );
    let options = GenerationOptions::default()
        .reasoning(Reasoning::Budget { tokens: 1024 })
        .service_tier(ServiceTier::Priority)
        .top_p(0.9)
        .seed(7)
        .stop(["x"])
        .on_unsupported(OnUnsupported::Ignore);
    let request = CompletionRequest::new("hi")
        .preamble("be brief")
        .temperature(0.5)
        .max_tokens(100)
        .tool(tool)
        .tool_choice(ToolChoice::Auto)
        .output_schema(schemars::json_schema!({"type": "object"}))
        .options(options);
    let body = sent(wire, request).expect("the body builds");
    let mut reserved = Vec::new();
    leaves(&body, String::new(), &mut reserved);
    reserved
}

/// The leaves a route gets from `options`: the shared section, then its
/// own.
fn provider_leaves(options: &GeminiOptions, api: &str) -> Vec<String> {
    let value = serde_json::to_value(options).expect("options serialize");
    let mut merged = Map::new();
    for section in ["*", api] {
        if let Some(Value::Object(fields)) = value.get(section) {
            merged.extend(fields.clone());
        }
    }
    let mut leaves_of = Vec::new();
    leaves(&Value::Object(merged), String::new(), &mut leaves_of);
    leaves_of
}

#[test]
fn no_option_writes_a_reserved_leaf() {
    let options = every_field();
    for (api, reserved) in [
        (GENERATE_CONTENT, reserved(&rest())),
        (INTERACTIONS, reserved(&interactions())),
    ] {
        let provider = provider_leaves(&options, api);
        assert!(provider.len() > 3, "{api}: {provider:?}");
        for leaf in &provider {
            let clash = reserved.iter().find(|owned| overlap(leaf, owned));
            assert!(
                clash.is_none(),
                "{api}: {leaf} writes the reserved {clash:?}"
            );
        }
    }
}

#[test]
fn extras_read_a_generate_content_reply() {
    let raw = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "hi"}]},
            "finishReason": "STOP",
            "finishMessage": "done",
            "safetyRatings": [{"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"}],
            "citationMetadata": {"citationSources": [{"startIndex": 1, "endIndex": 5, "uri": "https://a"}]},
            "groundingMetadata": {
                "webSearchQueries": ["q"],
                "groundingChunks": [{"web": {"uri": "https://b", "title": "B"}}],
                "groundingSupports": [{"segment": {"startIndex": 0, "endIndex": 2, "text": "hi"}, "groundingChunkIndices": [0], "confidenceScores": [0.9]}]
            },
            "urlContextMetadata": {"urlMetadata": [{"retrievedUrl": "https://c", "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"}]},
            "avgLogprobs": -0.5,
            "logprobsResult": {"chosenCandidates": [{"token": "hi", "logProbability": -0.5}]}
        }],
        "promptFeedback": {"blockReason": "SAFETY"},
        "modelVersion": "gemini-2.5-flash",
        "responseId": "r1",
        "usageMetadata": {"serviceTier": "standard", "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 3}]}
    });
    let extras = GeminiExtras::from_reply(&Api::from_static(GENERATE_CONTENT), &raw)
        .expect("a generateContent reply");
    assert_eq!(extras.model_version.as_deref(), Some("gemini-2.5-flash"));
    assert_eq!(extras.response_id.as_deref(), Some("r1"));
    assert_eq!(extras.service_tier.as_deref(), Some("standard"));
    assert_eq!(extras.finish_message.as_deref(), Some("done"));
    assert_eq!(extras.avg_logprobs, Some(-0.5));
    assert_eq!(
        extras
            .prompt_feedback
            .and_then(|feedback| feedback.block_reason)
            .as_deref(),
        Some("SAFETY")
    );
    let ratings = extras.safety_ratings.unwrap_or_default();
    assert_eq!(
        ratings.first().and_then(|r| r.probability.as_deref()),
        Some("NEGLIGIBLE")
    );
    let sources = extras
        .citation_metadata
        .and_then(|m| m.citation_sources)
        .unwrap_or_default();
    assert_eq!(sources.first().and_then(|s| s.end_index), Some(5));
    let grounding = extras.grounding_metadata.expect("grounding");
    assert_eq!(grounding.web_search_queries, Some(vec!["q".to_owned()]));
    let url = extras
        .url_context_metadata
        .and_then(|m| m.url_metadata)
        .unwrap_or_default();
    assert_eq!(
        url.first().and_then(|u| u.retrieved_url.as_deref()),
        Some("https://c")
    );
    let chosen = extras
        .logprobs_result
        .and_then(|r| r.chosen_candidates)
        .unwrap_or_default();
    assert_eq!(chosen.first().and_then(|c| c.token.as_deref()), Some("hi"));
    assert_eq!(extras.id, None);
}

#[test]
fn extras_refuse_another_api() {
    let result = GeminiExtras::from_reply(&Api::from_static("openai.chat"), &json!({}));
    assert!(result.is_err());
}

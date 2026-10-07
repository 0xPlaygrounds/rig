//! Each Vertex AI provider option reaches the SDK request, in a typed field
//! rather than the SDK's unknown fields. These are unit tests: there is no
//! Vertex AI recording, and the request side is deterministic encoding.

use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::{CompletionRequest, ProviderOptions};
use rig_core::providers::gemini::extension::{
    CandidateCount, ImageConfig, MediaResolution, ResponseModality, SpeechConfig,
};
use rig_core::wire::{Mode, Operation, Wire};
use serde_json::json;

use super::*;
use crate::completion::GenerateContent;

/// The SDK request the Vertex AI wire builds for `options`.
fn sent(options: &VertexOptions) -> vertexai::model::GenerateContentRequest {
    let request = CompletionRequest::new("hi").provider_options(
        ProviderOptions::new()
            .with::<VertexExt>(options)
            .expect("Vertex options are sections"),
    );
    let wire = GenerateContent::new("gemini-2.5-flash");
    let request = rig_core::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    wire.encode(request, Mode::Unary)
        .expect("the request encodes")
}

/// The SDK `generationConfig` for `config`.
fn config(config: GenerationConfig) -> vertexai::model::GenerationConfig {
    sent(&VertexOptions::new().generation_config(config))
        .generation_config
        .expect("a generation config")
}

#[test]
fn labels_reach_the_sdk_request() {
    let request = sent(&VertexOptions::new().label("team", "rig"));
    assert_eq!(request.labels.get("team").map(String::as_str), Some("rig"));
}

#[test]
fn model_armor_config_reaches_the_sdk_request() {
    let request = sent(
        &VertexOptions::new().model_armor(
            ModelArmorConfig::new()
                .prompt_template("projects/p/locations/l/templates/in")
                .response_template("projects/p/locations/l/templates/out"),
        ),
    );
    let armor = request.model_armor_config.expect("a Model Armor config");
    assert_eq!(
        armor.prompt_template_name,
        "projects/p/locations/l/templates/in"
    );
    assert_eq!(
        armor.response_template_name,
        "projects/p/locations/l/templates/out"
    );
}

#[test]
fn routing_config_reaches_generation_config() {
    let auto = sent(
        &VertexOptions::new().routing(RoutingConfig::Auto(ModelRoutingPreference::PrioritizeCost)),
    )
    .generation_config
    .and_then(|config| config.routing_config)
    .expect("a routing config");
    let mode = auto.auto_mode().expect("auto mode");
    assert_eq!(
        mode.model_routing_preference
            .as_ref()
            .and_then(|preference| preference.name()),
        Some("PRIORITIZE_COST")
    );
    let manual =
        sent(&VertexOptions::new().routing(RoutingConfig::Manual("gemini-2.5-pro".to_owned())))
            .generation_config
            .and_then(|config| config.routing_config)
            .expect("a routing config");
    let mode = manual.manual_mode().expect("manual mode");
    assert_eq!(mode.model_name.as_deref(), Some("gemini-2.5-pro"));
}

#[test]
fn audio_timestamp_reaches_generation_config() {
    let request = sent(&VertexOptions::new().audio_timestamp(true));
    assert_eq!(
        request
            .generation_config
            .and_then(|config| config.audio_timestamp),
        Some(true)
    );
}

#[test]
fn safety_setting_method_reaches_the_sdk_request() {
    let request = sent(
        &VertexOptions::new().safety_setting(
            VertexSafetySetting::new(HarmCategory::Jailbreak, HarmBlockThreshold::BlockOnlyHigh)
                .method(HarmBlockMethod::Probability),
        ),
    );
    let setting = request.safety_settings.first().expect("a safety setting");
    assert_eq!(setting.category.name(), Some("HARM_CATEGORY_JAILBREAK"));
    assert_eq!(setting.threshold.name(), Some("BLOCK_ONLY_HIGH"));
    assert_eq!(setting.method.name(), Some("PROBABILITY"));
}

#[test]
fn include_thoughts_reaches_generation_config() {
    let config = config(GenerationConfig::new().include_thoughts(true));
    assert_eq!(
        config
            .thinking_config
            .and_then(|thinking| thinking.include_thoughts),
        Some(true)
    );
}

#[test]
fn top_k_reaches_generation_config() {
    assert_eq!(config(GenerationConfig::new().top_k(40)).top_k, Some(40.0));
}

#[test]
fn presence_penalty_reaches_generation_config() {
    let config = config(GenerationConfig::new().presence_penalty(0.5));
    assert_eq!(config.presence_penalty, Some(0.5));
}

#[test]
fn frequency_penalty_reaches_generation_config() {
    let config = config(GenerationConfig::new().frequency_penalty(0.25));
    assert_eq!(config.frequency_penalty, Some(0.25));
}

#[test]
fn response_logprobs_reaches_generation_config() {
    let config = config(GenerationConfig::new().response_logprobs(true));
    assert_eq!(config.response_logprobs, Some(true));
}

#[test]
fn logprobs_reaches_generation_config() {
    assert_eq!(
        config(GenerationConfig::new().logprobs(3)).logprobs,
        Some(3)
    );
}

#[test]
fn candidate_count_reaches_generation_config() {
    let config = config(GenerationConfig::new().candidate_count(CandidateCount::One));
    assert_eq!(config.candidate_count, Some(1));
}

#[test]
fn response_modalities_reach_generation_config() {
    let config = config(
        GenerationConfig::new()
            .response_modalities([ResponseModality::Text, ResponseModality::Audio]),
    );
    let names: Vec<_> = config
        .response_modalities
        .iter()
        .map(|modality| modality.name())
        .collect();
    assert_eq!(names, [Some("TEXT"), Some("AUDIO")]);
}

#[test]
fn image_config_reaches_generation_config() {
    let config = config(
        GenerationConfig::new()
            .image_config(ImageConfig::new().aspect_ratio("16:9").image_size("2K")),
    );
    let image = config.image_config.expect("an image config");
    assert_eq!(image.aspect_ratio.as_deref(), Some("16:9"));
    assert_eq!(image.image_size.as_deref(), Some("2K"));
}

#[test]
fn speech_config_reaches_generation_config() {
    let config = config(
        GenerationConfig::new().speech_config(SpeechConfig::voice("Kore").language_code("en-US")),
    );
    let speech = config.speech_config.expect("a speech config");
    assert_eq!(speech.language_code, "en-US");
    let voice = speech
        .voice_config
        .and_then(|voice| voice.prebuilt_voice_config().cloned())
        .expect("a prebuilt voice");
    assert_eq!(voice.voice_name.as_deref(), Some("Kore"));
}

#[test]
fn media_resolution_reaches_generation_config() {
    let config = config(GenerationConfig::new().media_resolution(MediaResolution::Medium));
    assert_eq!(
        config
            .media_resolution
            .as_ref()
            .and_then(|resolution| resolution.name()),
        Some("MEDIA_RESOLUTION_MEDIUM")
    );
}

/// A Vertex AI reply, read through a fake transport as the wire reads it.
/// There is no Vertex AI recording: the GenerateContent fields are the
/// Gemini API's recorded reply in
/// `rig-cassette/fixtures/cassettes/gemini/raw_capture_matrix/raw_exposes_forced_function_call.yaml`,
/// and `createTime`, `trafficType` and the citation are added by hand.
#[test]
fn extras_read_a_vertex_reply() {
    use rig_core::Model;
    use rig_core::driver::{Exchange, Opened, Opening, Transport};

    #[derive(Clone)]
    struct Stored(vertexai::model::GenerateContentResponse);

    impl Transport<GenerateContent> for Stored {
        fn send(
            &self,
            _payload: vertexai::model::GenerateContentRequest,
            _exchange: Exchange,
        ) -> Opening<vertexai::model::GenerateContentResponse> {
            Opening::ready(Opened::new(futures::stream::iter([Ok(self.0.clone())])))
        }
    }

    let reply: vertexai::model::GenerateContentResponse = serde_json::from_value(json!({
        "candidates": [{
            "content": {"parts": [{"functionCall": {"args": {"x": 2, "y": 3}, "name": "add"}}], "role": "model"},
            "finishMessage": "Model generated function call(s).",
            "finishReason": "STOP",
            "index": 0,
            "citationMetadata": {"citations": [{
                "startIndex": 1, "endIndex": 9, "uri": "https://example.com", "title": "Example",
                "publicationDate": {"year": 2024, "month": 5, "day": 1}
            }]}
        }],
        "modelVersion": "gemini-2.5-flash-lite",
        "responseId": "Ag7BauSXOKisz7IPlZS6iAg",
        "createTime": "2026-10-06T12:00:00Z",
        "usageMetadata": {
            "candidatesTokenCount": 18,
            "promptTokenCount": 67,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 67}],
            "totalTokenCount": 85,
            "trafficType": "ON_DEMAND"
        }
    }))
    .expect("an SDK reply");
    let response = futures::executor::block_on(
        Model::new(GenerateContent::new("gemini-2.5-flash-lite"), Stored(reply))
            .call(CompletionRequest::new("hi")),
    )
    .expect("the reply decodes");
    let extras = response
        .extras::<VertexExt>()
        .expect("a Vertex AI reply has Vertex extras")
        .expect("the reply holds the extras' shape");
    assert_eq!(
        extras.model_version.as_deref(),
        Some("gemini-2.5-flash-lite")
    );
    assert_eq!(
        extras.response_id.as_deref(),
        Some("Ag7BauSXOKisz7IPlZS6iAg")
    );
    assert_eq!(
        extras.finish_message.as_deref(),
        Some("Model generated function call(s).")
    );
    assert_eq!(extras.traffic_type.as_deref(), Some("ON_DEMAND"));
    assert_eq!(extras.create_time.as_deref(), Some("2026-10-06T12:00:00Z"));
    let detail = extras
        .prompt_tokens_details
        .unwrap_or_default()
        .into_iter()
        .next()
        .expect("a prompt token detail");
    assert_eq!(detail.modality.as_deref(), Some("TEXT"));
    assert_eq!(detail.token_count, Some(67));
    let citation = extras
        .citations
        .unwrap_or_default()
        .into_iter()
        .next()
        .expect("a citation");
    assert_eq!(citation.uri.as_deref(), Some("https://example.com"));
    assert_eq!(citation.end_index, Some(9));
    assert_eq!(
        citation.publication_date.and_then(|date| date.year),
        Some(2024)
    );
    assert!(
        response
            .extras::<rig_core::providers::gemini::extension::GeminiExt>()
            .is_none(),
        "a Vertex AI reply is not the Gemini API's"
    );
}

//! Each Gemini option the proto declares reaches the gRPC request, and the
//! ones it does not declare are refused by name. These are unit tests:
//! there is no gRPC recording, and the request side is deterministic
//! encoding.

use rig_core::completion::{CompletionRequest, GenerationOptions, OnUnsupported, ProviderOptions};
use rig_core::error::ProviderError;
use rig_core::operation::Completion;
use rig_core::providers::gemini::extension::{
    CandidateCount, GenerateContentOptions, GenerationConfig, HarmBlockThreshold, ImageConfig,
    InteractionsOptions, MediaResolution, ResponseModality, SpeechConfig,
};
use rig_core::wire::{Mode, Operation, Wire};

use super::*;
use crate::completion::GenerateContent;
use crate::proto;

/// The gRPC request for `options` under `policy`.
fn encoded(
    options: GeminiOptions,
    policy: OnUnsupported,
) -> Result<proto::GenerateContentRequest, ProviderError> {
    let request = CompletionRequest::new("hi")
        .options(GenerationOptions::default().on_unsupported(policy))
        .provider_options(
            ProviderOptions::new()
                .with::<GeminiGrpcExt>(&GeminiGrpcOptions::from(options))
                .expect("Gemini options are sections"),
        );
    let wire = GenerateContent::new("gemini-2.5-flash");
    let request = Completion::prepare(request, &wire.describe())?;
    Ok(wire.encode(request, Mode::Unary)?)
}

/// The gRPC `generation_config` for `config`.
fn config(config: GenerationConfig) -> proto::GenerationConfig {
    let options = GeminiOptions::new()
        .generate_content(GenerateContentOptions::new().generation_config(config));
    encoded(options, OnUnsupported::Error)
        .expect("the request encodes")
        .generation_config
        .expect("a generation config")
}

#[test]
fn top_k_reaches_generation_config() {
    assert_eq!(config(GenerationConfig::new().top_k(40)).top_k, Some(40));
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
            .response_modalities([ResponseModality::Text, ResponseModality::Image]),
    );
    assert_eq!(
        config.response_modalities,
        [
            proto::generation_config::Modality::Text as i32,
            proto::generation_config::Modality::Image as i32
        ]
    );
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
    let Some(proto::voice_config::VoiceConfig::PrebuiltVoiceConfig(voice)) =
        speech.voice_config.and_then(|voice| voice.voice_config)
    else {
        panic!("a prebuilt voice");
    };
    assert_eq!(voice.voice_name.as_deref(), Some("Kore"));
}

#[test]
fn media_resolution_reaches_generation_config() {
    let config = config(GenerationConfig::new().media_resolution(MediaResolution::High));
    assert_eq!(
        config.media_resolution,
        Some(proto::generation_config::MediaResolution::High as i32)
    );
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
fn enable_enhanced_civic_answers_reaches_generation_config() {
    let options = GeminiOptions::new()
        .generate_content(GenerateContentOptions::new().enable_enhanced_civic_answers(true));
    let request = encoded(options, OnUnsupported::Error).expect("the request encodes");
    assert_eq!(
        request
            .generation_config
            .and_then(|config| config.enable_enhanced_civic_answers),
        Some(true)
    );
}

#[test]
fn safety_settings_reach_the_request() {
    let options = GeminiOptions::new().generate_content(
        GenerateContentOptions::new()
            .safety_setting(HarmCategory::Harassment, HarmBlockThreshold::Off),
    );
    let request = encoded(options, OnUnsupported::Error).expect("the request encodes");
    let setting = request.safety_settings.first().expect("a safety setting");
    assert_eq!(setting.category, proto::HarmCategory::Harassment as i32);
    assert_eq!(
        setting.threshold,
        proto::safety_setting::HarmBlockThreshold::Off as i32
    );
}

#[test]
fn interactions_fields_never_reach_the_request() {
    let options = GeminiOptions::new().interactions(InteractionsOptions::new().background(true));
    let with = encoded(options, OnUnsupported::Error).expect("the request encodes");
    let without = encoded(GeminiOptions::new(), OnUnsupported::Error).expect("the request encodes");
    assert_eq!(with, without);
}

/// The option name a refusal reports.
fn refused(result: Result<proto::GenerateContentRequest, ProviderError>) -> String {
    match result {
        Err(ProviderError::UnsupportedOption(option)) => option.option.into_owned(),
        other => panic!("an unsupported option, got {other:?}"),
    }
}

#[test]
fn store_and_labels_are_unsupported() {
    let store = encoded(GeminiOptions::new().store(true), OnUnsupported::Error);
    assert_eq!(refused(store), "gemini-grpc.*.store");
    let labels = encoded(
        GeminiOptions::new().label("team", "rig"),
        OnUnsupported::Error,
    );
    assert_eq!(refused(labels), "gemini-grpc.*.labels");
    let jailbreak = GeminiOptions::new().generate_content(
        GenerateContentOptions::new()
            .safety_setting(HarmCategory::Jailbreak, HarmBlockThreshold::BlockNone),
    );
    assert_eq!(
        refused(encoded(jailbreak, OnUnsupported::Error)),
        "gemini-grpc.gemini.generate_content.safetySettings"
    );
    let ignored = GeminiOptions::new()
        .store(true)
        .label("team", "rig")
        .generate_content(
            GenerateContentOptions::new().generation_config(GenerationConfig::new().top_k(40)),
        );
    let request = encoded(ignored, OnUnsupported::Ignore).expect("the refused fields are left out");
    assert_eq!(
        request.generation_config.and_then(|config| config.top_k),
        Some(40)
    );
}

/// A gRPC reply, read through a fake transport as the wire reads it. There
/// is no gRPC recording: this is the Gemini API's recorded reply in
/// `rig-cassette/fixtures/cassettes/gemini/raw_capture_matrix/raw_exposes_forced_function_call.yaml`
/// without `usageMetadata.serviceTier`, which the proto does not declare.
#[test]
fn extras_read_a_grpc_reply() {
    use rig_core::Model;
    use rig_core::driver::{Exchange, Opened, Opening, Transport};

    #[derive(Clone)]
    struct Stored(proto::GenerateContentResponse);

    impl Transport<GenerateContent> for Stored {
        fn send(
            &self,
            _request: proto::GenerateContentRequest,
            _exchange: Exchange,
        ) -> Opening<proto::GenerateContentResponse> {
            Opening::ready(Opened::new(futures::stream::iter([Ok(self.0.clone())])))
        }
    }

    let reply: proto::GenerateContentResponse = crate::rest::from_rest(serde_json::json!({
        "candidates": [{
            "content": {"parts": [{"functionCall": {"args": {"x": 2, "y": 3}, "name": "add"}}], "role": "model"},
            "finishMessage": "Model generated function call(s).",
            "finishReason": "STOP",
            "index": 0
        }],
        "modelVersion": "gemini-2.5-flash-lite",
        "responseId": "Ag7BauSXOKisz7IPlZS6iAg",
        "usageMetadata": {
            "candidatesTokenCount": 18,
            "promptTokenCount": 67,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 67}],
            "totalTokenCount": 85
        }
    }))
    .expect("a proto reply");
    let response = futures::executor::block_on(
        Model::new(GenerateContent::new("gemini-2.5-flash-lite"), Stored(reply))
            .call(CompletionRequest::new("hi")),
    )
    .expect("the reply decodes");
    let extras = response
        .extras::<GeminiGrpcExt>()
        .expect("a gRPC reply has gRPC extras")
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
    let detail = extras
        .prompt_tokens_details
        .unwrap_or_default()
        .into_iter()
        .next()
        .expect("a prompt token detail");
    assert_eq!(detail.modality.as_deref(), Some("TEXT"));
    assert_eq!(detail.token_count, Some(67));
    assert!(
        response
            .extras::<rig_core::providers::gemini::extension::GeminiExt>()
            .is_none(),
        "a gRPC reply is not the Gemini API's"
    );
}

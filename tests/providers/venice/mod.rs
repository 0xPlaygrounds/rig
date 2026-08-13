mod support;

/// Small, cheap, reasoning- and vision-capable Venice text model.
const DEFAULT_MODEL: &str = rig::providers::venice::QWEN3_5_9B;
/// Tool and structured-output scenarios run on a fast *non-reasoning* model.
///
/// Venice's own `function_calling_default` (`zai-org-glm-4.7`) is a reasoning
/// model, and multi-turn tool scenarios recorded against it hit Venice's
/// non-streaming gateway timeout (504: "use the streaming API") or capacity
/// 429s while it thought. The model below answers the same scenarios in
/// seconds.
const TOOL_MODEL: &str = rig::providers::venice::MISTRAL_SMALL_3_2_24B;

mod cassette {
    mod agent;
    #[cfg(feature = "audio")]
    mod audio_generation;
    mod conformance;
    mod embeddings;
    mod error_envelope;
    mod extractor;
    #[cfg(feature = "image")]
    mod image_generation;
    mod model_listing;
    mod request_hook;
    mod streaming;
    mod streaming_tools;
    mod structured_output;
    mod tools;
    mod transcription;
    mod typed_prompt_tools;
    mod venice_parameters;
}

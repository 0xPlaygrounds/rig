use crate::providers::internal::audio_generation::{
    GenericAudioGenerationModel, RawAudioGenerationProvider,
};
use crate::providers::openai::{OpenAICompletionsExt, OpenAIResponsesExt};

pub const TTS_1: &str = "tts-1";
pub const TTS_1_HD: &str = "tts-1-hd";

/// OpenAI audio generation model.
pub type AudioGenerationModel<T> = GenericAudioGenerationModel<OpenAIResponsesExt, T>;

/// OpenAI audio generation model for a client using Chat Completions.
pub type CompletionsAudioGenerationModel<T> = GenericAudioGenerationModel<OpenAICompletionsExt, T>;

impl RawAudioGenerationProvider for OpenAIResponsesExt {
    const AUDIO_GENERATION_PATH: &'static str = "/audio/speech";
    const PROVIDER_NAME: &'static str = "openai";
    const REQUEST_ID_HEADER: Option<&'static str> = Some("x-request-id");
}

impl RawAudioGenerationProvider for OpenAICompletionsExt {
    const AUDIO_GENERATION_PATH: &'static str = "/audio/speech";
    const PROVIDER_NAME: &'static str = "openai";
    const REQUEST_ID_HEADER: Option<&'static str> = Some("x-request-id");
}

#[cfg(test)]
mod tests;

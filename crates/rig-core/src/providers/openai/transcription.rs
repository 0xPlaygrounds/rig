//! OpenAI transcription models.
use super::{Client, CompletionsClient};
use crate::http_client::HttpClientExt;
use crate::providers::internal::transcription::OpenAiTranscriptionClient;
pub use crate::providers::internal::transcription::{
    DurationTag, TokensTag, TranscriptionInputTokenDetails, TranscriptionResponse,
    TranscriptionUsage,
};
/// Whisper transcription model.
pub const WHISPER_1: &str = "whisper-1";

/// OpenAI transcription model using the shared OpenAI-style implementation.
pub type TranscriptionModel<T = crate::http_client::BoxedHttpClient> =
    crate::providers::internal::transcription::OpenAiTranscriptionModel<Client<T>>;

/// OpenAI transcription model for a client using Chat Completions.
pub type CompletionsTranscriptionModel<T = crate::http_client::BoxedHttpClient> =
    crate::providers::internal::transcription::OpenAiTranscriptionModel<CompletionsClient<T>>;

impl<T> OpenAiTranscriptionClient for Client<T>
where
    T: HttpClientExt + Clone + 'static,
{
    const MODEL_IN_FORM: bool = true;
    const PROVIDER_NAME: &'static str = "openai";
    const REQUEST_ID_HEADER: Option<&'static str> = Some("x-request-id");

    fn transcription_request(
        &self,
        _model: &str,
    ) -> crate::http_client::Result<crate::http_client::Builder> {
        self.post("/audio/transcriptions")
    }
}

impl<T> OpenAiTranscriptionClient for CompletionsClient<T>
where
    T: HttpClientExt + Clone + 'static,
{
    const MODEL_IN_FORM: bool = true;
    const PROVIDER_NAME: &'static str = "openai";
    const REQUEST_ID_HEADER: Option<&'static str> = Some("x-request-id");

    fn transcription_request(
        &self,
        _model: &str,
    ) -> crate::http_client::Result<crate::http_client::Builder> {
        self.post("/audio/transcriptions")
    }
}

#[cfg(all(test, feature = "openai"))]
mod tests;

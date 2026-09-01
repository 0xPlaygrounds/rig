//! Venice speech-to-text (`POST /audio/transcriptions`).
//!
//! Venice accepts the OpenAI multipart body (`file` plus `model`, `language`,
//! `prompt`, `temperature`) and answers with `{ "text": … }`, so the shared
//! OpenAI-style transcription model drives it unchanged.

use crate::http_client::HttpClientExt;
use crate::providers::internal::transcription::OpenAiTranscriptionClient;

use super::client::Client;

// ================================================================
// Venice Transcription API
// ================================================================
/// `openai/whisper-large-v3`
pub const WHISPER_LARGE_V3: &str = "openai/whisper-large-v3";
/// `nvidia/parakeet-tdt-0.6b-v3`
pub const PARAKEET_TDT_0_6B_V3: &str = "nvidia/parakeet-tdt-0.6b-v3";
/// `elevenlabs/scribe-v2`
pub const SCRIBE_V2: &str = "elevenlabs/scribe-v2";
/// `fal-ai/wizper`
pub const WIZPER: &str = "fal-ai/wizper";

/// Venice transcription model using the shared OpenAI-style implementation.
pub type TranscriptionModel<T> =
    crate::providers::internal::transcription::OpenAiTranscriptionModel<Client<T>>;

impl<T> OpenAiTranscriptionClient for Client<T>
where
    T: HttpClientExt + Clone + 'static,
{
    const MODEL_IN_FORM: bool = true;
    const PROVIDER_NAME: &'static str = "venice";
    const REQUEST_ID_HEADER: Option<&'static str> = None;

    fn transcription_request(
        &self,
        _model: &str,
    ) -> crate::http_client::Result<crate::http_client::Builder> {
        self.post("/audio/transcriptions")
    }
}

#[cfg(test)]
mod tests;

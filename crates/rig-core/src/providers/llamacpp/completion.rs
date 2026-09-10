//! llama.cpp completion models.
//!
//! Completions run through the shared OpenAI-compatible
//! [`GenericCompletionModel`](openai::completion::GenericCompletionModel); the
//! dialect is declared by the `OpenAICompatibleProvider` impl on
//! [`Llamacpp`](super::client::Llamacpp) in `client.rs`.

use crate::providers::openai;
use serde::{Deserialize, Serialize};

// ================================================================
// llama.cpp Completion Models
// ================================================================
/// The model identifier `llamafile` reported, kept as a convenience constant.
///
/// `llama-server` **ignores the request's `model` field entirely** — it serves
/// whichever GGUF it was started with and echoes that file's path back in the
/// response — so any string works here and none of them selects anything.
/// Measured on b10499-6d05498: a request naming a model the server has never
/// heard of returns 200 with a normal completion, not a 404. The multi-model
/// router (`llama-server --models-dir`) is the exception, and there the
/// identifier is whatever `GET /v1/models` lists.
pub const LLAMA_CPP: &str = "LLaMA_CPP";

/// llama.cpp completion model, driven by the shared OpenAI Chat Completions
/// path.
pub type CompletionModel<H = crate::http_client::BoxedHttpClient> =
    openai::completion::GenericCompletionModel<super::client::Llamacpp, H>;

/// Server-side timing accounting `llama-server` reports beside `usage`.
///
/// Not an OpenAI field. It is the only latency accounting a llama.cpp caller
/// gets, and for local inference it is the number people actually watch —
/// tokens per second is the difference between a usable model and an unusable
/// one on a given machine, and nothing in rig's normalized
/// [`Usage`](crate::completion::Usage) has a home for it.
///
/// Every field is optional: the shape has changed across builds (a
/// `timings_per_token` mode adds more), and a missing key must degrade to
/// `None` rather than fail the whole response.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct Timings {
    /// Prompt tokens served from the KV cache. Agrees with
    /// `usage.prompt_tokens_details.cached_tokens`, which rig already
    /// normalizes; kept because this is the field llama.cpp's own tooling
    /// reads and the two are independently populated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_n: Option<u64>,
    /// Prompt tokens actually evaluated this turn (total minus `cache_n`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_n: Option<u64>,
    /// Wall-clock milliseconds spent evaluating the prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_ms: Option<f64>,
    /// `prompt_ms / prompt_n`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_per_token_ms: Option<f64>,
    /// Prompt-evaluation throughput.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_per_second: Option<f64>,
    /// Tokens generated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_n: Option<u64>,
    /// Wall-clock milliseconds spent generating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_ms: Option<f64>,
    /// `predicted_ms / predicted_n`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_per_token_ms: Option<f64>,
    /// Generation throughput — the tokens-per-second figure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_per_second: Option<f64>,
}

/// `llama-server`'s chat-completions payload: OpenAI's, plus `timings`.
///
/// Rig's shared [`openai::CompletionResponse`] has no field for `timings` and
/// no catch-all, so a provider whose `Response` is that type drops it — while
/// the *streaming* path keeps it, because
/// `StreamingCompletionChunk` does carry a `#[serde(flatten)]` catch-all for
/// exactly this reason. Preserving it here removes that asymmetry for
/// llama.cpp and follows the precedent
/// [`deepseek`](crate::providers::deepseek) and [`mira`](crate::providers::mira)
/// set: a provider that adds fields to the wire declares its own response type
/// rather than pretending the OpenAI one describes it.
///
/// Everything OpenAI-shaped is reached through [`Self::openai`], and every
/// trait a `Response` must satisfy delegates to it, so this wrapper cannot
/// drift from the shared conversion.
///
/// What llama.cpp does **not** put on this wire is worth stating too:
/// `tokens_evaluated`, `tokens_cached`, `truncated`, `stop_type`,
/// `stopping_word` and `generation_settings` exist only on llama.cpp's
/// *native* `POST /completion`, which rig never calls, so they are absent
/// rather than dropped. `system_fingerprint` — which carries the build tag —
/// is already a named field on the OpenAI type.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    /// The OpenAI-compatible half of the payload.
    #[serde(flatten)]
    pub openai: openai::CompletionResponse,
    /// llama.cpp's server-side timing accounting, when the server reported it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}

impl CompletionResponse {
    /// Generation throughput in tokens per second, when the server reported it.
    pub fn predicted_tokens_per_second(&self) -> Option<f64> {
        self.timings
            .as_ref()
            .and_then(|timings| timings.predicted_per_second)
    }
}

impl crate::completion::NormalizeCompletionResponse for CompletionResponse {
    fn normalize(
        self,
        provider: &str,
    ) -> Result<crate::completion::CompletionResponse, crate::completion::CompletionError> {
        self.openai.normalize(provider)
    }
}

impl crate::telemetry::ProviderResponseExt for CompletionResponse {
    type Usage = openai::Usage;

    fn response_id(&self) -> Option<&str> {
        self.openai.response_id()
    }

    fn response_model_name(&self) -> Option<&str> {
        self.openai.response_model_name()
    }

    fn text_response(&self) -> Option<String> {
        self.openai.text_response()
    }

    fn usage(&self) -> Option<Self::Usage> {
        self.openai.usage()
    }
}

#[cfg(test)]
mod tests;

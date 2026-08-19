//! llama.cpp completion models.
//!
//! Completions run through the shared OpenAI-compatible
//! [`GenericCompletionModel`](openai::completion::GenericCompletionModel); the
//! dialect is declared by the `OpenAICompatibleProvider` impl on
//! [`LlamacppExt`](super::client::LlamacppExt) in `client.rs`.

use crate::providers::openai;
use crate::telemetry::ProviderResponseExt as _;
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
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<super::client::LlamacppExt, H>;

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

    fn get_response_id(&self) -> Option<String> {
        self.openai.get_response_id()
    }

    fn get_response_model_name(&self) -> Option<String> {
        self.openai.get_response_model_name()
    }

    fn get_text_response(&self) -> Option<String> {
        self.openai.get_text_response()
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        self.openai.get_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::NormalizeCompletionResponse as _;
    use crate::telemetry::ProviderResponseExt as _;

    /// A real b10499 chat-completions body, trimmed to the fields under test.
    const BODY: &str = r#"{
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": { "role": "assistant", "content": "ok" }
        }],
        "created": 0,
        "id": "chatcmpl-abc",
        "model": "/models/Qwen3-1.7B-Q4_K_M.gguf",
        "object": "chat.completion",
        "system_fingerprint": "b10499-6d05498",
        "usage": { "completion_tokens": 2, "prompt_tokens": 9, "total_tokens": 11,
                   "prompt_tokens_details": { "cached_tokens": 8 } },
        "timings": { "cache_n": 8, "prompt_n": 1, "prompt_ms": 6.175,
                     "predicted_n": 2, "predicted_ms": 16.9,
                     "predicted_per_second": 118.3 }
    }"#;

    #[test]
    fn timings_survive_deserialization_and_normalization() {
        let response: CompletionResponse =
            serde_json::from_str(BODY).expect("a llama.cpp body should deserialize");

        let timings = response
            .timings
            .clone()
            .expect("llama.cpp reports timings on every chat completion");
        assert_eq!(timings.cache_n, Some(8));
        assert_eq!(timings.predicted_n, Some(2));
        assert_eq!(response.predicted_tokens_per_second(), Some(118.3));

        // The OpenAI half is intact and normalizes exactly as before.
        assert_eq!(response.get_response_id().as_deref(), Some("chatcmpl-abc"));
        assert_eq!(
            response.openai.system_fingerprint.as_deref(),
            Some("b10499-6d05498")
        );
        let normalized = response
            .normalize("llamacpp")
            .expect("a llama.cpp body should normalize");
        assert_eq!(normalized.provider, "llamacpp");
        // `cache_n` and the normalized cached-token count are independently
        // populated and must agree.
        assert_eq!(normalized.usage.cached_input_tokens, 8);
    }

    /// A response with no `timings` is not an error.
    ///
    /// llama.cpp always sends them today, but a `.llamafile` built from an
    /// older core, or a proxy that strips unknown fields, must still decode.
    #[test]
    fn a_response_without_timings_still_decodes() {
        let mut body: serde_json::Value = serde_json::from_str(BODY).expect("fixture should parse");
        body.as_object_mut()
            .expect("body is an object")
            .remove("timings")
            .expect("the fixture carries timings to remove");

        let response: CompletionResponse = serde_json::from_value(body)
            .unwrap_or_else(|error| panic!("should decode without timings: {error}"));
        assert!(response.timings.is_none());
        assert!(response.predicted_tokens_per_second().is_none());
        assert_eq!(response.get_text_response().as_deref(), Some("ok"));
    }

    /// Round-tripping must not invent or lose the extra field.
    #[test]
    fn serialization_round_trips_the_extra_field() {
        let response: CompletionResponse = serde_json::from_str(BODY).expect("should deserialize");
        let value = serde_json::to_value(&response).expect("should serialize");
        assert_eq!(value["timings"]["cache_n"], serde_json::json!(8));
        assert_eq!(
            value["system_fingerprint"],
            serde_json::json!("b10499-6d05498"),
            "the flattened OpenAI half must stay at the top level, not nest under `openai`"
        );
        let again: CompletionResponse = serde_json::from_value(value).expect("should round-trip");
        assert_eq!(again.timings, response.timings);
    }
}

use crate::client::{self, BearerAuth, DebugExt, Provider};
use http::HeaderValue;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

// ================================================================
// Main openrouter Client
// ================================================================
const OPENROUTER_API_BASE_URL: &str = "https://openrouter.ai/api/v1";

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenRouterExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenRouterExtBuilder;

type OpenRouterApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<OpenRouterExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<OpenRouterExtBuilder, OpenRouterApiKey, H>;

impl Provider for OpenRouterExt {
    type Builder = OpenRouterExtBuilder;

    const VERIFY_PATH: &'static str = "/key";
}

client::impl_capabilities!(
    OpenRouterExt,
    completion = super::CompletionModel<H>,
    embeddings = super::EmbeddingModel<H>,
    transcription = super::transcription::TranscriptionModel<H>,
    model_listing = super::OpenRouterModelLister<H>,
    audio_generation = super::audio_generation::AudioGenerationModel<H>,
);

impl DebugExt for OpenRouterExt {}

client::impl_default_provider_builder!(
    OpenRouterExtBuilder => OpenRouterExt,
    api_key = OpenRouterApiKey,
    base_url = OPENROUTER_API_BASE_URL,
);

client::impl_provider_client!(
    Client,
    input = OpenRouterApiKey,
    api_key_env = "OPENROUTER_API_KEY",
);

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    #[serde(default)]
    pub completion_tokens: usize,
    pub total_tokens: usize,
    #[serde(default)]
    pub cost: f64,
    /// OpenAI-compatible prompt-token details, returned by OpenRouter when a
    /// provider reports cache activity (Anthropic with cache_control, OpenAI
    /// with server-side automatic caching).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// OpenAI-compatible completion-token breakdown. OpenRouter includes full
    /// usage accounting on every response, so a reasoning-capable route
    /// reports here how much of `completion_tokens` went to hidden reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Prompt-token breakdown reported by OpenRouter for cached requests.
// `usize` matches the parent `Usage` struct in this module; the streaming counterpart
// in `streaming.rs` uses `u32` to match its own parent.
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct PromptTokensDetails {
    /// Tokens served from cache (cache hit).
    #[serde(default)]
    pub cached_tokens: usize,
    /// Tokens written to cache on this call (cache miss that populated the cache).
    #[serde(default)]
    pub cache_write_tokens: usize,
}

/// Completion-token breakdown reported by OpenRouter.
///
/// Only the reasoning share is modeled: it is the one entry rig's normalized
/// [`crate::completion::Usage`] has a slot for, and OpenRouter documents usage
/// accounting as always present (on the final SSE message when streaming).
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct CompletionTokensDetails {
    /// Tokens the upstream spent on hidden reasoning, counted inside
    /// `completion_tokens`.
    #[serde(default)]
    pub reasoning_tokens: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} Total tokens: {}",
            self.prompt_tokens, self.total_tokens
        )
    }
}

impl From<&Usage> for crate::completion::Usage {
    fn from(value: &Usage) -> crate::completion::Usage {
        let (cached_input, cache_creation) = value
            .prompt_tokens_details
            .as_ref()
            .map(|d| (d.cached_tokens as u64, d.cache_write_tokens as u64))
            .unwrap_or((0, 0));
        crate::completion::Usage {
            input_tokens: value.prompt_tokens as u64,
            // Reported completion tokens, falling back to saturating
            // total - prompt for gateways that omit the field (it
            // deserializes to 0).
            output_tokens: if value.completion_tokens > 0 {
                value.completion_tokens as u64
            } else {
                value.total_tokens.saturating_sub(value.prompt_tokens) as u64
            },
            total_tokens: value.total_tokens as u64,
            cached_input_tokens: cached_input,
            cache_creation_input_tokens: cache_creation,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: value
                .completion_tokens_details
                .as_ref()
                .map(|d| d.reasoning_tokens as u64)
                .unwrap_or(0),
        }
    }
}

impl From<Usage> for crate::completion::Usage {
    fn from(value: Usage) -> crate::completion::Usage {
        crate::completion::Usage::from(&value)
    }
}
impl<ApiKey, H> client::ClientBuilder<OpenRouterExtBuilder, ApiKey, H> {
    /// Attach OpenRouter app-identification headers (`X-OpenRouter-Title` and `HTTP-Referer`)
    /// to every request made by this client. `title` appears in the dashboard activity feed
    /// and rankings page; `url` is the primary app identifier required to create an app page
    /// on OpenRouter. Invalid (non-ASCII) values are silently skipped.
    pub fn with_app_identity(mut self, title: impl AsRef<str>, url: impl AsRef<str>) -> Self {
        if let Ok(val) = HeaderValue::from_str(title.as_ref()) {
            self.headers_mut().insert(
                http::header::HeaderName::from_static("x-openrouter-title"),
                val,
            );
        }
        if let Ok(val) = HeaderValue::from_str(url.as_ref()) {
            self.headers_mut()
                .insert(http::header::HeaderName::from_static("http-referer"), val);
        }
        self
    }

    /// Assign this app to up to two OpenRouter marketplace categories via the
    /// `X-OpenRouter-Categories` header. Categories must be lowercase and hyphen-separated
    /// (e.g. `"cli-agent"`, `"ide-extension"`). OpenRouter silently ignores unrecognized
    /// categories. Extra categories beyond the first two are not sent. Invalid (non-ASCII)
    /// values are silently skipped.
    pub fn with_app_categories<S>(mut self, categories: &[S]) -> Self
    where
        S: AsRef<str>,
    {
        let joined = categories
            .iter()
            .take(2)
            .map(|c| c.as_ref())
            .collect::<Vec<_>>()
            .join(",");
        if !joined.is_empty()
            && let Ok(val) = HeaderValue::from_str(&joined)
        {
            self.headers_mut().insert(
                http::header::HeaderName::from_static("x-openrouter-categories"),
                val,
            );
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::Usage;

    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::openrouter::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::openrouter::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }

    #[test]
    fn test_with_app_identity_sets_headers() {
        let client = crate::providers::openrouter::Client::builder()
            .with_app_identity("My App", "https://myapp.example.com")
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");

        let headers = client.headers();
        assert_eq!(
            headers
                .get("x-openrouter-title")
                .and_then(|v| v.to_str().ok()),
            Some("My App"),
        );
        assert_eq!(
            headers.get("http-referer").and_then(|v| v.to_str().ok()),
            Some("https://myapp.example.com"),
        );
    }

    #[test]
    fn test_without_app_identity_no_extra_headers() {
        let client = crate::providers::openrouter::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");

        let headers = client.headers();
        assert!(headers.get("x-openrouter-title").is_none());
        assert!(headers.get("http-referer").is_none());
    }

    #[test]
    fn test_with_app_categories_sets_header() {
        let client = crate::providers::openrouter::Client::builder()
            .with_app_categories(&["cli-agent", "ide-extension"])
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");

        assert_eq!(
            client
                .headers()
                .get("x-openrouter-categories")
                .and_then(|v| v.to_str().ok()),
            Some("cli-agent,ide-extension"),
        );
    }

    #[test]
    fn test_with_app_categories_sends_at_most_two_categories() {
        let client = crate::providers::openrouter::Client::builder()
            .with_app_categories(&["cli-agent", "ide-extension", "chat"])
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");

        assert_eq!(
            client
                .headers()
                .get("x-openrouter-categories")
                .and_then(|v| v.to_str().ok()),
            Some("cli-agent,ide-extension"),
        );
    }

    #[test]
    fn test_with_app_categories_empty_list_no_header() {
        let empty: [&str; 0] = [];
        let client = crate::providers::openrouter::Client::builder()
            .with_app_categories(&empty)
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");

        assert!(client.headers().get("x-openrouter-categories").is_none());
    }

    #[test]
    fn test_without_app_categories_no_header() {
        let client = crate::providers::openrouter::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");

        assert!(client.headers().get("x-openrouter-categories").is_none());
    }

    /// A real usage object, copied verbatim out of
    /// `tests/cassettes/openrouter/reasoning_usage_matrix/blocking_anthropic_routed_reports_reasoning_tokens.yaml`:
    /// the breakdown must survive deserialization and reach the normalized
    /// `reasoning_tokens` slot, unmodeled siblings and all.
    #[test]
    fn completion_tokens_details_reaches_normalized_usage() {
        let usage: Usage = serde_json::from_str(
            r#"{"completion_tokens":540,
                "completion_tokens_details":{"audio_tokens":0,"image_tokens":0,"reasoning_tokens":531},
                "cost":0.002794,
                "cost_details":{"upstream_inference_completions_cost":0.0027,"upstream_inference_cost":0.002794,"upstream_inference_prompt_cost":0.000094},
                "is_byok":false,
                "prompt_tokens":94,
                "prompt_tokens_details":{"audio_tokens":0,"cache_write_tokens":0,"cached_tokens":0,"video_tokens":0},
                "total_tokens":634}"#,
        )
        .expect("recorded usage should deserialize");

        let normalized = crate::completion::Usage::from(&usage);
        assert_eq!(normalized.reasoning_tokens, 531);
        assert_eq!(normalized.output_tokens, 540);
        assert_eq!(normalized.input_tokens, 94);
        assert_eq!(normalized.total_tokens, 634);
        // The reasoning share is counted *inside* the completion tokens.
        assert!(normalized.reasoning_tokens <= normalized.output_tokens);
    }

    /// A non-reasoning route reports the object with a zero share; a gateway
    /// that omits it entirely, or sends it as `null`, must read the same.
    #[test]
    fn completion_tokens_details_absent_null_or_zero_all_read_zero() {
        for body in [
            r#"{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}"#,
            r#"{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":null}"#,
            r#"{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{"reasoning_tokens":0}}"#,
            r#"{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{}}"#,
        ] {
            let usage: Usage = serde_json::from_str(body).expect("usage should deserialize");
            let normalized = crate::completion::Usage::from(&usage);
            assert_eq!(normalized.reasoning_tokens, 0, "body: {body}");
            assert_eq!(normalized.output_tokens, 3, "body: {body}");
        }
    }

    /// Unknown siblings inside the breakdown (OpenRouter sends `audio_tokens`
    /// and `image_tokens`) must not fail the decode — the object is read for
    /// the one entry rig has a slot for.
    #[test]
    fn completion_tokens_details_tolerates_unmodeled_siblings() {
        let usage: Usage = serde_json::from_str(
            r#"{"prompt_tokens":9,"completion_tokens":1291,"total_tokens":1300,
                "completion_tokens_details":{"audio_tokens":0,"image_tokens":1290,"reasoning_tokens":7}}"#,
        )
        .expect("usage should deserialize");

        assert_eq!(crate::completion::Usage::from(&usage).reasoning_tokens, 7);
    }

    /// The completion-token fallback (`total - prompt` for gateways that omit
    /// `completion_tokens`) must stay independent of the new field.
    #[test]
    fn completion_tokens_details_does_not_disturb_the_output_token_fallback() {
        let usage: Usage = serde_json::from_str(
            r#"{"prompt_tokens":10,"total_tokens":30,
                "completion_tokens_details":{"reasoning_tokens":12}}"#,
        )
        .expect("usage should deserialize");

        let normalized = crate::completion::Usage::from(&usage);
        assert_eq!(normalized.output_tokens, 20);
        assert_eq!(normalized.reasoning_tokens, 12);
    }

    /// Round-tripping the type must not start sending a breakdown rig never
    /// received: the field is skipped when absent.
    #[test]
    fn completion_tokens_details_is_omitted_when_absent() {
        let usage: Usage =
            serde_json::from_str(r#"{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}"#)
                .expect("usage should deserialize");
        let encoded = serde_json::to_string(&usage).expect("usage should serialize");

        assert!(!encoded.contains("completion_tokens_details"), "{encoded}");
    }
}

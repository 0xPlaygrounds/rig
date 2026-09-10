//! Perplexity API client and Rig integration
//!
//! # Example
//! ```ignore
//! use rig_core::{client::CompletionClient, providers::perplexity};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = perplexity::Client::new("YOUR_API_KEY")?;
//!
//! let sonar = client.completion_model(perplexity::SONAR);
//! # Ok(())
//! # }
//! ```
use crate::client::BearerAuth;
use crate::client::{self, HasCompletion, ModelTransport, Provider, ProviderClientResult};
use crate::completion::CompletionError;
use crate::http_client::{self, HttpClientExt};
use crate::providers::openai;

// ================================================================
// Main Perplexity Client
// ================================================================
const PERPLEXITY_API_BASE_URL: &str = "https://api.perplexity.ai";

#[derive(Debug, Default, Clone, Copy)]
pub struct Perplexity;

type PerplexityApiKey = BearerAuth;

impl Provider for Perplexity {
    const NAME: &'static str = "perplexity";
    const BASE_URL: &'static str = PERPLEXITY_API_BASE_URL;
    // There is currently no way to verify a perplexity api key without consuming tokens
    const VERIFY_PATH: &'static str = "";
    type ApiKey = PerplexityApiKey;
    type Config = ();
    type EnvInput = String;

    fn build(_: (), _: &PerplexityApiKey) -> http_client::Result<Self> {
        Ok(Perplexity)
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<H>> {
        Client::from_env_api_key("PERPLEXITY_API_KEY", None, http)
    }

    fn from_val<H: HttpClientExt>(input: String, http: H) -> ProviderClientResult<Client<H>> {
        Client::new_with(input, http)
    }
}

impl HasCompletion for Perplexity {
    type Model<H>
        = CompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        CompletionModel::new(client.clone(), model)
    }
}

impl openai::completion::OpenAICompatibleProvider for Perplexity {
    const PROVIDER_NAME: &'static str = "perplexity";

    type StreamingUsage = openai::Usage;

    // Perplexity has no tool-calling support; `tools`/`tool_choice` are
    // dropped with a warning during request conversion.
    const SUPPORTS_TOOLS: bool = false;

    // Perplexity's structured-output support predates rig's `output_schema`
    // mapping; keep the pre-migration behavior of dropping it with a warning.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    // The pre-migration streaming request sent `stream: true` with no
    // `stream_options`.
    const STREAM_INCLUDE_USAGE: bool = false;

    type Response = openai::CompletionResponse;

    fn finalize_request_body(&self, body: &mut serde_json::Value) -> Result<(), CompletionError> {
        // Perplexity historically only accepted plain `{role, content: String}`
        // messages, and its API accepts only system/user/assistant roles
        // with strict user/assistant alternation. Strip tool-exchange
        // remnants from shared histories and flatten text-only content-part
        // arrays; arrays with non-text parts (e.g. images on sonar models)
        // are left for the API's multimodal handling.
        if let Some(messages) = body
            .get_mut("messages")
            .and_then(serde_json::Value::as_array_mut)
        {
            openai::completion::sanitize_plain_text_history(
                messages,
                Some(("\n", true)),
                false,
                true,
            );
        }

        Ok(())
    }
}

pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<Perplexity, H>;
pub type ClientBuilder<H = crate::markers::Missing> = client::ClientBuilder<Perplexity, H>;

/// Perplexity completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = crate::http_client::BoxedHttpClient> =
    openai::completion::GenericCompletionModel<Perplexity, H>;

/// Raw completion payload, shared with the OpenAI Chat Completions path.
pub type CompletionResponse = openai::CompletionResponse;

// ================================================================
// Perplexity Completion API
// ================================================================

pub const SONAR_PRO: &str = "sonar_pro";
pub const SONAR: &str = "sonar";

#[cfg(test)]
mod tests;

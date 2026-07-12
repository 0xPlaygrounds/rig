//! Llamafile API client and Rig integration
//!
//! [Llamafile](https://github.com/Mozilla-Ocho/llamafile) is a Mozilla Builders project
//! that distributes LLMs as single-file executables. When started, it exposes an
//! OpenAI-compatible API at `http://localhost:8080/v1`.
//!
//! # Example
//! ```rust,ignore
//! use rig_core::providers::llamafile;
//! use rig_core::completion::Prompt;
//!
//! // Create a new Llamafile client (defaults to http://localhost:8080)
//! let client = llamafile::Client::from_url("http://localhost:8080")?;
//!
//! // Create an agent with a preamble
//! let agent = client
//!     .agent(llamafile::LLAMA_CPP)
//!     .preamble("You are a helpful assistant.")
//!     .build();
//!
//! // Prompt the agent and print the response
//! let response = agent.prompt("Hello!").await?;
//! println!("{response}");
//! ```

use crate::client::{
    self, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder, ProviderClient,
    Transport,
};
use crate::http_client::{self, HttpClientExt};
use crate::providers::openai;

// ================================================================
// Main Llamafile Client
// ================================================================
const LLAMAFILE_API_BASE_URL: &str = "http://localhost:8080";

/// The default model identifier reported by llamafile.
pub const LLAMA_CPP: &str = "LLaMA_CPP";

#[derive(Debug, Default, Clone, Copy)]
pub struct LlamafileExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct LlamafileBuilder;

impl Provider for LlamafileExt {
    type Builder = LlamafileBuilder;
    const VERIFY_PATH: &'static str = "/models";

    // Llamafile clients are constructed from a bare host URL
    // (e.g. `http://localhost:8080`) while the shared OpenAI-compatible
    // endpoints are relative to `/v1`.
    fn build_uri(&self, base_url: &str, path: &str, _transport: Transport) -> String {
        let base_url = base_url.trim_end_matches('/');
        format!("{base_url}/v1/{}", path.trim_start_matches('/'))
    }
}

impl openai::completion::OpenAICompatibleProvider for LlamafileExt {
    const PROVIDER_NAME: &'static str = "llamafile";

    type StreamingUsage = openai::Usage;

    // llama.cpp-based servers can emit a whole tool call in one streaming chunk.
    const EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS: bool = true;

    type Response = openai::CompletionResponse;
}

// llama.cpp's `build_uri` injects the `/v1/` segment, so the default
// `/embeddings` path is correct.
impl openai::embedding::OpenAIEmbeddingsCompatible for LlamafileExt {}

impl<H> Capabilities<H> for LlamafileExt {
    type Completion = Capable<openai::completion::GenericCompletionModel<LlamafileExt, H>>;
    type Embeddings = Capable<openai::embedding::GenericEmbeddingModel<LlamafileExt, H>>;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl DebugExt for LlamafileExt {}

impl ProviderBuilder for LlamafileBuilder {
    type Extension<H>
        = LlamafileExt
    where
        H: HttpClientExt;
    type ApiKey = Nothing;

    const BASE_URL: &'static str = LLAMAFILE_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(LlamafileExt)
    }
}

pub type Client<H = reqwest::Client> = client::Client<LlamafileExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<LlamafileBuilder, Nothing, H>;

/// Llamafile completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<LlamafileExt, H>;

/// Llamafile embedding model, driven by the shared OpenAI embeddings path.
pub type EmbeddingModel<H = reqwest::Client> =
    openai::embedding::GenericEmbeddingModel<LlamafileExt, H>;

impl Client {
    /// Create a client pointing at the given llamafile base URL
    /// (e.g. `http://localhost:8080`).
    pub fn from_url(base_url: &str) -> crate::client::ProviderClientResult<Self> {
        Self::builder()
            .api_key(Nothing)
            .base_url(base_url)
            .build()
            .map_err(Into::into)
    }
}

impl ProviderClient for Client {
    type Input = Nothing;
    type Error = crate::client::ProviderClientError;

    fn from_env() -> Result<Self, Self::Error> {
        let api_base = crate::client::required_env_var("LLAMAFILE_API_BASE_URL")?;
        Self::from_url(&api_base)
    }

    fn from_val(_: Self::Input) -> Result<Self, Self::Error> {
        Self::builder().api_key(Nothing).build().map_err(Into::into)
    }
}

// ================================================================
// Tests
// ================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::Nothing;

    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::llamafile::Client::new(Nothing).expect("Client::new() failed");
        let _client_from_builder = crate::providers::llamafile::Client::builder()
            .api_key(Nothing)
            .build()
            .expect("Client::builder() failed");
    }

    #[test]
    fn test_client_from_url() {
        let _client = crate::providers::llamafile::Client::from_url("http://localhost:8080");
    }

    #[test]
    fn test_build_uri_routes_through_v1() {
        let ext = LlamafileExt;
        assert_eq!(
            ext.build_uri(
                "http://localhost:8080",
                "/chat/completions",
                Transport::Http
            ),
            "http://localhost:8080/v1/chat/completions"
        );
        assert_eq!(
            ext.build_uri("http://localhost:8080/", "/embeddings", Transport::Http),
            "http://localhost:8080/v1/embeddings"
        );
        assert_eq!(
            ext.build_uri(
                "http://localhost:8080",
                LlamafileExt::VERIFY_PATH,
                Transport::Http
            ),
            "http://localhost:8080/v1/models"
        );
    }
}

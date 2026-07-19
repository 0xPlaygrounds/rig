//! AWS Bedrock Mantle (OpenAI-compatible) client.
//!
//! Mantle exposes selected Bedrock models (including OpenAI GPT-OSS) through an
//! OpenAI-compatible HTTP API. Auth uses a short-term IAM bearer token
//! (`bedrock-api-key-…`) minted via SigV4, or a pre-supplied
//! [`AWS_BEARER_TOKEN_BEDROCK`] env value.
//!
//! This module reuses Rig's OpenAI Responses / Completions clients pointed at
//! the Mantle base URL. It does **not** change the Converse client path.
//!
//! # Example
//!
//! ```no_run
//! use rig_bedrock::mantle::{self, ClientBuilder, OPENAI_GPT_OSS_20B};
//! use rig_core::client::CompletionClient;
//! use rig_core::completion::Prompt;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = ClientBuilder::from_env().await?;
//! let agent = client.agent(OPENAI_GPT_OSS_20B).build();
//! let reply = agent.prompt("Say hello").await?;
//! println!("{reply}");
//! # Ok(())
//! # }
//! ```
//!
//! # Environment
//!
//! | Variable | Purpose |
//! |----------|---------|
//! | `AWS_REGION` / `AWS_DEFAULT_REGION` | Region for Mantle URL + token mint (default `us-east-1`) |
//! | `AWS_BEARER_TOKEN_BEDROCK` | Optional pre-minted bearer token; skips IAM mint when set |
//! | AWS credential chain | Used to mint a short-term token when no bearer env is set |

mod token;

pub use token::{
    format_api_key_token, generate_short_term_token, generate_token_from_credentials,
};

/// Env var for a pre-minted Bedrock Mantle bearer token (see AWS docs / #1713).
pub const AWS_BEARER_TOKEN_BEDROCK_ENV: &str = "AWS_BEARER_TOKEN_BEDROCK";

/// OpenAI GPT-OSS 20B on Bedrock Mantle (versioned model id).
pub const OPENAI_GPT_OSS_20B: &str = "openai.gpt-oss-20b-1:0";
/// OpenAI GPT-OSS 120B on Bedrock Mantle (versioned model id).
pub const OPENAI_GPT_OSS_120B: &str = "openai.gpt-oss-120b-1:0";
/// Unversioned Mantle alias used in some AWS examples (`openai.gpt-oss-20b`).
pub const OPENAI_GPT_OSS_20B_MANTLE: &str = "openai.gpt-oss-20b";
/// Unversioned Mantle alias used in some AWS examples (`openai.gpt-oss-120b`).
pub const OPENAI_GPT_OSS_120B_MANTLE: &str = "openai.gpt-oss-120b";

/// OpenAI Responses API client pointed at Mantle.
pub type Client = rig_core::providers::openai::Client;
/// OpenAI Chat Completions API client pointed at Mantle.
pub type CompletionsClient = rig_core::providers::openai::CompletionsClient;

const DEFAULT_REGION: &str = "us-east-1";

/// Errors from Mantle token minting or OpenAI-compatible client construction.
#[derive(Debug, thiserror::Error)]
pub enum MantleError {
    /// Failed to mint or format a short-term bearer token.
    #[error("failed to mint Bedrock Mantle bearer token: {0}")]
    Token(String),
    /// Failed to resolve AWS credentials for token minting.
    #[error("failed to resolve AWS credentials for Bedrock Mantle: {0}")]
    Credentials(String),
    /// Failed to build the OpenAI-compatible HTTP client.
    #[error("failed to build OpenAI-compatible Mantle client: {0}")]
    ClientBuild(String),
}

/// Build the Mantle OpenAI-compatible base URL for a region.
///
/// Example: `https://bedrock-mantle.us-east-1.api.aws/openai/v1`
pub fn openai_base_url(region: &str) -> String {
    format!("https://bedrock-mantle.{region}.api.aws/openai/v1")
}

/// Builder for an OpenAI-compatible client pointed at Bedrock Mantle.
///
/// By default the builder mints a short-term IAM bearer token for the configured
/// region. Call [`ClientBuilder::api_key`] to skip minting and use a supplied
/// token (for example from [`AWS_BEARER_TOKEN_BEDROCK_ENV`]).
#[derive(Debug, Clone)]
pub struct ClientBuilder {
    region: String,
    api_key: Option<String>,
    base_url: Option<String>,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ClientBuilder {
    /// Create a builder with default region `us-east-1`.
    pub fn new() -> Self {
        Self {
            region: DEFAULT_REGION.to_string(),
            api_key: None,
            base_url: None,
        }
    }

    /// Set the AWS region used for the Mantle base URL and (when minting) SigV4.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    /// Supply a bearer token explicitly and skip short-term IAM minting.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Override the Mantle base URL (defaults to [`openai_base_url`] for the region).
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Build a Responses API client (default OpenAI surface).
    pub async fn build(self) -> Result<Client, MantleError> {
        let (api_key, base_url) = self.resolve_auth().await?;
        Client::builder()
            .api_key(api_key)
            .base_url(base_url)
            .build()
            .map_err(|e| MantleError::ClientBuild(e.to_string()))
    }

    /// Build a Chat Completions API client.
    pub async fn build_completions(self) -> Result<CompletionsClient, MantleError> {
        let (api_key, base_url) = self.resolve_auth().await?;
        CompletionsClient::builder()
            .api_key(api_key)
            .base_url(base_url)
            .build()
            .map_err(|e| MantleError::ClientBuild(e.to_string()))
    }

    /// Build a Responses client from environment variables.
    ///
    /// - Region: `AWS_REGION`, then `AWS_DEFAULT_REGION`, else `us-east-1`
    /// - Token: `AWS_BEARER_TOKEN_BEDROCK` if set, otherwise a short-term IAM mint
    pub async fn from_env() -> Result<Client, MantleError> {
        Self::from_env_builder().build().await
    }

    /// Build a Completions client from environment variables (same resolution as [`from_env`]).
    pub async fn from_env_completions() -> Result<CompletionsClient, MantleError> {
        Self::from_env_builder().build_completions().await
    }

    fn from_env_builder() -> Self {
        let mut builder = Self::new().region(resolve_region_from_env());
        if let Ok(token) = std::env::var(AWS_BEARER_TOKEN_BEDROCK_ENV)
            && !token.is_empty()
        {
            builder = builder.api_key(token);
        }
        builder
    }

    async fn resolve_auth(self) -> Result<(String, String), MantleError> {
        let base_url = self
            .base_url
            .unwrap_or_else(|| openai_base_url(&self.region));
        let api_key = match self.api_key {
            Some(key) => key,
            None => generate_short_term_token(&self.region).await?,
        };
        Ok((api_key, base_url))
    }
}

fn resolve_region_from_env() -> String {
    std::env::var("AWS_REGION")
        .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
        .unwrap_or_else(|_| DEFAULT_REGION.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_base_url_shape() {
        assert_eq!(
            openai_base_url("us-east-1"),
            "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
        );
        assert_eq!(
            openai_base_url("eu-west-1"),
            "https://bedrock-mantle.eu-west-1.api.aws/openai/v1"
        );
    }

    #[test]
    fn model_constants() {
        assert_eq!(OPENAI_GPT_OSS_20B, "openai.gpt-oss-20b-1:0");
        assert_eq!(OPENAI_GPT_OSS_120B, "openai.gpt-oss-120b-1:0");
        assert_eq!(OPENAI_GPT_OSS_20B_MANTLE, "openai.gpt-oss-20b");
        assert_eq!(OPENAI_GPT_OSS_120B_MANTLE, "openai.gpt-oss-120b");
    }

    #[test]
    fn bearer_env_name() {
        assert_eq!(AWS_BEARER_TOKEN_BEDROCK_ENV, "AWS_BEARER_TOKEN_BEDROCK");
    }

    #[test]
    fn client_builder_defaults() {
        let builder = ClientBuilder::new();
        assert_eq!(builder.region, DEFAULT_REGION);
        assert!(builder.api_key.is_none());
        assert!(builder.base_url.is_none());
    }

    #[tokio::test]
    async fn build_with_explicit_api_key_skips_mint() {
        let client = ClientBuilder::new()
            .region("us-west-2")
            .api_key("bedrock-api-key-test")
            .build()
            .await
            .expect("client builds without AWS credentials");
        assert_eq!(
            client.base_url(),
            "https://bedrock-mantle.us-west-2.api.aws/openai/v1"
        );
    }

    #[tokio::test]
    async fn build_with_base_url_override() {
        let client = ClientBuilder::new()
            .api_key("test-key")
            .base_url("http://127.0.0.1:9/openai/v1")
            .build()
            .await
            .expect("client builds");
        assert_eq!(client.base_url(), "http://127.0.0.1:9/openai/v1");
    }
}

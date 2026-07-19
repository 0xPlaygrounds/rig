//! AWS Bedrock Mantle (OpenAI-compatible) client.
//!
//! Mantle exposes selected Bedrock models (including OpenAI GPT-OSS) through an
//! OpenAI-compatible HTTP API. Auth uses a short-term IAM bearer token
//! (`bedrock-api-key-…`) minted via SigV4, or a pre-supplied
//! [`AWS_BEARER_TOKEN_BEDROCK`] env value.
//!
//! This module uses Mantle-specific Rig client types (OpenAI wire format, Bedrock
//! defaults and telemetry). It does **not** type-alias the public OpenAI client
//! and does **not** change the Converse path.
//!
//! # Base URL dual path
//!
//! | Path | Helper | Models |
//! |------|--------|--------|
//! | `https://bedrock-mantle.{region}.api.aws/v1` | [`openai_base_url`] (default) | GPT-OSS and most Mantle models (Completions + Responses) |
//! | `https://bedrock-mantle.{region}.api.aws/openai/v1` | [`openai_gpt5_base_url`] | GPT-5.x family Responses (e.g. `openai.gpt-5.6-luna`) |
//!
//! # Token lifetime
//!
//! Short-term IAM tokens last at most **12 hours** ([`TOKEN_TTL`]), capped by
//! the source AWS credential session (SSO / AssumeRole / instance role often
//! expire much sooner). The token is **snapshotted at client build time**.
//! Long-lived processes must rebuild the client (or supply a fresh `api_key`)
//! before the effective TTL elapses.
//!
//! # `store: false` gotcha
//!
//! Some Mantle Responses models reject or mis-handle default OpenAI `store: true`
//! semantics. Prefer
//! `.additional_params(serde_json::json!({"store": false}))` on agents that use
//! the Responses path.
//!
//! # Example
//!
//! ```no_run
//! use rig_bedrock::mantle::{self, ClientBuilder, OPENAI_GPT_OSS_20B};
//! use rig_core::client::CompletionClient;
//! use rig_core::completion::Prompt;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = mantle::from_env().await?;
//! let agent = client
//!     .agent(OPENAI_GPT_OSS_20B)
//!     .additional_params(serde_json::json!({"store": false}))
//!     .build();
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

mod provider;
mod token;

pub use provider::{
    CompletionsClient, CompletionsClientBuilder, MantleCompletionsBuilder, MantleCompletionsExt,
    MantleResponsesBuilder, MantleResponsesExt, PROVIDER_NAME, ResponsesClient,
    ResponsesClientBuilder, DEFAULT_MANTLE_BASE_URL,
};
pub use token::{
    effective_token_ttl, format_api_key_token, generate_short_term_token,
    generate_short_term_token_with_profile, generate_token_from_credentials, refresh_after_from_ttl,
    TOKEN_TTL, TOKEN_TTL_SECS,
};

/// Env var for a pre-minted Bedrock Mantle bearer token (see AWS docs / #1713).
pub const AWS_BEARER_TOKEN_BEDROCK_ENV: &str = "AWS_BEARER_TOKEN_BEDROCK";

/// OpenAI GPT-OSS 20B on Bedrock Mantle (unversioned catalog id).
pub const OPENAI_GPT_OSS_20B: &str = "openai.gpt-oss-20b";
/// OpenAI GPT-OSS 120B on Bedrock Mantle (unversioned catalog id).
pub const OPENAI_GPT_OSS_120B: &str = "openai.gpt-oss-120b";
/// OpenAI GPT-5.4 on Bedrock Mantle (use [`openai_gpt5_base_url`] for Responses).
pub const OPENAI_GPT_5_4: &str = "openai.gpt-5.4";
/// OpenAI GPT-5.5 on Bedrock Mantle (use [`openai_gpt5_base_url`] for Responses).
pub const OPENAI_GPT_5_5: &str = "openai.gpt-5.5";
/// OpenAI GPT-5.6 Luna on Bedrock Mantle (use [`openai_gpt5_base_url`] for Responses).
pub const OPENAI_GPT_5_6_LUNA: &str = "openai.gpt-5.6-luna";
/// OpenAI GPT-5.6 Sol on Bedrock Mantle (use [`openai_gpt5_base_url`] for Responses).
pub const OPENAI_GPT_5_6_SOL: &str = "openai.gpt-5.6-sol";
/// OpenAI GPT-5.6 Terra on Bedrock Mantle (use [`openai_gpt5_base_url`] for Responses).
pub const OPENAI_GPT_5_6_TERRA: &str = "openai.gpt-5.6-terra";

/// Versioned id used by Bedrock Runtime / Converse — **not** the Mantle OpenAI catalog id.
///
/// Mantle `GET /v1/models` lists unversioned ids such as [`OPENAI_GPT_OSS_20B`].
pub const OPENAI_GPT_OSS_20B_VERSIONED: &str = "openai.gpt-oss-20b-1:0";
/// Versioned id used by Bedrock Runtime / Converse — **not** the Mantle OpenAI catalog id.
pub const OPENAI_GPT_OSS_120B_VERSIONED: &str = "openai.gpt-oss-120b-1:0";

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

/// Default Mantle OpenAI-compatible base (GPT-OSS and most Mantle models).
///
/// Example: `https://bedrock-mantle.us-east-1.api.aws/v1`
pub fn openai_base_url(region: &str) -> String {
    format!("https://bedrock-mantle.{region}.api.aws/v1")
}

/// GPT-5.x Mantle path (Responses for the `openai.gpt-5.*` family).
///
/// Example: `https://bedrock-mantle.us-east-1.api.aws/openai/v1`
///
/// Note: this path does **not** support Responses for GPT-OSS models; use
/// [`openai_base_url`] for those.
pub fn openai_gpt5_base_url(region: &str) -> String {
    format!("https://bedrock-mantle.{region}.api.aws/openai/v1")
}

/// Builder for an OpenAI-compatible client pointed at Bedrock Mantle.
///
/// By default the builder mints a short-term IAM bearer token for the configured
/// region. Call [`ClientBuilder::api_key`] to skip minting and use a supplied
/// token (for example from [`AWS_BEARER_TOKEN_BEDROCK_ENV`]).
///
/// The bearer token is **snapshotted when [`ClientBuilder::build`] /
/// [`ClientBuilder::build_completions`] runs**. Tokens expire after
/// [`TOKEN_TTL`] (12 hours); rebuild the client for long-lived processes.
#[derive(Clone)]
pub struct ClientBuilder {
    region: String,
    api_key: Option<String>,
    base_url: Option<String>,
    profile_name: Option<String>,
}

impl std::fmt::Debug for ClientBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientBuilder")
            .field("region", &self.region)
            .field(
                "api_key",
                &self.api_key.as_ref().map(|_| "[REDACTED]"),
            )
            .field("base_url", &self.base_url)
            .field("profile_name", &self.profile_name)
            .finish()
    }
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
            profile_name: None,
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
    ///
    /// Use [`openai_gpt5_base_url`] for GPT-5.x Responses.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Use a named AWS shared-config profile when minting a short-term token.
    ///
    /// Ignored when [`ClientBuilder::api_key`] is set.
    pub fn profile_name(mut self, profile_name: impl Into<String>) -> Self {
        self.profile_name = Some(profile_name.into());
        self
    }

    /// Build a Mantle Responses API client.
    ///
    /// The bearer token is snapshotted into the client. See [`TOKEN_TTL`] and
    /// [`effective_token_ttl`]. Defaults never point at `api.openai.com`.
    pub async fn build(self) -> Result<ResponsesClient, MantleError> {
        let (api_key, base_url) = self.resolve_auth().await?;
        ResponsesClient::builder()
            .api_key(api_key)
            .base_url(base_url)
            .build()
            .map_err(|e| MantleError::ClientBuild(e.to_string()))
    }

    /// Build a Mantle Chat Completions API client.
    ///
    /// The bearer token is snapshotted into the client. See [`TOKEN_TTL`] and
    /// [`effective_token_ttl`]. Defaults never point at `api.openai.com`.
    pub async fn build_completions(self) -> Result<CompletionsClient, MantleError> {
        let (api_key, base_url) = self.resolve_auth().await?;
        CompletionsClient::builder()
            .api_key(api_key)
            .base_url(base_url)
            .build()
            .map_err(|e| MantleError::ClientBuild(e.to_string()))
    }

    /// Create a builder from environment variables (sync; env defaults only).
    ///
    /// - Region: `AWS_REGION`, then `AWS_DEFAULT_REGION`, else `us-east-1`
    /// - Token: `AWS_BEARER_TOKEN_BEDROCK` if set (otherwise mint on [`build`])
    ///
    /// This does **not** build a client. Call [`.build().await`](Self::build) or
    /// [`.build_completions().await`](Self::build_completions), or use the free
    /// functions [`from_env`] / [`from_env_completions`].
    pub fn from_env() -> Self {
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
            None => {
                generate_short_term_token_with_profile(
                    &self.region,
                    self.profile_name.as_deref(),
                )
                .await?
            }
        };
        Ok((api_key, base_url))
    }
}

/// Build a Mantle Responses client from environment variables.
///
/// Equivalent to `ClientBuilder::from_env().build().await`.
pub async fn from_env() -> Result<ResponsesClient, MantleError> {
    ClientBuilder::from_env().build().await
}

/// Build a Mantle Completions client from environment variables.
///
/// Equivalent to `ClientBuilder::from_env().build_completions().await`.
pub async fn from_env_completions() -> Result<CompletionsClient, MantleError> {
    ClientBuilder::from_env().build_completions().await
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
            "https://bedrock-mantle.us-east-1.api.aws/v1"
        );
        assert_eq!(
            openai_base_url("eu-west-1"),
            "https://bedrock-mantle.eu-west-1.api.aws/v1"
        );
    }

    #[test]
    fn openai_gpt5_base_url_shape() {
        assert_eq!(
            openai_gpt5_base_url("us-east-1"),
            "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
        );
        assert_eq!(
            openai_gpt5_base_url("eu-west-1"),
            "https://bedrock-mantle.eu-west-1.api.aws/openai/v1"
        );
    }

    #[test]
    fn model_constants() {
        assert_eq!(OPENAI_GPT_OSS_20B, "openai.gpt-oss-20b");
        assert_eq!(OPENAI_GPT_OSS_120B, "openai.gpt-oss-120b");
        assert_eq!(OPENAI_GPT_5_4, "openai.gpt-5.4");
        assert_eq!(OPENAI_GPT_5_5, "openai.gpt-5.5");
        assert_eq!(OPENAI_GPT_5_6_LUNA, "openai.gpt-5.6-luna");
        assert_eq!(OPENAI_GPT_5_6_SOL, "openai.gpt-5.6-sol");
        assert_eq!(OPENAI_GPT_5_6_TERRA, "openai.gpt-5.6-terra");
        assert_eq!(OPENAI_GPT_OSS_20B_VERSIONED, "openai.gpt-oss-20b-1:0");
        assert_eq!(OPENAI_GPT_OSS_120B_VERSIONED, "openai.gpt-oss-120b-1:0");
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
        assert!(builder.profile_name.is_none());
    }

    #[test]
    fn from_env_builder_is_sync() {
        // Compiles as a pure env snapshot — does not mint or build HTTP client.
        let builder = ClientBuilder::from_env();
        let _ = builder.region;
    }

    #[test]
    fn client_builder_debug_redacts_api_key() {
        let builder = ClientBuilder::new()
            .region("us-east-1")
            .api_key("bedrock-api-key-super-secret-token-value");
        let debug = format!("{builder:?}");
        assert!(
            debug.contains("[REDACTED]"),
            "Debug should redact api_key: {debug}"
        );
        assert!(
            !debug.contains("super-secret"),
            "Debug must not leak api_key: {debug}"
        );
        assert!(debug.contains("us-east-1"));
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
            "https://bedrock-mantle.us-west-2.api.aws/v1"
        );
    }

    #[tokio::test]
    async fn responses_client_builder_defaults_to_mantle_not_openai() {
        let client = ResponsesClient::builder()
            .api_key("bedrock-api-key-test")
            .build()
            .expect("mantle responses client");
        assert!(
            client.base_url().contains("bedrock-mantle"),
            "base_url={}",
            client.base_url()
        );
        assert!(
            !client.base_url().contains("api.openai.com"),
            "must not default to OpenAI: {}",
            client.base_url()
        );
    }

    #[tokio::test]
    async fn completions_client_builder_defaults_to_mantle_not_openai() {
        let client = CompletionsClient::builder()
            .api_key("bedrock-api-key-test")
            .build()
            .expect("mantle completions client");
        assert!(client.base_url().contains("bedrock-mantle"));
        assert!(!client.base_url().contains("api.openai.com"));
    }

    #[tokio::test]
    async fn build_with_base_url_override() {
        let client = ClientBuilder::new()
            .api_key("test-key")
            .base_url("http://127.0.0.1:9/v1")
            .build()
            .await
            .expect("client builds");
        assert_eq!(client.base_url(), "http://127.0.0.1:9/v1");
    }

    #[tokio::test]
    async fn build_completions_with_explicit_api_key() {
        let client = ClientBuilder::new()
            .region("us-east-1")
            .api_key("bedrock-api-key-test")
            .build_completions()
            .await
            .expect("completions client builds");
        assert_eq!(
            client.base_url(),
            "https://bedrock-mantle.us-east-1.api.aws/v1"
        );
    }

    #[tokio::test]
    async fn build_with_gpt5_base_url() {
        let client = ClientBuilder::new()
            .api_key("test-key")
            .base_url(openai_gpt5_base_url("us-east-1"))
            .build()
            .await
            .expect("client builds");
        assert_eq!(
            client.base_url(),
            "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
        );
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;
    use rig_core::client::CompletionClient;
    use rig_core::completion::Prompt;

    /// Live Mantle Completions smoke for GPT-OSS 20B on the default `/v1` path.
    ///
    /// ```shell
    /// export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
    /// eval "$(aws configure export-credentials --format env)"
    /// cargo test -p rig-bedrock live_mantle -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore = "requires AWS credentials and Mantle model access in us-east-1"]
    async fn live_mantle_completions_gpt_oss_20b() {
        let client = from_env_completions()
            .await
            .expect("mantle completions client");
        let agent = client
            .agent(OPENAI_GPT_OSS_20B)
            .preamble("You are a concise assistant.")
            .build();
        let reply = agent
            .prompt("Reply with the single word: pong")
            .await
            .expect("completions prompt should succeed");
        assert!(
            !reply.trim().is_empty(),
            "expected non-empty Completions reply"
        );
        eprintln!("live_mantle completions reply: {reply}");
    }

    /// Live Mantle Responses smoke for GPT-OSS 20B (needs float `created_at` fix).
    #[tokio::test]
    #[ignore = "requires AWS credentials and Mantle model access in us-east-1"]
    async fn live_mantle_responses_gpt_oss_20b() {
        let client = from_env().await.expect("mantle responses client");
        let agent = client
            .agent(OPENAI_GPT_OSS_20B)
            .preamble("You are a concise assistant.")
            .additional_params(serde_json::json!({"store": false}))
            .build();
        let reply = agent
            .prompt("Reply with the single word: pong")
            .await
            .expect("responses prompt should succeed");
        assert!(
            !reply.trim().is_empty(),
            "expected non-empty Responses reply"
        );
        eprintln!("live_mantle responses reply: {reply}");
    }
}

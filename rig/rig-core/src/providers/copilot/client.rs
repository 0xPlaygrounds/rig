use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    http_client::{self, HttpClientExt},
};

// ================================================================
// Copilot Client (OpenAI-compatible with relaxed response parsing)
// ================================================================
const COPILOT_API_BASE_URL: &str = "https://api.githubcopilot.com";

#[derive(Debug, Default, Clone, Copy)]
pub struct CopilotExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct CopilotExtBuilder;

type CopilotApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<CopilotExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<CopilotExtBuilder, CopilotApiKey, H>;

impl Provider for CopilotExt {
    type Builder = CopilotExtBuilder;
    /// Copilot exposes a `/models` listing endpoint.
    const VERIFY_PATH: &'static str = "/models";
}

impl<H> Capabilities<H> for CopilotExt {
    type Completion = Capable<super::completion::CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for CopilotExt {}

impl ProviderBuilder for CopilotExtBuilder {
    type Extension<H>
        = CopilotExt
    where
        H: HttpClientExt;
    type ApiKey = CopilotApiKey;

    const BASE_URL: &'static str = COPILOT_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(CopilotExt)
    }
}

impl ProviderClient for Client {
    type Input = CopilotApiKey;

    /// Create a new Copilot client from environment variables.
    ///
    /// Reads `COPILOT_API_KEY` (or `GITHUB_TOKEN`) for the bearer token and
    /// optionally `COPILOT_BASE_URL` for a custom endpoint.
    fn from_env() -> Self {
        let api_key = std::env::var("COPILOT_API_KEY")
            .or_else(|_| std::env::var("GITHUB_TOKEN"))
            .expect("COPILOT_API_KEY or GITHUB_TOKEN not set");

        let base_url: Option<String> = std::env::var("COPILOT_BASE_URL").ok();

        let mut builder = Client::builder().api_key(&api_key);

        if let Some(base) = base_url {
            builder = builder.base_url(&base);
        }

        builder.build().unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(input).unwrap()
    }
}

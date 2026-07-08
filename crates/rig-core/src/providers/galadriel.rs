//! Galadriel API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::{client::CompletionClient, providers::galadriel};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = galadriel::Client::new("YOUR_API_KEY")?;
//! // to use a fine-tuned model
//! // let client = galadriel::Client::builder()
//! //     .api_key("YOUR_API_KEY")
//! //     .fine_tune_api_key("FINE_TUNE_API_KEY")
//! //     .build()?;
//!
//! let gpt4o = client.completion_model(galadriel::GPT_4O);
//! # Ok(())
//! # }
//! ```
use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::http_client::{self, HttpClientExt};
use serde::{Deserialize, Serialize};

// ================================================================
// Main Galadriel Client
// ================================================================
const GALADRIEL_API_BASE_URL: &str = "https://api.galadriel.com/v1/verified";

#[derive(Debug, Default, Clone)]
pub struct GaladrielExt {
    fine_tune_api_key: Option<String>,
}

#[derive(Debug, Default, Clone)]
pub struct GaladrielBuilder {
    fine_tune_api_key: Option<String>,
}

type GaladrielApiKey = BearerAuth;

impl crate::providers::openai::completion::OpenAICompatibleProvider for GaladrielExt {
    const PROVIDER_NAME: &'static str = "galadriel";

    // Galadriel's structured-output support is unverified; keep the
    // pre-migration behavior of dropping `output_schema` with a warning.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    type StreamingUsage = crate::providers::openai::Usage;

    type Response = crate::providers::openai::CompletionResponse;
}

impl Provider for GaladrielExt {
    type Builder = GaladrielBuilder;

    /// There is currently no way to verify a Galadriel api key without consuming tokens
    const VERIFY_PATH: &'static str = "";
}

impl<H> Capabilities<H> for GaladrielExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl DebugExt for GaladrielExt {
    fn fields(&self) -> impl Iterator<Item = (&'static str, &dyn std::fmt::Debug)> {
        std::iter::once((
            "fine_tune_api_key",
            (&self.fine_tune_api_key as &dyn std::fmt::Debug),
        ))
    }
}

impl ProviderBuilder for GaladrielBuilder {
    type Extension<H>
        = GaladrielExt
    where
        H: HttpClientExt;
    type ApiKey = GaladrielApiKey;

    const BASE_URL: &'static str = GALADRIEL_API_BASE_URL;

    fn build<H>(
        builder: &crate::client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        let GaladrielBuilder { fine_tune_api_key } = builder.ext().clone();

        Ok(GaladrielExt { fine_tune_api_key })
    }
}

pub type Client<H = reqwest::Client> = client::Client<GaladrielExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<GaladrielBuilder, GaladrielApiKey, H>;

impl<T> ClientBuilder<T> {
    pub fn fine_tune_api_key<S>(mut self, fine_tune_api_key: S) -> Self
    where
        S: AsRef<str>,
    {
        *self.ext_mut() = GaladrielBuilder {
            fine_tune_api_key: Some(fine_tune_api_key.as_ref().into()),
        };

        self
    }
}

impl ProviderClient for Client {
    type Input = (String, Option<String>);
    type Error = crate::client::ProviderClientError;

    /// Create a new Galadriel client from the `GALADRIEL_API_KEY` environment variable,
    /// and optionally from the `GALADRIEL_FINE_TUNE_API_KEY` environment variable.
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("GALADRIEL_API_KEY")?;
        let fine_tune_api_key = crate::client::optional_env_var("GALADRIEL_FINE_TUNE_API_KEY")?;

        let mut builder = Self::builder().api_key(api_key);

        if let Some(fine_tune_api_key) = fine_tune_api_key {
            builder = builder.fine_tune_api_key(fine_tune_api_key);
        }

        builder.build().map_err(Into::into)
    }

    fn from_val((api_key, fine_tune_api_key): Self::Input) -> Result<Self, Self::Error> {
        let mut builder = Self::builder().api_key(api_key);

        if let Some(fine_tune_key) = fine_tune_api_key {
            builder = builder.fine_tune_api_key(fine_tune_key)
        }

        builder.build().map_err(Into::into)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
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

// ================================================================
// Galadriel Completion API
// ================================================================

/// `o1-preview` completion model
pub const O1_PREVIEW: &str = "o1-preview";
/// `o1-preview-2024-09-12` completion model
pub const O1_PREVIEW_2024_09_12: &str = "o1-preview-2024-09-12";
/// `o1-mini completion model
pub const O1_MINI: &str = "o1-mini";
/// `o1-mini-2024-09-12` completion model
pub const O1_MINI_2024_09_12: &str = "o1-mini-2024-09-12";
/// `gpt-4o` completion model
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-2024-05-13` completion model
pub const GPT_4O_2024_05_13: &str = "gpt-4o-2024-05-13";
/// `gpt-4-turbo` completion model
pub const GPT_4_TURBO: &str = "gpt-4-turbo";
/// `gpt-4-turbo-2024-04-09` completion model
pub const GPT_4_TURBO_2024_04_09: &str = "gpt-4-turbo-2024-04-09";
/// `gpt-4-turbo-preview` completion model
pub const GPT_4_TURBO_PREVIEW: &str = "gpt-4-turbo-preview";
/// `gpt-4-0125-preview` completion model
pub const GPT_4_0125_PREVIEW: &str = "gpt-4-0125-preview";
/// `gpt-4-1106-preview` completion model
pub const GPT_4_1106_PREVIEW: &str = "gpt-4-1106-preview";
/// `gpt-4-vision-preview` completion model
pub const GPT_4_VISION_PREVIEW: &str = "gpt-4-vision-preview";
/// `gpt-4-1106-vision-preview` completion model
pub const GPT_4_1106_VISION_PREVIEW: &str = "gpt-4-1106-vision-preview";
/// `gpt-4` completion model
pub const GPT_4: &str = "gpt-4";
/// `gpt-4-0613` completion model
pub const GPT_4_0613: &str = "gpt-4-0613";
/// `gpt-4-32k` completion model
pub const GPT_4_32K: &str = "gpt-4-32k";
/// `gpt-4-32k-0613` completion model
pub const GPT_4_32K_0613: &str = "gpt-4-32k-0613";
/// `gpt-3.5-turbo` completion model
pub const GPT_35_TURBO: &str = "gpt-3.5-turbo";
/// `gpt-3.5-turbo-0125` completion model
pub const GPT_35_TURBO_0125: &str = "gpt-3.5-turbo-0125";
/// `gpt-3.5-turbo-1106` completion model
pub const GPT_35_TURBO_1106: &str = "gpt-3.5-turbo-1106";
/// `gpt-3.5-turbo-instruct` completion model
pub const GPT_35_TURBO_INSTRUCT: &str = "gpt-3.5-turbo-instruct";

/// Galadriel completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = reqwest::Client> =
    crate::providers::openai::completion::GenericCompletionModel<GaladrielExt, H>;

#[cfg(test)]
mod tests {
    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::galadriel::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::galadriel::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }
}

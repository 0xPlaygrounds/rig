//! OrcaRouter API client and Rig integration
//!
//! [OrcaRouter](https://www.orcarouter.ai) is a gateway that aggregates
//! thousands of AI models behind one OpenAI-compatible API, routing each
//! request to the best provider and model. It also runs gateway-level,
//! zero-trust security for AI agents on the same endpoint — screening every
//! prompt/response and governing every tool call on a default-deny basis, with
//! no application code changes.
//!
//! # Example
//! ```no_run
//! use rig_core::{client::CompletionClient, providers::orcarouter};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = orcarouter::Client::from_env()?;
//!
//! let auto = client.completion_model(orcarouter::ORCAROUTER_AUTO);
//! # Ok(())
//! # }
//! ```

use crate::client::{self, BearerAuth, DebugExt, Provider};

use super::openai;

// ================================================================
// Main OrcaRouter Client
// ================================================================
const ORCAROUTER_API_BASE_URL: &str = "https://api.orcarouter.ai/v1";

#[derive(Debug, Default, Clone, Copy)]
pub struct OrcaRouterExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct OrcaRouterBuilder;

type OrcaRouterApiKey = BearerAuth;

impl Provider for OrcaRouterExt {
    type Builder = OrcaRouterBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

impl openai::completion::OpenAICompatibleProvider for OrcaRouterExt {
    const PROVIDER_NAME: &'static str = "orcarouter";

    type StreamingUsage = openai::Usage;

    // OrcaRouter's gateway forwards OpenAI-style requests and responses; the
    // shared OpenAI Chat Completions path applies unchanged.
    type Response = openai::CompletionResponse;
}

client::impl_capabilities!(
    OrcaRouterExt,
    completion = CompletionModel<H>,
    model_listing = OrcaRouterModelLister<H>,
);

impl DebugExt for OrcaRouterExt {}

client::impl_default_provider_builder!(
    OrcaRouterBuilder => OrcaRouterExt,
    api_key = OrcaRouterApiKey,
    base_url = ORCAROUTER_API_BASE_URL,
);

crate::providers::internal::model_listing::impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) implementation for the
    /// OrcaRouter API (`GET /models`), the same path [`OrcaRouterExt::VERIFY_PATH`]
    /// already uses.
    OrcaRouterModelLister,
    Client<H>,
    crate::providers::internal::model_listing::ListModelEntry,
    "OrcaRouter",
    "/models"
);

pub type Client<H = reqwest::Client> = client::Client<OrcaRouterExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<OrcaRouterBuilder, OrcaRouterApiKey, H>;

/// OrcaRouter completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<OrcaRouterExt, H>;

/// OrcaRouter's provider-native terminal streaming record: the value carried by
/// the final item of the stream returned by `CompletionModel::raw_stream`. Shared
/// with the OpenAI Chat Completions path, usage payload included.
pub type StreamingCompletionResponse = openai::StreamingCompletionResponse;

client::impl_provider_client!(Client, input = String, api_key_env = "ORCAROUTER_API_KEY");

// ================================================================
// OrcaRouter Completion API
// ================================================================

/// `orcarouter/auto` — OrcaRouter's default routing model. The gateway picks
/// the best available model for the request across its aggregated providers.
pub const ORCAROUTER_AUTO: &str = "orcarouter/auto";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::ProviderBuilder;

    #[test]
    fn test_client_initialization() {
        let _client = Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }

    #[test]
    fn default_base_url_points_at_orcarouter_v1() {
        assert_eq!(OrcaRouterBuilder::BASE_URL, "https://api.orcarouter.ai/v1");
    }
}

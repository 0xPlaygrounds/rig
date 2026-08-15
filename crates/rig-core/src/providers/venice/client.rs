//! Venice client, provider extension, and capability wiring.

use crate::client::{self, BearerAuth, DebugExt, Provider};
use crate::model::Model;

// ================================================================
// Venice Client
// ================================================================
// The base URL carries the `/api/v1` prefix, so request paths are bare
// (`/chat/completions`), matching every other OpenAI-compatible provider here.
/// Venice's API base URL.
pub const VENICE_API_BASE_URL: &str = "https://api.venice.ai/api/v1";

/// Provider extension type for Venice.
#[derive(Debug, Default, Clone, Copy)]
pub struct VeniceExt;

/// Builder state for [`VeniceExt`].
#[derive(Debug, Default, Clone, Copy)]
pub struct VeniceBuilder;

type VeniceApiKey = BearerAuth;

/// Venice client.
pub type Client<H = reqwest::Client> = client::Client<VeniceExt, H>;
/// Builder for the Venice [`Client`].
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<VeniceBuilder, VeniceApiKey, H>;

impl Provider for VeniceExt {
    type Builder = VeniceBuilder;

    const VERIFY_PATH: &'static str = "/models";
}

impl DebugExt for VeniceExt {}

impl crate::providers::openai::completion::OpenAICompatibleProvider for VeniceExt {
    const PROVIDER_NAME: &'static str = "venice";

    type StreamingUsage = crate::providers::openai::Usage;

    // Venice echoes its resolved `venice_parameters` block (including web
    // search citations) and a per-request `cost` alongside the OpenAI-shaped
    // payload; the Venice response type preserves both.
    type Response = super::completion::CompletionResponse;
}

client::impl_capabilities!(
    VeniceExt,
    completion = super::completion::CompletionModel<H>,
    embeddings = super::embedding::EmbeddingModel<H>,
    transcription = super::transcription::TranscriptionModel<H>,
    model_listing = VeniceModelLister<H>,
    image_generation = super::image_generation::ImageGenerationModel<H>,
    audio_generation = super::audio_generation::AudioGenerationModel<H>,
);

client::impl_default_provider_builder!(
    VeniceBuilder => VeniceExt,
    api_key = VeniceApiKey,
    base_url = VENICE_API_BASE_URL,
);

client::impl_provider_client!(
    Client,
    input = String,
    api_key_env = "VENICE_API_KEY",
    base_url_env_first = "VENICE_BASE_URL",
);

/// A `GET /models` entry.
///
/// Venice returns the OpenAI-compatible envelope plus a `type` discriminator
/// (`text`, `image`, `embedding`, `tts`, `asr`, …) and a `model_spec` object;
/// only the fields [`Model`] can carry are decoded here.
#[derive(Debug, serde::Deserialize)]
struct ListModelEntry {
    id: String,
    #[serde(default)]
    owned_by: Option<String>,
}

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        let mut model = Model::from_id(value.id);
        model.owned_by = value.owned_by;
        model
    }
}

crate::providers::internal::model_listing::impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) implementation for the
    /// Venice API (`GET /models`).
    ///
    /// Venice also accepts a `?type=` filter; [`list_all`](crate::client::ModelLister::list_all) requests the
    /// unfiltered listing, which Venice answers with its text models.
    VeniceModelLister,
    Client<H>,
    ListModelEntry,
    "Venice",
    "/models"
);

#[cfg(test)]
mod tests {
    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::venice::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::venice::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }
}

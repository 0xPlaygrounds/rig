//! Venice client, provider extension, and capability wiring.

#[cfg(feature = "audio")]
use crate::client::HasAudioGeneration;
#[cfg(feature = "image")]
use crate::client::HasImageGeneration;
use crate::client::{
    self, BearerAuth, HasCompletion, HasEmbeddings, HasModelListing, HasTranscription,
    ModelTransport, Provider, ProviderClientResult,
};
use crate::http_client::{self, HttpClientExt};
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
pub struct Venice;

/// Builder state for [`Venice`].
type VeniceApiKey = BearerAuth;

/// Venice client.
pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<Venice, H>;
/// Builder for the Venice [`Client`].
pub type ClientBuilder<H = crate::markers::Missing> = client::ClientBuilder<Venice, H>;

impl Provider for Venice {
    const NAME: &'static str = "venice";
    const BASE_URL: &'static str = VENICE_API_BASE_URL;
    const VERIFY_PATH: &'static str = "/models";
    type ApiKey = VeniceApiKey;
    type Config = ();
    type EnvInput = String;

    fn build(_: (), _: &VeniceApiKey) -> http_client::Result<Self> {
        Ok(Venice)
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<H>> {
        Client::from_env_api_key("VENICE_API_KEY", Some("VENICE_BASE_URL"), http)
    }

    fn from_val<H: HttpClientExt>(input: String, http: H) -> ProviderClientResult<Client<H>> {
        Client::new_with(input, http)
    }
}

impl HasCompletion for Venice {
    type Model<H>
        = super::completion::CompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        super::completion::CompletionModel::new(client.clone(), model)
    }
}

impl HasEmbeddings for Venice {
    type Model<H>
        = super::embedding::EmbeddingModel<H>
    where
        H: ModelTransport;

    fn embedding_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
        ndims: Option<usize>,
    ) -> Self::Model<H> {
        super::embedding::EmbeddingModel::make(client, model, ndims)
    }
}

impl HasTranscription for Venice {
    type Model<H>
        = super::transcription::TranscriptionModel<H>
    where
        H: ModelTransport;

    fn transcription_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        super::transcription::TranscriptionModel::new(client.clone(), model)
    }
}

impl HasModelListing for Venice {
    type Lister<H>
        = VeniceModelLister<H>
    where
        H: ModelTransport;

    fn model_lister<H: ModelTransport>(client: &Client<H>) -> Self::Lister<H> {
        VeniceModelLister::new(client.clone())
    }
}

#[cfg(feature = "image")]
impl HasImageGeneration for Venice {
    type Model<H>
        = super::image_generation::ImageGenerationModel<H>
    where
        H: ModelTransport;

    fn image_generation_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
    ) -> Self::Model<H> {
        super::image_generation::ImageGenerationModel::new(client.clone(), model)
    }
}

#[cfg(feature = "audio")]
impl HasAudioGeneration for Venice {
    type Model<H>
        = super::audio_generation::AudioGenerationModel<H>
    where
        H: ModelTransport;

    fn audio_generation_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
    ) -> Self::Model<H> {
        super::audio_generation::AudioGenerationModel::new(client.clone(), model)
    }
}

impl crate::providers::openai::completion::OpenAICompatibleProvider for Venice {
    const PROVIDER_NAME: &'static str = "venice";

    type StreamingUsage = crate::providers::openai::Usage;

    // Venice echoes its resolved `venice_parameters` block (including web
    // search citations) and a per-request `cost` alongside the OpenAI-shaped
    // payload; the Venice response type preserves both.
    type Response = super::completion::CompletionResponse;
}

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
mod tests;

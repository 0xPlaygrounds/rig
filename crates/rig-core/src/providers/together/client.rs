use crate::client::{
    self, BearerAuth, HasCompletion, HasEmbeddings, ModelTransport, Provider, ProviderClientResult,
};
use crate::http_client::{self, HttpClientExt};

// ================================================================
// Together AI Client
// ================================================================
const TOGETHER_AI_BASE_URL: &str = "https://api.together.xyz";

#[derive(Debug, Default, Clone, Copy)]
pub struct Together;
type TogetherApiKey = BearerAuth;

pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<Together, H>;
pub type ClientBuilder<H = crate::markers::Missing> = client::ClientBuilder<Together, H>;

impl Provider for Together {
    const NAME: &'static str = "together";
    const BASE_URL: &'static str = TOGETHER_AI_BASE_URL;
    const VERIFY_PATH: &'static str = "/models";
    type ApiKey = TogetherApiKey;
    type Config = ();
    type EnvInput = String;

    fn build(_: (), _: &TogetherApiKey) -> http_client::Result<Self> {
        Ok(Together)
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<H>> {
        Client::from_env_api_key("TOGETHER_API_KEY", None, http)
    }

    fn from_val<H: HttpClientExt>(input: String, http: H) -> ProviderClientResult<Client<H>> {
        Client::new_with(input, http)
    }
}

impl HasCompletion for Together {
    type Model<H>
        = super::CompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        super::CompletionModel::new(client.clone(), model)
    }
}

impl HasEmbeddings for Together {
    type Model<H>
        = super::EmbeddingModel<H>
    where
        H: ModelTransport;

    fn embedding_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
        ndims: Option<usize>,
    ) -> Self::Model<H> {
        super::EmbeddingModel::make(client, model, ndims)
    }
}

impl crate::providers::openai::completion::OpenAICompatibleProvider for Together {
    const PROVIDER_NAME: &'static str = "together";

    type StreamingUsage = crate::providers::openai::Usage;

    // Together's structured-output support is model-dependent; keep the
    // pre-migration behavior of dropping `output_schema` with a warning.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    type Response = crate::providers::openai::CompletionResponse;

    // The client base URL is the bare host; embeddings build their own v1 path.
    fn completion_path(&self, _model: &str) -> String {
        "/v1/chat/completions".to_string()
    }
}

#[cfg(test)]
mod tests;

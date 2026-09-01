use super::*;
use crate::embeddings::{EmbeddingError, EmbeddingModel, EmbeddingResponse};
use crate::rerank::{RerankError, RerankModel, RerankResponse};
use crate::transcription::{
    TranscriptionError, TranscriptionModel, TranscriptionRequest, TranscriptionResponse,
};

#[derive(Debug, Default, Clone, Copy)]
struct ExternalExt;
#[derive(Debug, Default, Clone, Copy)]
struct ExternalExtBuilder;

impl Provider for ExternalExt {
    type Builder = ExternalExtBuilder;
    const VERIFY_PATH: &'static str = "/";
}

impl ProviderBuilder for ExternalExtBuilder {
    type Extension<H>
        = ExternalExt
    where
        H: HttpClientExt;
    type ApiKey = BearerAuth;

    const BASE_URL: &'static str = "https://external.invalid";

    fn build<H>(
        _builder: &ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(ExternalExt)
    }
}

impl<H> Capabilities<H> for ExternalExt {
    type Completion = Nothing;
    type Embeddings = Capable<ExternalModel<H>>;
    type Transcription = Capable<ExternalModel<H>>;
    type ModelListing = Capable<ExternalModel<H>>;
    #[cfg(feature = "image")]
    type ImageGeneration = Capable<ExternalModel<H>>;
    #[cfg(feature = "audio")]
    type AudioGeneration = Capable<ExternalModel<H>>;
    type Rerank = Capable<ExternalModel<H>>;
}

impl DebugExt for ExternalExt {}

/// One model type standing in for every modality; deliberately not
/// `Clone`, which the relaxed supertraits no longer require.
struct ExternalModel<H> {
    _client: Client<ExternalExt, H>,
    model: String,
    ndims: Option<usize>,
}

impl<H> TranscriptionModel for ExternalModel<H>
where
    H: Send + Sync + 'static,
{
    async fn transcription(
        &self,
        _request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse, TranscriptionError> {
        Err(TranscriptionError::ResponseError(self.model.clone()))
    }
}

impl<H> ConstructTranscriptionModel<Client<ExternalExt, H>> for ExternalModel<H>
where
    H: Clone,
{
    fn construct(client: &Client<ExternalExt, H>, model: String) -> Self {
        Self {
            _client: client.clone(),
            model,
            ndims: None,
        }
    }
}

impl<H> EmbeddingModel for ExternalModel<H>
where
    H: Send + Sync + 'static,
{
    fn max_documents(&self) -> usize {
        1
    }

    fn ndims(&self) -> usize {
        self.ndims.unwrap_or(3)
    }

    async fn embed_texts_response(
        &self,
        _texts: impl IntoIterator<Item = String> + Send,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        Err(EmbeddingError::ResponseError(self.model.clone()))
    }
}

impl<H> ConstructEmbeddingModel<Client<ExternalExt, H>> for ExternalModel<H>
where
    H: Clone,
{
    fn construct(client: &Client<ExternalExt, H>, model: String, ndims: Option<usize>) -> Self {
        Self {
            _client: client.clone(),
            model,
            ndims,
        }
    }
}

impl<H> RerankModel for ExternalModel<H>
where
    H: Send + Sync + 'static,
{
    fn max_documents(&self) -> usize {
        1
    }

    async fn rerank(
        &self,
        _query: &str,
        _documents: Vec<String>,
    ) -> Result<RerankResponse, RerankError> {
        Err(RerankError::ResponseError(self.model.clone()))
    }
}

impl<H> ConstructRerankModel<Client<ExternalExt, H>> for ExternalModel<H>
where
    H: Clone,
{
    fn construct(client: &Client<ExternalExt, H>, model: String) -> Self {
        Self {
            _client: client.clone(),
            model,
            ndims: None,
        }
    }
}

#[cfg(feature = "image")]
impl<H> ImageGenerationModel for ExternalModel<H>
where
    H: Send + Sync + 'static,
{
    async fn image_generation(
        &self,
        _request: crate::image_generation::ImageGenerationRequest,
    ) -> Result<
        crate::image_generation::ImageGenerationResponse,
        crate::image_generation::ImageGenerationError,
    > {
        Err(crate::image_generation::ImageGenerationError::ResponseError(self.model.clone()))
    }
}

#[cfg(feature = "image")]
impl<H> ConstructImageGenerationModel<Client<ExternalExt, H>> for ExternalModel<H>
where
    H: Clone,
{
    fn construct(client: &Client<ExternalExt, H>, model: String) -> Self {
        Self {
            _client: client.clone(),
            model,
            ndims: None,
        }
    }
}

#[cfg(feature = "audio")]
impl<H> AudioGenerationModel for ExternalModel<H>
where
    H: Send + Sync + 'static,
{
    async fn audio_generation(
        &self,
        _request: AudioGenerationRequest,
    ) -> Result<AudioGenerationResponse, AudioGenerationError> {
        Err(AudioGenerationError::ResponseError(self.model.clone()))
    }
}

#[cfg(feature = "audio")]
impl<H> ConstructAudioGenerationModel<Client<ExternalExt, H>> for ExternalModel<H>
where
    H: Clone,
{
    fn construct(client: &Client<ExternalExt, H>, model: String) -> Self {
        Self {
            _client: client.clone(),
            model,
            ndims: None,
        }
    }
}

impl<H> ModelLister<H> for ExternalModel<H>
where
    H: Send + Sync + 'static,
{
    async fn list_all(&self) -> Result<crate::model::ModelList, crate::model::ModelListingError> {
        Err(crate::model::ModelListingError::ParseError {
            message: self.model.clone(),
        })
    }
}

impl<H> ConstructModelLister<Client<ExternalExt, H>> for ExternalModel<H>
where
    H: Clone,
{
    fn construct(client: &Client<ExternalExt, H>) -> Self {
        Self {
            _client: client.clone(),
            model: "lister".to_owned(),
            ndims: None,
        }
    }
}

#[test]
fn external_extension_reaches_every_blanket_capability_client_impl() {
    fn assert_transcription<C: TranscriptionClient>() {}
    fn assert_embeddings<C: EmbeddingsClient>() {}
    fn assert_rerank<C: RerankingClient>() {}
    fn assert_listing<C: ModelListingClient>() {}
    #[cfg(feature = "image")]
    fn assert_image<C: ImageGenerationClient>() {}
    #[cfg(feature = "audio")]
    fn assert_audio<C: AudioGenerationClient>() {}

    type ExternalClient = Client<ExternalExt, crate::test_utils::RecordingHttpClient>;
    assert_transcription::<ExternalClient>();
    assert_embeddings::<ExternalClient>();
    assert_rerank::<ExternalClient>();
    assert_listing::<ExternalClient>();
    #[cfg(feature = "image")]
    assert_image::<ExternalClient>();
    #[cfg(feature = "audio")]
    assert_audio::<ExternalClient>();
}

#[test]
fn embedding_hook_receives_the_requested_dims() {
    let client: Client<ExternalExt, crate::test_utils::RecordingHttpClient> =
        Client::<ExternalExt, Missing>::builder()
            .api_key("key")
            .http_client(crate::test_utils::RecordingHttpClient::new(""))
            .build()
            .expect("client should build");
    assert_eq!(client.embedding_model("m").ndims(), 3);
    assert_eq!(client.embedding_model_with_ndims("m", 7).ndims(), 7);
}

/// `Arc<M>` is a model: the relaxed supertraits make "wrap it in an Arc"
/// real through the generic APIs, as for `CompletionModel`.
#[test]
fn arc_wrapped_models_satisfy_the_modality_traits() {
    fn assert_transcription_model<M: TranscriptionModel>() {}
    #[cfg(feature = "image")]
    fn assert_image_model<M: ImageGenerationModel>() {}
    #[cfg(feature = "audio")]
    fn assert_audio_model<M: AudioGenerationModel>() {}

    assert_transcription_model::<Arc<ExternalModel<crate::test_utils::RecordingHttpClient>>>();
    #[cfg(feature = "image")]
    assert_image_model::<Arc<ExternalModel<crate::test_utils::RecordingHttpClient>>>();
    #[cfg(feature = "audio")]
    assert_audio_model::<Arc<ExternalModel<crate::test_utils::RecordingHttpClient>>>();
}

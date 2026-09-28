//! The Gemini client: a [`GeminiConfig`] on a transport, and the models it
//! builds.

use crate::client::macros::http_client;
use crate::completion::CompletionRequest;
use crate::driver::{Model, Transport};
use crate::error::ProviderError;
use crate::model::ModelList;

use crate::providers::gemini::GeminiConfig;
use crate::providers::gemini::api;
use crate::providers::gemini::batches::{self, BatchRequest, Batches};
use crate::providers::gemini::cached_content::CachedContents;
use crate::providers::gemini::completion::GenerateContent;
use crate::providers::gemini::count_tokens::CountTokens;
use crate::providers::gemini::embedding::Embeddings;
use crate::providers::gemini::files::{FileRequest, Files};
#[cfg(feature = "image")]
use crate::providers::gemini::image_generation::Images;
use crate::providers::gemini::interactions_api::{InteractionResume, Interactions};
use crate::providers::gemini::transcription::Transcriptions;

http_client!(
    /// Gemini: its [`GeminiConfig`] on a transport. Every model it builds
    /// sends through that transport.
    Gemini,
    GeminiConfig
);

impl Gemini {
    /// Gemini with `api_key` and the public base URL, on the shared reqwest
    /// client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn new(api_key: impl Into<crate::wire::Secret>) -> Self {
        GeminiConfig::new(api_key).client()
    }

    /// Gemini from `GEMINI_API_KEY`, on the shared reqwest client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn from_env() -> Result<Self, crate::client::env::EnvError> {
        Ok(GeminiConfig::from_env()?.client())
    }

    /// The `generateContent` / `streamGenerateContent` model for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Model<GenerateContent> {
        self.model(self.config.completion(model))
    }

    /// The Interactions API model for `model`.
    pub fn interactions(&self, model: impl Into<String>) -> Model<Interactions> {
        self.model(self.config.interactions(model))
    }

    /// The `batchEmbedContents` embedding model for `model`, asking for
    /// `ndims` output dimensions.
    pub fn embedding(&self, model: impl Into<String>, ndims: usize) -> Model<Embeddings> {
        self.model(self.config.embedding(model, ndims))
    }

    /// The audio transcription model for `model`.
    pub fn transcription(&self, model: impl Into<String>) -> Model<Transcriptions> {
        self.model(self.config.transcription(model))
    }

    /// The image-generation model for `model`.
    #[cfg(feature = "image")]
    #[cfg_attr(docsrs, doc(cfg(feature = "image")))]
    pub fn image_generation(&self, model: impl Into<String>) -> Model<Images> {
        self.model(self.config.image_generation(model))
    }

    /// Gemini's explicit context cache (`cachedContents`).
    pub fn cached_contents(&self) -> Model<CachedContents> {
        self.model(self.config.cached_contents())
    }

    /// The Files API.
    pub fn files(&self) -> Model<Files> {
        self.model(self.config.files())
    }

    /// Batch mode.
    pub fn batches(&self) -> Model<Batches> {
        self.model(self.config.batches())
    }

    /// The model that retrieves the interaction `interaction_id`, or resumes
    /// its stream. A unary call fetches the current resource; the caller
    /// controls repeated polling.
    pub fn interaction(&self, interaction_id: impl Into<String>) -> Model<InteractionResume> {
        self.model(self.config.interaction(interaction_id))
    }

    /// [`Self::interaction`], resuming a streamed read after the last event
    /// the consumer saw.
    pub fn interaction_resumed(
        &self,
        interaction_id: impl Into<String>,
        last_event_id: Option<&str>,
    ) -> Model<InteractionResume> {
        self.model(
            self.config
                .interaction_resumed(interaction_id, last_event_id),
        )
    }

    /// The models this API key can use, every page followed.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        self.model(self.config.models()).list().await
    }

    /// Check that the provider accepts the configured key. A 401 or 403
    /// reply is [`ProviderError::InvalidAuthentication`].
    pub async fn verify(&self) -> Result<(), ProviderError> {
        self.model(self.config.verify()).verify().await
    }
}

impl<T> Model<GenerateContent, T>
where
    T: Transport<CountTokens>,
{
    /// The tokens `request` would send, as Gemini counts them: this model's
    /// settings and cache included.
    pub async fn count_tokens(
        &self,
        request: CompletionRequest,
    ) -> Result<api::CountTokensResponse, ProviderError> {
        Model::new(
            CountTokens {
                generate: self.wire.clone(),
            },
            self.transport.clone(),
        )
        .call(request)
        .await
    }
}

impl<T> Model<Files, T>
where
    T: Transport<Files>,
{
    /// Upload `bytes` of `mime_type`. Use the returned file's `uri` in a
    /// message as [`DocumentSourceKind::FileId`](crate::message::DocumentSourceKind::FileId);
    /// a video is usable once its `state` is `ACTIVE`.
    pub async fn upload(
        &self,
        bytes: Vec<u8>,
        mime_type: impl Into<String>,
        display_name: Option<&str>,
    ) -> Result<api::File, ProviderError> {
        self.call(FileRequest::Upload {
            bytes,
            mime_type: mime_type.into(),
            display_name: display_name.map(str::to_owned),
        })
        .await?
        .file()
    }

    /// Read a file by name.
    pub async fn get(&self, name: &str) -> Result<api::File, ProviderError> {
        self.call(FileRequest::Get(name.to_owned())).await?.file()
    }

    /// Every file this key can see, every page followed.
    pub async fn list(&self) -> Result<Vec<api::File>, ProviderError> {
        let pages = crate::driver::follow_cursors(self.name(), "files", |page_token| async move {
            let reply = self.call(FileRequest::List { page_token }).await?;
            let next = reply.next_page_token();
            Ok((reply.entries()?, next))
        })
        .await?;
        Ok(pages.into_iter().flatten().collect())
    }

    /// Delete a file by name.
    pub async fn delete(&self, name: &str) -> Result<(), ProviderError> {
        self.call(FileRequest::Delete(name.to_owned())).await?;
        Ok(())
    }
}

impl<T> Model<Batches, T>
where
    T: Transport<Batches>,
{
    /// Start a batch of `requests`, each rendered as `model` would send it.
    /// Results are keyed by position in each response's `metadata.key`.
    pub async fn create(
        &self,
        model: &GenerateContent,
        display_name: impl Into<String>,
        requests: impl IntoIterator<Item = CompletionRequest>,
    ) -> Result<api::Operation, ProviderError> {
        let batch = batches::batch(model, display_name, requests)?;
        self.call(BatchRequest::Create {
            model: model.model.clone(),
            batch: Box::new(batch),
        })
        .await?
        .operation()
    }

    /// Read a batch by name; its `response` holds the results once `done`.
    pub async fn get(&self, name: &str) -> Result<api::Operation, ProviderError> {
        self.call(BatchRequest::Get(name.to_owned()))
            .await?
            .operation()
    }

    /// Every batch this key can see, every page followed.
    pub async fn list(&self) -> Result<Vec<api::Operation>, ProviderError> {
        let pages =
            crate::driver::follow_cursors(self.name(), "batches", |page_token| async move {
                let reply = self.call(BatchRequest::List { page_token }).await?;
                let next = reply.next_page_token();
                Ok((reply.entries()?, next))
            })
            .await?;
        Ok(pages.into_iter().flatten().collect())
    }

    /// Cancel a running batch.
    pub async fn cancel(&self, name: &str) -> Result<(), ProviderError> {
        self.call(BatchRequest::Cancel(name.to_owned())).await?;
        Ok(())
    }

    /// Delete a batch.
    pub async fn delete(&self, name: &str) -> Result<(), ProviderError> {
        self.call(BatchRequest::Delete(name.to_owned())).await?;
        Ok(())
    }
}

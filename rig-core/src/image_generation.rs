//! Everything related to core image generation abstractions in Rig.
//! Rig allows calling a number of different providers (that support image generation) using the [ImageGenerationModel] trait.
use crate::{client::image_generation::ImageGenerationModelHandle, http_client};
use futures::future::BoxFuture;
use serde_json::Value;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ImageGenerationError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] http_client::Error),

    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Error building the transcription request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    /// Error parsing the transcription response
    #[error("ResponseError: {0}")]
    ResponseError(String),

    /// Error returned by the transcription model provider
    #[error("ProviderError: {0}")]
    ProviderError(String),
}
pub trait ImageGeneration<M>
where
    M: ImageGenerationModel,
{
    /// Generates a transcription request builder for the given `file`.
    /// This function is meant to be called by the user to further customize the
    /// request at transcription time before sending it.
    ///
    /// ❗IMPORTANT: The type that implements this trait might have already
    /// populated fields in the builder (the exact fields depend on the type).
    /// For fields that have already been set by the model, calling the corresponding
    /// method on the builder will overwrite the value set by the model.
    fn image_generation(
        &self,
        prompt: &str,
        size: &(u32, u32),
    ) -> impl std::future::Future<
        Output = Result<ImageGenerationRequestBuilder<M>, ImageGenerationError>,
    > + Send;
}

/// A unified response for a model image generation, returning both the image and the raw response.
#[derive(Debug)]
pub struct ImageGenerationResponse<T> {
    pub image: Vec<u8>,
    pub response: T,
}

pub trait ImageGenerationModel: Clone + Send + Sync {
    type Response: Send + Sync;

    fn image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> impl std::future::Future<
        Output = Result<ImageGenerationResponse<Self::Response>, ImageGenerationError>,
    > + Send;

    fn image_generation_request(&self) -> ImageGenerationRequestBuilder<Self> {
        ImageGenerationRequestBuilder::new(self.clone())
    }
}

pub trait ImageGenerationModelDyn: Send + Sync {
    fn image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> BoxFuture<'_, Result<ImageGenerationResponse<()>, ImageGenerationError>>;

    fn image_generation_request(
        &self,
    ) -> ImageGenerationRequestBuilder<ImageGenerationModelHandle<'_>>;
}

impl<T> ImageGenerationModelDyn for T
where
    T: ImageGenerationModel,
{
    fn image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> BoxFuture<'_, Result<ImageGenerationResponse<()>, ImageGenerationError>> {
        Box::pin(async {
            let resp = self.image_generation(request).await;
            resp.map(|r| ImageGenerationResponse {
                image: r.image,
                response: (),
            })
        })
    }

    fn image_generation_request(
        &self,
    ) -> ImageGenerationRequestBuilder<ImageGenerationModelHandle<'_>> {
        ImageGenerationRequestBuilder::new(ImageGenerationModelHandle {
            inner: Arc::new(self.clone()),
        })
    }
}

/// An image generation request.
#[non_exhaustive]
pub struct ImageGenerationRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub additional_params: Option<Value>,
}

/// A builder for `ImageGenerationRequest`.
/// Can be sent to a model provider.
#[non_exhaustive]
pub struct ImageGenerationRequestBuilder<M>
where
    M: ImageGenerationModel,
{
    model: M,
    prompt: String,
    width: u32,
    height: u32,
    additional_params: Option<Value>,
}

impl<M> ImageGenerationRequestBuilder<M>
where
    M: ImageGenerationModel,
{
    pub fn new(model: M) -> Self {
        Self {
            model,
            prompt: "".to_string(),
            height: 256,
            width: 256,
            additional_params: None,
        }
    }

    /// Sets the prompt for the image generation request
    pub fn prompt(mut self, prompt: &str) -> Self {
        self.prompt = prompt.to_string();
        self
    }

    /// The width of the generated image
    pub fn width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    /// The height of the generated image
    pub fn height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    /// Adds additional parameters to the image generation request.
    pub fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    pub fn build(self) -> ImageGenerationRequest {
        ImageGenerationRequest {
            prompt: self.prompt,
            width: self.width,
            height: self.height,
            additional_params: self.additional_params,
        }
    }

    pub async fn send(self) -> Result<ImageGenerationResponse<M::Response>, ImageGenerationError> {
        let model = self.model.clone();

        model.image_generation(self.build()).await
    }
}

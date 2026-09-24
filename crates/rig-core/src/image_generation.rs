//! Image-generation requests, normalized responses, and model interfaces.
//!
//! ```no_run
//! use rig_core::image_generation::{ImageGenerationModel, ImageGenerationRequestBuilder};
//!
//! # async fn example(model: impl ImageGenerationModel) -> Result<(), Box<dyn std::error::Error>> {
//! let response = ImageGenerationRequestBuilder::new(model, "A mountain lake")
//!     .width(1024).height(1024).send().await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
use crate::error::ProviderError;
use crate::response::Response;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde_json::Value;
use std::sync::Arc;

/// The generated image, decoded to bytes, and the metadata the provider
/// reported.
pub type ImageGenerationResponse = Response<Vec<u8>>;

/// Generates images from prompts. Only [`Self::image_generation_request`]
/// requires cloning; `Arc<M>` forwards generation calls.
pub trait ImageGenerationModel: WasmCompatSend + WasmCompatSync {
    fn image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> impl std::future::Future<Output = Result<ImageGenerationResponse, ProviderError>> + WasmCompatSend;

    /// Creates a request builder for `prompt`.
    fn image_generation_request(
        &self,
        prompt: impl Into<String>,
    ) -> ImageGenerationRequestBuilder<Self>
    where
        Self: Sized + Clone,
    {
        ImageGenerationRequestBuilder::new(self.clone(), prompt)
    }
}

impl<M> ImageGenerationModel for Arc<M>
where
    M: ImageGenerationModel,
{
    fn image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> impl std::future::Future<Output = Result<ImageGenerationResponse, ProviderError>> + WasmCompatSend
    {
        (**self).image_generation(request)
    }
}

pub struct ImageGenerationRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub additional_params: Option<Value>,
}

/// Builds an image request for a prompt. Defaults to 256 by 256 pixels;
/// supported dimensions depend on the provider.
pub struct ImageGenerationRequestBuilder<M> {
    model: M,
    request: ImageGenerationRequest,
}

impl<M> ImageGenerationRequestBuilder<M> {
    /// A request for `prompt`.
    pub fn new(model: M, prompt: impl Into<String>) -> Self {
        Self {
            model,
            request: ImageGenerationRequest {
                prompt: prompt.into(),
                width: 256,
                height: 256,
                additional_params: None,
            },
        }
    }

    /// The width of the generated image.
    pub fn width(mut self, width: u32) -> Self {
        self.request.width = width;
        self
    }

    /// The height of the generated image.
    pub fn height(mut self, height: u32) -> Self {
        self.request.height = height;
        self
    }

    /// Merges provider-specific parameters over earlier ones, key by key for
    /// JSON objects; `None` clears existing parameters.
    pub fn additional_params(mut self, params: impl Into<Option<Value>>) -> Self {
        self.request.additional_params =
            crate::json_utils::merge_params(self.request.additional_params.take(), params.into());
        self
    }

    /// Builds the image generation request.
    pub fn build(self) -> ImageGenerationRequest {
        self.request
    }
}

impl<M: ImageGenerationModel> ImageGenerationRequestBuilder<M> {
    /// Sends the request to the model and returns the generated image.
    pub async fn send(self) -> Result<ImageGenerationResponse, ProviderError> {
        self.model.image_generation(self.request).await
    }
}

#[cfg(test)]
mod builder_tests;
#[cfg(test)]
mod provider_response_tests;

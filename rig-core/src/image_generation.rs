use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ImageGenerationError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] reqwest::Error),

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
pub trait ImageGeneration<M: ImageGenerationModel> {
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

pub struct ImageGenerationRequest {
    pub prompt: String,
    pub size: (u32, u32),
    pub additional_params: Option<Value>,
}

pub struct ImageGenerationRequestBuilder<M: ImageGenerationModel> {
    model: M,
    prompt: String,
    size: (u32, u32),
    additional_params: Option<Value>,
}

impl<M: ImageGenerationModel> ImageGenerationRequestBuilder<M> {
    pub fn new(model: M) -> Self {
        Self {
            model,
            prompt: "".to_string(),
            size: (256, 256),
            additional_params: None,
        }
    }

    pub fn prompt(mut self, prompt: &str) -> Self {
        self.prompt = prompt.to_string();
        self
    }

    pub fn size(mut self, size: (u32, u32)) -> Self {
        self.size = size;
        self
    }
    
    pub fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }
    
    pub fn build(self) -> ImageGenerationRequest {
        ImageGenerationRequest {
            prompt: self.prompt,
            size: self.size,
            additional_params: self.additional_params,
        }
    }
    
    pub async fn send(self) -> Result<ImageGenerationResponse<M::Response>, ImageGenerationError> {
        let model = self.model.clone();
        
        model.image_generation(self.build()).await
    }
}

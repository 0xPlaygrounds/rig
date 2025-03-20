use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioGenerationError {
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
pub trait AudioGeneration<M: AudioGenerationModel> {
    /// Generates an audio generation request builder for the given `text` and `voice`.
    /// This function is meant to be called by the user to further customize the
    /// request at generation time before sending it.
    ///
    /// ❗IMPORTANT: The type that implements this trait might have already
    /// populated fields in the builder (the exact fields depend on the type).
    /// For fields that have already been set by the model, calling the corresponding
    /// method on the builder will overwrite the value set by the model.
    fn audio_generation(
        &self,
        text: &str,
        voice: &str,
    ) -> impl std::future::Future<
        Output = Result<AudioGenerationRequestBuilder<M>, AudioGenerationError>,
    > + Send;
}

pub struct AudioGenerationResponse<T> {
    pub audio: Vec<u8>,
    pub response: T,
}

pub trait AudioGenerationModel: Clone + Send + Sync {
    type Response: Send + Sync;

    fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> impl std::future::Future<
        Output = Result<AudioGenerationResponse<Self::Response>, AudioGenerationError>,
    > + Send;

    fn audio_generation_request(&self) -> AudioGenerationRequestBuilder<Self> {
        AudioGenerationRequestBuilder::new(self.clone())
    }
}

pub struct AudioGenerationRequest {
    pub text: String,
    pub voice: String,
    pub speed: f32,
    pub additional_params: Option<Value>,
}

pub struct AudioGenerationRequestBuilder<M: AudioGenerationModel> {
    model: M,
    text: String,
    voice: String,
    speed: f32,
    additional_params: Option<Value>,
}

impl<M: AudioGenerationModel> AudioGenerationRequestBuilder<M> {
    pub fn new(model: M) -> Self {
        Self {
            model,
            text: "".to_string(),
            voice: "".to_string(),
            speed: 1.0,
            additional_params: None,
        }
    }

    /// Sets the text for the audio generation request
    pub fn text(mut self, text: &str) -> Self {
        self.text = text.to_string();
        self
    }

    /// The voice of the generated audio
    pub fn voice(mut self, voice: &str) -> Self {
        self.voice = voice.to_string();
        self
    }

    /// The speed of the generated audio
    pub fn speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    /// Adds additional parameters to the audio generation request.
    pub fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    pub fn build(self) -> AudioGenerationRequest {
        AudioGenerationRequest {
            text: self.text,
            voice: self.voice,
            speed: self.speed,
            additional_params: self.additional_params,
        }
    }

    pub async fn send(self) -> Result<AudioGenerationResponse<M::Response>, AudioGenerationError> {
        let model = self.model.clone();

        model.audio_generation(self.build()).await
    }
}

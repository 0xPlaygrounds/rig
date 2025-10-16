use crate::client::{AsTranscription, ProviderClient};
use crate::transcription::{
    TranscriptionError, TranscriptionModel, TranscriptionModelDyn, TranscriptionRequest,
    TranscriptionResponse,
};
use std::sync::Arc;

/// A provider client with audio transcription capabilities.
///
/// This trait extends [`ProviderClient`] to provide audio-to-text transcription functionality.
/// Providers that implement this trait can create transcription models for converting
/// audio files to text.
///
/// # When to Implement
///
/// Implement this trait for provider clients that support:
/// - Audio to text transcription
/// - Speech recognition
/// - Multiple audio format support
/// - Language detection and translation
///
/// # Examples
///
/// ```no_run
/// use rig::prelude::*;
/// use rig::providers::openai::{Client, self};
/// use rig::transcription::TranscriptionRequest;
/// use std::fs;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = Client::new("api-key");
///
/// // Create a transcription model
/// let model = client.transcription_model(openai::WHISPER_1);
///
/// // Read audio file
/// let audio_data = fs::read("audio.mp3")?;
///
/// // Transcribe audio
/// let response = model.transcription(TranscriptionRequest {
///     data: audio_data,
///     filename: "audio.mp3".to_string(),
///     language: "en".to_string(),
///     prompt: None,
///     temperature: None,
///     additional_params: None,
/// }).await?;
///
/// println!("Transcription: {}", response.text);
/// # Ok(())
/// # }
/// ```
///
/// # See Also
///
/// - [`crate::transcription::TranscriptionModel`] - The model trait for transcription operations
/// - [`crate::transcription::TranscriptionRequest`] - Request structure for transcriptions
/// - [`TranscriptionClientDyn`] - Dynamic dispatch version for runtime polymorphism
pub trait TranscriptionClient: ProviderClient + Clone {
    /// The type of TranscriptionModel used by the Client
    type TranscriptionModel: TranscriptionModel;

    /// Creates a transcription model with the specified model identifier.
    ///
    /// This method constructs a transcription model that can convert audio to text.
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier (e.g., "whisper-1", "whisper-large-v3")
    ///
    /// # Returns
    ///
    /// A transcription model that can process audio files.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rig::prelude::*;
    /// use rig::providers::openai::{Client, self};
    /// use rig::transcription::{TranscriptionModel, TranscriptionRequest};
    /// use std::fs;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new("your-api-key");
    /// let model = client.transcription_model(openai::WHISPER_1);
    ///
    /// let audio_data = fs::read("audio.mp3")?;
    /// let response = model.transcription(TranscriptionRequest {
    ///     data: audio_data,
    ///     filename: "audio.mp3".to_string(),
    ///     language: "en".to_string(),
    ///     prompt: None,
    ///     temperature: None,
    ///     additional_params: None,
    /// }).await?;
    ///
    /// println!("Transcription: {}", response.text);
    /// # Ok(())
    /// # }
    /// ```
    fn transcription_model(&self, model: &str) -> Self::TranscriptionModel;
}

/// Dynamic dispatch version of [`TranscriptionClient`].
///
/// This trait provides the same functionality as [`TranscriptionClient`] but returns
/// trait objects instead of associated types, enabling runtime polymorphism.
/// It is automatically implemented for all types that implement [`TranscriptionClient`].
///
/// # When to Use
///
/// Use this trait when you need to work with transcription clients of different types
/// at runtime, such as in the [`DynClientBuilder`](crate::client::builder::DynClientBuilder).
pub trait TranscriptionClientDyn: ProviderClient {
    /// Creates a boxed transcription model with the specified model identifier.
    ///
    /// Returns a trait object that can be used for dynamic dispatch.
    fn transcription_model<'a>(&self, model: &str) -> Box<dyn TranscriptionModelDyn + 'a>;
}

impl<M, T> TranscriptionClientDyn for T
where
    T: TranscriptionClient<TranscriptionModel = M>,
    M: TranscriptionModel + 'static,
{
    fn transcription_model<'a>(&self, model: &str) -> Box<dyn TranscriptionModelDyn + 'a> {
        Box::new(self.transcription_model(model))
    }
}

impl<T> AsTranscription for T
where
    T: TranscriptionClientDyn + Clone + 'static,
{
    fn as_transcription(&self) -> Option<Box<dyn TranscriptionClientDyn>> {
        Some(Box::new(self.clone()))
    }
}

/// A dynamic handle for transcription models enabling trait object usage.
///
/// This struct wraps a [`TranscriptionModel`] in a way that allows it to be used
/// as a trait object in generic contexts. It uses `Arc` internally for efficient cloning.
///
/// This type is primarily used internally by the dynamic client builder for
/// runtime polymorphism over different transcription model implementations.
#[derive(Clone)]
pub struct TranscriptionModelHandle<'a> {
    /// The inner dynamic transcription model.
    pub inner: Arc<dyn TranscriptionModelDyn + 'a>,
}

impl TranscriptionModel for TranscriptionModelHandle<'_> {
    type Response = ();

    async fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse<Self::Response>, TranscriptionError> {
        self.inner.transcription(request).await
    }
}

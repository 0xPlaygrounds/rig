use super::{Client, ModelTransport, Provider};
use crate::transcription::TranscriptionModel;

/// A provider client with transcription capabilities.
/// Clone is required for conversions between client types.
pub trait TranscriptionClient {
    /// The type of TranscriptionModel used by the Client
    type TranscriptionModel: TranscriptionModel;

    /// Create a transcription model with the given name.
    ///
    /// # Example with OpenAI
    /// ```ignore
    /// use rig_core::prelude::TranscriptionClient;
    /// use rig_core::providers::openai::{Client, self};
    ///
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key")?;
    ///
    /// let whisper = openai.transcription_model(openai::WHISPER_1);
    /// # Ok(())
    /// # }
    /// ```
    fn transcription_model(&self, model: impl Into<String>) -> Self::TranscriptionModel;
}

/// A [`Provider`] that offers transcription models. Implementing this is what
/// makes [`TranscriptionClient`] available on `Client<Self, H>`.
pub trait HasTranscription: Provider {
    /// The concrete transcription model built over transport `H`.
    type Model<H>: TranscriptionModel
    where
        H: ModelTransport;

    /// Build the transcription model `model` from `client`.
    fn transcription_model<H>(client: &Client<Self, H>, model: String) -> Self::Model<H>
    where
        H: ModelTransport;
}

impl<P, H> TranscriptionClient for Client<P, H>
where
    P: HasTranscription,
    H: ModelTransport,
{
    type TranscriptionModel = P::Model<H>;

    fn transcription_model(&self, model: impl Into<String>) -> Self::TranscriptionModel {
        P::transcription_model(self, model.into())
    }
}

use crate::transcription::TranscriptionModel;

/// A provider client with transcription capabilities.
/// Clone is required for conversions between client types.
pub trait TranscriptionClient {
    /// The type of TranscriptionModel used by the Client
    type TranscriptionModel: TranscriptionModel;

    /// Create a transcription model with the given name.
    ///
    /// # Example with OpenAI
    /// ```no_run
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

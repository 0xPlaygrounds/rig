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

/// Construction hook for the blanket [`TranscriptionClient`] implementation over
/// [`crate::client::Client`] — the transcription twin of
/// [`crate::client::ConstructCompletionModel`].
///
/// Public for the same reason: an out-of-tree provider extension built on the
/// generic `Client<Ext, H>` cannot implement [`TranscriptionClient`] for that foreign
/// type (orphan rule), so it implements this trait on its own model type and
/// the blanket implementation supplies the constructor. Providers with their
/// own client type implement [`TranscriptionClient`] directly and never need this.
pub trait ConstructTranscriptionModel<C>: Sized {
    /// Build this model from its provider client and a model identifier.
    fn construct(client: &C, model: String) -> Self;
}

#[cfg(feature = "audio")]
mod audio {
    use crate::audio_generation::AudioGenerationModel;

    /// A provider client with audio generation capabilities.
    /// Clone is required for conversions between client types.
    pub trait AudioGenerationClient {
        /// The AudioGenerationModel used by the Client
        type AudioGenerationModel: AudioGenerationModel;

        /// Create an audio generation model with the given name.
        ///
        /// # Example
        /// ```ignore
        /// use rig_core::prelude::AudioGenerationClient;
        /// use rig_core::providers::openai::{Client, self};
        ///
        /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
        /// // Initialize the OpenAI client
        /// let openai = Client::new("your-open-ai-api-key")?;
        ///
        /// let tts = openai.audio_generation_model(openai::TTS_1);
        /// # Ok(())
        /// # }
        /// ```
        fn audio_generation_model(&self, model: impl Into<String>) -> Self::AudioGenerationModel;
    }

    /// Construction hook for the blanket [`AudioGenerationClient`] implementation over
    /// [`crate::client::Client`] — the audio generation twin of
    /// [`crate::client::ConstructCompletionModel`].
    ///
    /// Public for the same reason: an out-of-tree provider extension built on the
    /// generic `Client<Ext, H>` cannot implement [`AudioGenerationClient`] for that foreign
    /// type (orphan rule), so it implements this trait on its own model type and
    /// the blanket implementation supplies the constructor. Providers with their
    /// own client type implement [`AudioGenerationClient`] directly and never need this.
    pub trait ConstructAudioGenerationModel<C>: Sized {
        /// Build this model from its provider client and a model identifier.
        fn construct(client: &C, model: String) -> Self;
    }
}

#[cfg(feature = "audio")]
pub use audio::*;

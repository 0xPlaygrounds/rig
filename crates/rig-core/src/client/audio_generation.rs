#[cfg(feature = "audio")]
mod audio {
    use crate::audio_generation::AudioGenerationModel;

    /// A provider client with audio generation capabilities.
    /// Clone is required for conversions between client types.
    pub trait AudioGenerationClient {
        /// The AudioGenerationModel used by the Client
        type AudioGenerationModel: AudioGenerationModel<Client = Self>;

        /// Create an audio generation model with the given name.
        ///
        /// # Example
        /// ```no_run
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
        fn audio_generation_model(&self, model: impl Into<String>) -> Self::AudioGenerationModel {
            Self::AudioGenerationModel::make(self, model)
        }
    }
}

#[cfg(feature = "audio")]
pub use audio::*;

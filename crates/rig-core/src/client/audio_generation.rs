#[cfg(feature = "audio")]
mod audio {
    use crate::audio_generation::AudioGenerationModel;
    use crate::client::{Client, ModelTransport, Provider};

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

    /// A [`Provider`] that offers audio generation models. Implementing this
    /// is what makes [`AudioGenerationClient`] available on `Client<Self, H>`.
    pub trait HasAudioGeneration: Provider {
        /// The concrete audio generation model built over transport `H`.
        type Model<H>: AudioGenerationModel
        where
            H: ModelTransport;

        /// Build the audio generation model `model` from `client`.
        fn audio_generation_model<H>(client: &Client<Self, H>, model: String) -> Self::Model<H>
        where
            H: ModelTransport;
    }

    impl<P, H> AudioGenerationClient for Client<P, H>
    where
        P: HasAudioGeneration,
        H: ModelTransport,
    {
        type AudioGenerationModel = P::Model<H>;

        fn audio_generation_model(&self, model: impl Into<String>) -> Self::AudioGenerationModel {
            P::audio_generation_model(self, model.into())
        }
    }
}

#[cfg(feature = "audio")]
pub use audio::*;

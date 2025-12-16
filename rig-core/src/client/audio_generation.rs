#[cfg(feature = "audio")]
mod audio {
    #[allow(deprecated)]
    use crate::audio_generation::AudioGenerationModelDyn;
    use crate::audio_generation::{
        AudioGenerationError, AudioGenerationModel, AudioGenerationRequest, AudioGenerationResponse,
    };
    use crate::client::Nothing;
    use std::future::Future;
    use std::sync::Arc;

    /// A provider client with audio generation capabilities.
    /// Clone is required for conversions between client types.
    pub trait AudioGenerationClient {
        /// The AudioGenerationModel used by the Client
        type AudioGenerationModel: AudioGenerationModel<Client = Self>;

        /// Create an audio generation model with the given name.
        ///
        /// # Example
        /// ```
        /// use rig::providers::openai::{Client, self};
        ///
        /// // Initialize the OpenAI client
        /// let openai = Client::new("your-open-ai-api-key");
        ///
        /// let tts = openai.audio_generation_model(openai::TTS_1);
        /// ```
        fn audio_generation_model(&self, model: impl Into<String>) -> Self::AudioGenerationModel {
            Self::AudioGenerationModel::make(self, model)
        }
    }

    #[allow(deprecated)]
    #[deprecated(
        since = "0.25.0",
        note = "`DynClientBuilder` and related features have been deprecated and will be removed in a future release. In this case, use `ImageGenerationModel` instead."
    )]
    pub trait AudioGenerationClientDyn {
        fn audio_generation_model<'a>(&self, model: &str) -> Box<dyn AudioGenerationModelDyn + 'a>;
    }

    #[allow(deprecated)]
    impl<T, M> AudioGenerationClientDyn for T
    where
        T: AudioGenerationClient<AudioGenerationModel = M>,
        M: AudioGenerationModel + 'static,
    {
        fn audio_generation_model<'a>(&self, model: &str) -> Box<dyn AudioGenerationModelDyn + 'a> {
            Box::new(self.audio_generation_model(model))
        }
    }

    #[deprecated(
        since = "0.25.0",
        note = "`DynClientBuilder` and related features have been deprecated and will be removed in a future release. In this case, use `ImageGenerationModel` instead."
    )]
    /// Wraps a AudioGenerationModel in a dyn-compatible way for AudioGenerationRequestBuilder.
    #[derive(Clone)]
    pub struct AudioGenerationModelHandle<'a> {
        #[allow(deprecated)]
        pub(crate) inner: Arc<dyn AudioGenerationModelDyn + 'a>,
    }

    #[allow(deprecated)]
    impl AudioGenerationModel for AudioGenerationModelHandle<'_> {
        type Response = ();
        type Client = Nothing;

        /// **PANICS**: DynClientBuilder and related features (like this model handle) are being phased out,
        /// during this transition period some methods will panic when called
        fn make(_: &Self::Client, _: impl Into<String>) -> Self {
            panic!(
                "Function should be unreachable as Self can only be constructed from another 'AudioGenerationModel'"
            )
        }

        fn audio_generation(
            &self,
            request: AudioGenerationRequest,
        ) -> impl Future<
            Output = Result<AudioGenerationResponse<Self::Response>, AudioGenerationError>,
        > + Send {
            self.inner.audio_generation(request)
        }
    }
}

#[cfg(feature = "audio")]
pub use audio::*;

#[cfg(feature = "audio")]
mod audio {
    use crate::audio_generation::{
        AudioGenerationError, AudioGenerationModel, AudioGenerationModelDyn,
        AudioGenerationRequest, AudioGenerationResponse,
    };
    use crate::client::{AsAudioGeneration, ProviderClient};
    use std::future::Future;
    use std::sync::Arc;

    pub trait AudioGenerationClient: ProviderClient + Clone {
        type AudioGenerationModel: AudioGenerationModel;
        fn audio_generation_model(&self, model: &str) -> Self::AudioGenerationModel;
    }

    pub trait AudioGenerationClientDyn: ProviderClient {
        fn audio_generation_model<'a>(
            &self,
            model: &'a str,
        ) -> Box<dyn AudioGenerationModelDyn + 'a>;
    }

    impl<T: AudioGenerationClient<AudioGenerationModel = M>, M: AudioGenerationModel + 'static>
        AudioGenerationClientDyn for T
    {
        fn audio_generation_model<'a>(
            &self,
            model: &'a str,
        ) -> Box<dyn AudioGenerationModelDyn + 'a> {
            Box::new(self.audio_generation_model(model))
        }
    }

    impl<T: AudioGenerationClientDyn + Clone + 'static> AsAudioGeneration for T {
        fn as_audio_generation(&self) -> Option<Box<dyn AudioGenerationClientDyn>> {
            Some(Box::new(self.clone()))
        }
    }

    #[derive(Clone)]
    pub struct AudioGenerationModelHandle<'a> {
        pub(crate) inner: Arc<dyn AudioGenerationModelDyn + 'a>,
    }
    impl AudioGenerationModel for AudioGenerationModelHandle<'_> {
        type Response = ();

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

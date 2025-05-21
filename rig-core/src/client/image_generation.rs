#[cfg(feature = "image")]
mod image {
    use crate::client::{AsImageGeneration, ProviderClient};
    use crate::image_generation::{
        ImageGenerationError, ImageGenerationModel, ImageGenerationModelDyn,
        ImageGenerationRequest, ImageGenerationResponse,
    };
    use std::future::Future;
    use std::sync::Arc;
    pub trait ImageGenerationClient: ProviderClient {
        type ImageGenerationModel: ImageGenerationModel;
        fn image_generation_model(&self, model: &str) -> Self::ImageGenerationModel;
    }

    pub trait ImageGenerationClientDyn: ProviderClient {
        fn image_generation_model<'a>(
            &'a self,
            model: &'a str,
        ) -> Box<dyn ImageGenerationModelDyn + 'a>;
    }

    impl<T: ImageGenerationClient> ImageGenerationClientDyn for T {
        fn image_generation_model<'a>(
            &'a self,
            model: &'a str,
        ) -> Box<dyn ImageGenerationModelDyn + 'a> {
            Box::new(self.image_generation_model(model))
        }
    }

    impl<T: ImageGenerationClientDyn> AsImageGeneration for T {
        fn as_image_generation(&self) -> Option<Box<&dyn ImageGenerationClientDyn>> {
            Some(Box::new(self))
        }
    }

    #[derive(Clone)]
    pub struct ImageGenerationModelHandle<'a> {
        pub(crate) inner: Arc<dyn ImageGenerationModelDyn + 'a>,
    }
    impl ImageGenerationModel for ImageGenerationModelHandle<'_> {
        type Response = ();

        fn image_generation(
            &self,
            request: ImageGenerationRequest,
        ) -> impl Future<
            Output = Result<ImageGenerationResponse<Self::Response>, ImageGenerationError>,
        > + Send {
            self.inner.image_generation(request)
        }
    }
}

#[cfg(feature = "image")]
pub use image::*;

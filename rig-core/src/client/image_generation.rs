#[cfg(feature = "image")]
mod image {
    use crate::client::{AsImageGeneration, ProviderClient};
    use crate::image_generation::{
        ImageGenerationError, ImageGenerationModel, ImageGenerationModelDyn,
        ImageGenerationRequest, ImageGenerationResponse,
    };
    use std::future::Future;
    use std::sync::Arc;

    /// A provider client with image generation capabilities.
    /// Clone is required for conversions between client types.
    pub trait ImageGenerationClient: ProviderClient + Clone {
        /// The ImageGenerationModel used by the Client
        type ImageGenerationModel: ImageGenerationModel;

        /// Create an image generation model with the given name.
        ///
        /// # Example with OpenAI
        /// ```
        /// use rig::prelude::*;
        /// use rig::providers::openai::{Client, self};
        ///
        /// // Initialize the OpenAI client
        /// let openai = Client::new("your-open-ai-api-key");
        ///
        /// let gpt4 = openai.image_generation_model(openai::DALL_E_3);
        /// ```
        fn image_generation_model(&self, model: &str) -> Self::ImageGenerationModel;
    }

    pub trait ImageGenerationClientDyn: ProviderClient {
        /// Create an image generation model with the given name.
        fn image_generation_model<'a>(&self, model: &str) -> Box<dyn ImageGenerationModelDyn + 'a>;
    }

    impl<T: ImageGenerationClient<ImageGenerationModel = M>, M: ImageGenerationModel + 'static>
        ImageGenerationClientDyn for T
    {
        fn image_generation_model<'a>(&self, model: &str) -> Box<dyn ImageGenerationModelDyn + 'a> {
            Box::new(self.image_generation_model(model))
        }
    }

    impl<T: ImageGenerationClientDyn + Clone + 'static> AsImageGeneration for T {
        fn as_image_generation(&self) -> Option<Box<dyn ImageGenerationClientDyn>> {
            Some(Box::new(self.clone()))
        }
    }

    /// Wraps a ImageGenerationModel in a dyn-compatible way for ImageGenerationRequestBuilder.
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

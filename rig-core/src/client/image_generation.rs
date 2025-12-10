#[cfg(feature = "image")]
mod image {
    use crate::client::Nothing;
    #[allow(deprecated)]
    use crate::image_generation::ImageGenerationModelDyn;
    use crate::image_generation::{
        ImageGenerationError, ImageGenerationModel, ImageGenerationRequest, ImageGenerationResponse,
    };
    use std::future::Future;
    use std::sync::Arc;

    /// A provider client with image generation capabilities.
    /// Clone is required for conversions between client types.
    pub trait ImageGenerationClient {
        /// The ImageGenerationModel used by the Client
        type ImageGenerationModel: ImageGenerationModel<Client = Self>;

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
        fn image_generation_model(&self, model: impl Into<String>) -> Self::ImageGenerationModel;

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
        fn custom_image_generation_model(
            &self,
            model: impl Into<String>,
        ) -> Self::ImageGenerationModel {
            Self::ImageGenerationModel::make(self, model)
        }
    }

    #[allow(deprecated)]
    #[deprecated(
        since = "0.25.0",
        note = "`DynClientBuilder` and related features have been deprecated and will be removed in a future release. In this case, use `ImageGenerationClient` instead."
    )]
    pub trait ImageGenerationClientDyn {
        /// Create an image generation model with the given name.
        fn image_generation_model<'a>(&self, model: &str) -> Box<dyn ImageGenerationModelDyn + 'a>;
    }

    #[allow(deprecated)]
    impl<T: ImageGenerationClient<ImageGenerationModel = M>, M: ImageGenerationModel + 'static>
        ImageGenerationClientDyn for T
    {
        fn image_generation_model<'a>(&self, model: &str) -> Box<dyn ImageGenerationModelDyn + 'a> {
            Box::new(self.image_generation_model(model))
        }
    }

    #[deprecated(
        since = "0.25.0",
        note = "`DynClientBuilder` and related features have been deprecated and will be removed in a future release."
    )]
    /// Wraps a ImageGenerationModel in a dyn-compatible way for ImageGenerationRequestBuilder.
    #[derive(Clone)]
    pub struct ImageGenerationModelHandle<'a> {
        #[allow(deprecated)]
        pub(crate) inner: Arc<dyn ImageGenerationModelDyn + 'a>,
    }

    #[allow(deprecated)]
    impl ImageGenerationModel for ImageGenerationModelHandle<'_> {
        type Response = ();
        type Client = Nothing;

        /// **PANICS** if called
        fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
            panic!(
                "'ImageGenerationModel::make' should not be called on 'ImageGenerationModelHandle'"
            )
        }

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

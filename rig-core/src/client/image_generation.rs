#[cfg(feature = "image")]
mod image {
    use crate::client::Nothing;
    use crate::image_generation::{
        ImageGenerationError, ImageGenerationModel, ImageGenerationModelDyn,
        ImageGenerationRequest, ImageGenerationResponse,
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
        fn image_generation_model(
            &self,
            model: <Self::ImageGenerationModel as ImageGenerationModel>::Models,
        ) -> Self::ImageGenerationModel;

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
        fn custom_image_generation_model(&self, model: &str) -> Self::ImageGenerationModel {
            Self::ImageGenerationModel::make_custom(self, model)
        }
    }

    pub trait ImageGenerationClientDyn {
        /// Create an image generation model with the given name.
        fn image_generation_model<'a>(&self, model: &str) -> Box<dyn ImageGenerationModelDyn + 'a>;
    }

    impl<T: ImageGenerationClient<ImageGenerationModel = M>, M: ImageGenerationModel + 'static>
        ImageGenerationClientDyn for T
    {
        fn image_generation_model<'a>(&self, model: &str) -> Box<dyn ImageGenerationModelDyn + 'a> {
            let model = model
                .to_string()
                .try_into()
                .unwrap_or_else(|_| panic!("Invalid model name '{model}'"));

            Box::new(self.image_generation_model(model))
        }
    }

    /// Wraps a ImageGenerationModel in a dyn-compatible way for ImageGenerationRequestBuilder.
    #[derive(Clone)]
    pub struct ImageGenerationModelHandle<'a> {
        pub(crate) inner: Arc<dyn ImageGenerationModelDyn + 'a>,
    }

    impl ImageGenerationModel for ImageGenerationModelHandle<'_> {
        type Response = ();
        type Models = Nothing;
        type Client = Nothing;

        // NOTE: @FayCarsons - This is not ideal, we would ideally have a way to statically prevent
        // anyone from calling this method but that doesn't seem possible without gutting the trait
        // and finding a new wait to implement `ImageGenerationClient` for arbitrary `Client<Ext, H>`
        fn make(_client: &Self::Client, _model: Self::Models) -> Self {
            panic!(
                "'ImageGenerationModel::make' should not be called on 'ImageGenerationModelHandle'"
            )
        }

        fn make_custom(_: &Self::Client, _: &str) -> Self {
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

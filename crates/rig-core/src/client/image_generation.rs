#[cfg(feature = "image")]
mod image {
    use crate::client::{Client, ModelTransport, Provider};
    use crate::image_generation::ImageGenerationModel;

    /// A provider client with image generation capabilities.
    /// Clone is required for conversions between client types.
    pub trait ImageGenerationClient {
        /// The ImageGenerationModel used by the Client
        type ImageGenerationModel: ImageGenerationModel;

        /// Create an image generation model with the given name.
        ///
        /// # Example with OpenAI
        /// ```ignore
        /// use rig_core::prelude::*;
        /// use rig_core::providers::openai::{Client, self};
        ///
        /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
        /// // Initialize the OpenAI client
        /// let openai = Client::new("your-open-ai-api-key")?;
        ///
        /// let gpt4 = openai.image_generation_model(openai::DALL_E_3);
        /// # Ok(())
        /// # }
        /// ```
        fn image_generation_model(&self, model: impl Into<String>) -> Self::ImageGenerationModel;
    }

    /// A [`Provider`] that offers image generation models. Implementing this
    /// is what makes [`ImageGenerationClient`] available on `Client<Self, H>`.
    pub trait HasImageGeneration: Provider {
        /// The concrete image generation model built over transport `H`.
        type Model<H>: ImageGenerationModel
        where
            H: ModelTransport;

        /// Build the image generation model `model` from `client`.
        fn image_generation_model<H>(client: &Client<Self, H>, model: String) -> Self::Model<H>
        where
            H: ModelTransport;
    }

    impl<P, H> ImageGenerationClient for Client<P, H>
    where
        P: HasImageGeneration,
        H: ModelTransport,
    {
        type ImageGenerationModel = P::Model<H>;

        fn image_generation_model(&self, model: impl Into<String>) -> Self::ImageGenerationModel {
            P::image_generation_model(self, model.into())
        }
    }
}

#[cfg(feature = "image")]
pub use image::*;

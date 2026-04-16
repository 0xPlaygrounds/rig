#[cfg(feature = "image")]
mod image {
    use crate::image_generation::ImageGenerationModel;

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
}

#[cfg(feature = "image")]
pub use image::*;

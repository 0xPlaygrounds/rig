#[cfg(feature = "image")]
mod image {
    use crate::image_generation::ImageGenerationModel;

    /// A provider client with image generation capabilities.
    /// Clone is required for conversions between client types.
    pub trait ImageGenerationClient {
        /// The ImageGenerationModel used by the Client
        type ImageGenerationModel: ImageGenerationModel;

        /// Create an image generation model with the given name.
        ///
        /// # Example with OpenAI
        /// ```no_run
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

    /// Construction hook for the blanket [`ImageGenerationClient`] implementation over
    /// [`crate::client::Client`] — the image generation twin of
    /// [`crate::client::ConstructCompletionModel`].
    ///
    /// Public for the same reason: an out-of-tree provider extension built on the
    /// generic `Client<Ext, H>` cannot implement [`ImageGenerationClient`] for that foreign
    /// type (orphan rule), so it implements this trait on its own model type and
    /// the blanket implementation supplies the constructor. Providers with their
    /// own client type implement [`ImageGenerationClient`] directly and never need this.
    pub trait ConstructImageGenerationModel<C>: Sized {
        /// Build this model from its provider client and a model identifier.
        fn construct(client: &C, model: String) -> Self;
    }
}

#[cfg(feature = "image")]
pub use image::*;

pub use crate::client::ProviderClient;
pub use crate::client::completion::CompletionClient;
pub use crate::client::embeddings::EmbeddingsClient;
pub use crate::client::transcription::TranscriptionClient;
pub use crate::client::verify::{VerifyClient, VerifyError};

#[cfg(feature = "image")]
pub use crate::client::image_generation::ImageGenerationClient;

#[cfg(feature = "audio")]
pub use crate::client::audio_generation::AudioGenerationClient;

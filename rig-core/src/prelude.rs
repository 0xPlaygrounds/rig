pub use crate::client::{
    AsAudioGeneration, AsCompletion, AsEmbeddings, AsImageGeneration, AsTranscription,
    ProviderClient,
};

pub use crate::client::completion::CompletionClient;

pub use crate::client::embeddings::EmbeddingsClient;

pub use crate::client::transcription::TranscriptionClient;

#[cfg(feature = "image")]
pub use crate::client::image_generation::ImageGenerationClient;

#[cfg(feature = "audio")]
pub use crate::client::audio_generation::AudioGenerationClient;

pub use crate::client::{VerifyClient, VerifyError};

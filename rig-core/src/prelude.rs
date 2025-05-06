pub use crate::client::{
    AsAudioGeneration, AsCompletion, AsEmbeddings, AsImageGeneration, AsTranscription,
    CompletionClient, EmbeddingsClient, ProviderClient, TranscriptionClient,
};

#[cfg(feature = "image")]
pub use crate::client::ImageGenerationClient;

#[cfg(feature = "audio")]
pub use crate::client::AudioGenerationClient;

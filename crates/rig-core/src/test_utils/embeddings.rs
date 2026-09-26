//! Embedding helpers for deterministic tests.

use crate::driver::{Local, Model, Observation, Opened, Transport};
use crate::error::ProviderError;
use crate::operation::{Embedding as EmbeddingOp, EmbeddingCapabilities};
use crate::wire::Mode;
use crate::{
    Embed,
    embeddings::{
        Embedding, EmbeddingResponse,
        embed::{EmbedError, TextEmbedder},
    },
    wasm_compat::WasmCompatSend,
};

/// The mock embedding transport: every text embeds to one fixed vector,
/// five texts to a request at ten dimensions.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MockEmbeddings;

/// A deterministic embedding model that returns a fixed vector for each input document.
pub type MockEmbeddingModel = Model<Local<EmbeddingOp>, MockEmbeddings>;

impl MockEmbeddings {
    /// The mock's wire: the mock provider at the mock's batch size and width.
    pub fn wire() -> Local<EmbeddingOp> {
        Local::new(super::MOCK_PROVIDER).with_capabilities(EmbeddingCapabilities::new(5, 10))
    }

    /// The mock model: the mock's wire over this transport.
    pub fn model() -> MockEmbeddingModel {
        Model::new(Self::wire(), Self)
    }

    /// The reply the mock gives `texts`: one fixed vector each.
    pub fn embed(texts: Vec<String>) -> EmbeddingResponse {
        EmbeddingResponse::new(
            texts
                .into_iter()
                .map(|document| Embedding {
                    document,
                    vec: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                })
                .collect(),
            super::MOCK_PROVIDER,
        )
    }
}

impl Transport<Local<EmbeddingOp>> for MockEmbeddings {
    fn send(
        &self,
        texts: Vec<String>,
        _mode: Mode,
        _observation: Option<Observation>,
    ) -> Result<
        impl Future<Output = Opened<Vec<String>, EmbeddingResponse>> + WasmCompatSend + 'static + use<>,
        ProviderError,
    > {
        Ok(async move { Opened::new(futures::stream::iter([Ok(Self::embed(texts))])) })
    }
}

/// A test document that contributes one text fragment to an embedding request.
#[derive(Clone, Debug)]
pub struct MockTextDocument {
    /// Stable document identifier used by tests.
    pub id: String,
    /// Text to embed.
    pub text: String,
}

impl MockTextDocument {
    /// Create a single-text embedding fixture.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
        }
    }
}

impl Embed for MockTextDocument {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.text.clone());
        Ok(())
    }
}

/// A test document that contributes multiple text fragments to an embedding request.
#[derive(Clone, Debug)]
pub struct MockMultiTextDocument {
    /// Stable document identifier used by tests.
    pub id: String,
    /// Text fragments to embed.
    pub texts: Vec<String>,
}

impl MockMultiTextDocument {
    /// Create a multi-text embedding fixture.
    pub fn new(id: impl Into<String>, texts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            id: id.into(),
            texts: texts.into_iter().map(Into::into).collect(),
        }
    }
}

impl Embed for MockMultiTextDocument {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        for text in &self.texts {
            embedder.embed(text.clone());
        }
        Ok(())
    }
}

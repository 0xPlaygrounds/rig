//! Embedding helpers for deterministic tests.

use crate::driver::{Local, Model, Observation, Opened, Transport};
use crate::error::ProviderError;
use crate::operation::EmbeddingCapabilities;
use crate::wire::Mode;
use crate::{
    Embed,
    embeddings::{
        Embedding, EmbeddingResponse,
        embed::{EmbedError, TextEmbedder},
    },
    wasm_compat::WasmCompatSend,
};

/// The mock embedding runtime: every text embeds to one fixed vector, five
/// texts to a request at ten dimensions. It is the transport of a [`Local`]
/// embedding wire ([`Self::model`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MockEmbeddings;

/// A deterministic embedding model that returns a fixed vector for each input document.
pub type MockEmbeddingModel = Model<Local<crate::operation::Embedding>, MockEmbeddings>;

impl MockEmbeddings {
    /// The mock embedding model: its local wire over this runtime.
    pub fn model() -> MockEmbeddingModel {
        Model::new(
            Local::new(super::MOCK_PROVIDER).with_capabilities(EmbeddingCapabilities::new(5, 10)),
            Self,
        )
    }
}

impl Transport<Local<crate::operation::Embedding>> for MockEmbeddings {
    fn send(
        &self,
        texts: Vec<String>,
        _mode: Mode,
        _observation: Option<Observation>,
    ) -> Result<
        impl Future<Output = Opened<Vec<String>, Result<EmbeddingResponse, ProviderError>>>
        + WasmCompatSend
        + 'static
        + use<>,
        ProviderError,
    > {
        let response = Self::embed(texts);
        Ok(async move { Opened::new(futures::stream::iter([Ok(Ok(response))])) })
    }
}

impl MockEmbeddings {
    /// The reply this runtime gives for `texts`: one fixed ten-dimension
    /// vector per text, in order.
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

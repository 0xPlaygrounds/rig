//! Embedding helpers for deterministic tests.

use crate::driver::{Exchange, Local, Model, Opened, Opening, Step, Transport};
use crate::error::ProviderError;
use crate::wire::Capabilities;
use crate::{
    Embed,
    embeddings::{
        Embedding, EmbeddingResponse,
        embed::{EmbedError, TextEmbedder},
    },
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
            Local::new(super::MOCK_PROVIDER).with_capabilities(Capabilities::embedding(5, 10)),
            Self,
        )
    }
}

impl Transport<Local<crate::operation::Embedding>> for MockEmbeddings {
    fn send(
        &self,
        texts: Vec<String>,
        _exchange: Exchange,
    ) -> Opening<Step<crate::operation::Embedding>> {
        let response = Self::embed(texts);
        Opening::ready(Opened::new(futures::stream::iter([
            Ok::<_, ProviderError>(Step::End(response)),
        ])))
    }
}

impl MockEmbeddings {
    /// The reply this runtime gives for `texts`: one fixed ten-dimension
    /// vector per text, in order.
    pub fn embed(texts: Vec<String>) -> EmbeddingResponse {
        EmbeddingResponse {
            provider: super::MOCK_PROVIDER.to_owned(),
            ..EmbeddingResponse::new(
                texts
                    .into_iter()
                    .map(|document| Embedding {
                        document,
                        vec: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    })
                    .collect(),
            )
        }
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

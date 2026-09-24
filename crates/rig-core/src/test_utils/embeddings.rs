//! Embedding helpers for deterministic tests.

use crate::driver::{Model, Observation, Opened, Transport};
use crate::error::{EncodeError, ProviderError};
use crate::operation::EmbeddingCapabilities;
use crate::wire::{Decoder, Mode, Output, Sink, Wire, WireEvent};
use crate::{
    Embed,
    embeddings::{
        Embedding, EmbeddingResponse,
        embed::{EmbedError, TextEmbedder},
    },
    wasm_compat::WasmCompatSend,
};

/// The mock embedding endpoint, and its transport: every text embeds to
/// one fixed vector, five texts to a request at ten dimensions.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MockEmbeddings;

/// A deterministic embedding model that returns a fixed vector for each input document.
pub type MockEmbeddingModel = Model<MockEmbeddings, MockEmbeddings>;

impl MockEmbeddingModel {
    /// The mock embedding model.
    pub const fn mock() -> Self {
        Model {
            wire: MockEmbeddings,
            transport: MockEmbeddings,
        }
    }
}

impl Wire for MockEmbeddings {
    type Op = crate::operation::Embedding;
    type Payload = Vec<String>;
    type Frame = Vec<String>;
    type Decoder = MockEmbeddings;

    fn name(&self) -> &str {
        super::MOCK_PROVIDER
    }

    fn encode(&self, texts: Vec<String>, _mode: Mode) -> Result<Vec<String>, EncodeError> {
        Ok(texts)
    }

    fn decoder(&self, _mode: Mode) -> Self {
        Self
    }

    fn capabilities(&self) -> EmbeddingCapabilities {
        EmbeddingCapabilities::new(5, 10)
    }
}

impl Transport<MockEmbeddings> for MockEmbeddings {
    fn send(
        &self,
        texts: Vec<String>,
        _mode: Mode,
        _observation: Option<Observation>,
    ) -> Result<
        impl Future<Output = Opened<Vec<String>, Vec<String>>> + WasmCompatSend + 'static + use<>,
        ProviderError,
    > {
        Ok(async move { Opened::new(futures::stream::iter([Ok(texts)])) })
    }
}

impl Decoder<crate::operation::Embedding, Vec<String>> for MockEmbeddings {
    type Event = Vec<String>;

    fn classify(&self, texts: Vec<String>) -> WireEvent<Vec<String>> {
        WireEvent::Known(texts)
    }

    fn interpret(&mut self, texts: Vec<String>, out: &mut Output<crate::operation::Embedding>) {
        out.push(Ok(EmbeddingResponse::new(
            texts
                .into_iter()
                .map(|document| Embedding {
                    document,
                    vec: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                })
                .collect(),
            super::MOCK_PROVIDER,
        )));
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

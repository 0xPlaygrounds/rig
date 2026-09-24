//! Embedding helpers for deterministic tests.

use crate::driver::{Model, Opened, Transport};
use crate::error::{EncodeError, ProviderError};
use crate::operation::{self, EmbeddingCapabilities};
use crate::wire::{Decoder, Mode, Output, Sink, Wire, WireEvent, WireFrame};
use crate::{
    Embed,
    embeddings::{
        Embedding, EmbeddingResponse,
        embed::{EmbedError, TextEmbedder},
    },
    wasm_compat::WasmCompatSend,
};

/// The mock embedding endpoint: every text embeds as the same fixed
/// 10-dimension vector, at most five texts per request.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MockEmbeddingWire;

/// The scripted transport behind [`MockEmbeddingModel`]: it answers every
/// batch with one fixed vector per text.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MockEmbeddings;

/// A deterministic embedding model for tests: the mock embedding wire over
/// its scripted transport.
pub type MockEmbeddingModel = Model<MockEmbeddingWire, MockEmbeddings>;

/// The vector every mock text embeds as.
const MOCK_VECTOR: [f64; 10] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

impl Wire for MockEmbeddingWire {
    type Op = operation::Embedding;
    type Payload = Vec<String>;
    type Frame = WireFrame;
    type Decoder = MockEmbeddingDecoder;

    fn name(&self) -> &str {
        "mock"
    }

    fn encode(&self, texts: Vec<String>, _mode: Mode) -> Result<Vec<String>, EncodeError> {
        Ok(texts)
    }

    fn decoder(&self, _mode: Mode) -> MockEmbeddingDecoder {
        MockEmbeddingDecoder
    }

    fn capabilities(&self) -> EmbeddingCapabilities {
        EmbeddingCapabilities::new(5, MOCK_VECTOR.len())
    }
}

impl Transport for MockEmbeddings {
    type Payload = Vec<String>;
    type Frame = WireFrame;

    fn send(
        &self,
        texts: Vec<String>,
        _mode: Mode,
        _extensions: http::Extensions,
    ) -> Result<
        impl std::future::Future<Output = Opened<Vec<String>, WireFrame>>
        + WasmCompatSend
        + 'static
        + use<>,
        ProviderError,
    > {
        let vectors = vec![MOCK_VECTOR; texts.len()];
        let frame = WireFrame::Text(serde_json::to_string(&vectors)?);
        Ok(async move { Opened::new(futures::stream::iter([Ok(frame)])) })
    }
}

/// Decodes the mock embedding reply: one vector per text, in order.
pub struct MockEmbeddingDecoder;

impl Decoder<operation::Embedding> for MockEmbeddingDecoder {
    type Event = Vec<Vec<f64>>;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        match serde_json::from_str(&frame.as_str()) {
            Ok(vectors) => WireEvent::Known(vectors),
            Err(error) => WireEvent::Corrupt(error),
        }
    }

    fn interpret(&mut self, vectors: Self::Event, out: &mut Output<operation::Embedding>) {
        let embeddings = vectors
            .into_iter()
            .map(|vec| Embedding {
                document: String::new(),
                vec,
            })
            .collect();
        out.push(Ok(EmbeddingResponse::new(embeddings, "mock")));
    }
}

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

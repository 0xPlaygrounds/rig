use rig::embeddings::Embedding as CoreEmbedding;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Embedding(CoreEmbedding);

#[wasm_bindgen]
impl Embedding {
    /// Create a new Embedding instance.
    /// Generally not recommended as these are typically generated automatically from sending embedding requests to model providers.
    #[wasm_bindgen(constructor)]
    pub fn new(document: String, embedding: Vec<f64>) -> Self {
        let embedding = CoreEmbedding {
            document,
            vec: embedding,
        };
        Self(embedding)
    }

    pub fn document(&self) -> String {
        self.0.document.clone()
    }

    pub fn embedding(&self) -> Vec<f64> {
        self.0.vec.clone()
    }
}

impl From<Embedding> for rig::embeddings::Embedding {
    fn from(value: Embedding) -> Self {
        value.0
    }
}

impl From<rig::embeddings::Embedding> for Embedding {
    fn from(value: rig::embeddings::Embedding) -> Self {
        Self(value)
    }
}

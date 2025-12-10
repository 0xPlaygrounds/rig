use rig::embeddings::Embedding as CoreEmbedding;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Embedding {
    document: String,
    vec: Vec<f64>,
}

#[wasm_bindgen]
impl Embedding {
    /// Create a new Embedding instance.
    /// Generally not recommended as these are typically generated automatically from sending embedding requests to model providers.
    #[wasm_bindgen(constructor)]
    pub fn new(document: String, embedding: Vec<f64>) -> Self {
        Self {
            document,
            vec: embedding,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn document(&self) -> String {
        self.document.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_document(&mut self, val: String) {
        self.document = val;
    }

    #[wasm_bindgen(getter)]
    pub fn vec(&self) -> Vec<f64> {
        self.vec.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_vec(&mut self, vec: Vec<f64>) {
        self.vec = vec;
    }
}

impl From<Embedding> for CoreEmbedding {
    fn from(value: Embedding) -> Self {
        let Embedding { document, vec } = value;
        CoreEmbedding { document, vec }
    }
}

impl From<CoreEmbedding> for Embedding {
    fn from(value: CoreEmbedding) -> Self {
        let CoreEmbedding { document, vec } = value;
        Self { document, vec }
    }
}

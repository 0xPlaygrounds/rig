// ================================================================
//! Together AI Embeddings Integration
//! From [Together AI Reference](https://docs.together.ai/reference/embeddings)
// ================================================================

use super::client::TogetherExt;
use crate::providers::openai::embedding::{GenericEmbeddingModel, OpenAIEmbeddingsCompatible};

// ================================================================
// Together AI Embedding API
// ================================================================
pub const BGE_BASE_EN_V1_5: &str = "BAAI/bge-base-en-v1.5";
pub const BGE_LARGE_EN_V1_5: &str = "BAAI/bge-large-en-v1.5";
pub const BERT_BASE_UNCASED: &str = "bert-base-uncased";
pub const M2_BERT_2K_RETRIEVAL_ENCODER_V1: &str = "hazyresearch/M2-BERT-2k-Retrieval-Encoder-V1";
pub const M2_BERT_80M_32K_RETRIEVAL: &str = "togethercomputer/m2-bert-80M-32k-retrieval";
pub const M2_BERT_80M_2K_RETRIEVAL: &str = "togethercomputer/m2-bert-80M-2k-retrieval";
pub const M2_BERT_80M_8K_RETRIEVAL: &str = "togethercomputer/m2-bert-80M-8k-retrieval";
pub const SENTENCE_BERT: &str = "sentence-transformers/msmarco-bert-base-dot-v5";
pub const UAE_LARGE_V1: &str = "WhereIsAI/UAE-Large-V1";

impl OpenAIEmbeddingsCompatible for TogetherExt {
    const PROVIDER_NAME: &'static str = "together";
    const REQUIRES_USAGE: bool = false;
    const SUPPORTS_ENCODING_FORMAT: bool = false;
    const SUPPORTS_USER: bool = false;

    fn embeddings_path(&self) -> String {
        "/v1/embeddings".to_string()
    }
}

/// Together AI embedding model, driven by the shared OpenAI-compatible transport.
pub type EmbeddingModel<H = crate::http_client::BoxedHttpClient> =
    GenericEmbeddingModel<TogetherExt, H>;

#[cfg(test)]
mod tests;

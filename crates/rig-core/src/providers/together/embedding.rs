//! Together AI's embedding model identifiers.
//!
//! ```no_run
//! use rig_core::providers::{together, openai::wire::{OpenAI, TOGETHER}};
//! let wire = OpenAI::from_env_with(&TOGETHER)?
//!     .embeddings(together::BGE_BASE_EN_V1_5, None);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub const BGE_BASE_EN_V1_5: &str = "BAAI/bge-base-en-v1.5";
pub const BGE_LARGE_EN_V1_5: &str = "BAAI/bge-large-en-v1.5";
pub const BERT_BASE_UNCASED: &str = "bert-base-uncased";
pub const M2_BERT_2K_RETRIEVAL_ENCODER_V1: &str = "hazyresearch/M2-BERT-2k-Retrieval-Encoder-V1";
pub const M2_BERT_80M_32K_RETRIEVAL: &str = "togethercomputer/m2-bert-80M-32k-retrieval";
pub const M2_BERT_80M_2K_RETRIEVAL: &str = "togethercomputer/m2-bert-80M-2k-retrieval";
pub const M2_BERT_80M_8K_RETRIEVAL: &str = "togethercomputer/m2-bert-80M-8k-retrieval";
pub const SENTENCE_BERT: &str = "sentence-transformers/msmarco-bert-base-dot-v5";
pub const UAE_LARGE_V1: &str = "WhereIsAI/UAE-Large-V1";

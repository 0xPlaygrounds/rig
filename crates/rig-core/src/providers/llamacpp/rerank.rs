//! llama.cpp reranking.
//!
//! `llama-server` serves one rerank handler behind four aliases (`/rerank`,
//! `/reranking`, `/v1/rerank`, `/v1/reranking`) and answers the Jina-shaped
//! body rig's shared `JinaCompatibleRerank` driver
//! (`providers::internal::rerank`) speaks.
//!
//! Two server-side preconditions decide whether this works at all, and both
//! fail loudly rather than silently:
//!
//! * The server must be started with `--reranking` (and therefore
//!   `--embeddings --pooling rank`), or every request answers
//!   `501 {"error":{"message":"This server does not support reranking. Start
//!   it with `--reranking`", "type":"not_supported_error"}}`.
//! * The loaded weights must be a reranker (a cross-encoder such as
//!   `bge-reranker-v2-m3` or `Qwen3-Reranker`). A causal LM has no rank
//!   pooling head and the server refuses to start with `--pooling rank`.
//!
//! Unlike Voyage AI's reranker, llama.cpp never echoes the document text back
//! — there is no `return_documents` on this path — so every
//! [`RerankResult::document`](crate::rerank::RerankResult::document) is
//! `None` and the caller maps results back through
//! [`index`](crate::rerank::RerankResult::index).

use crate::providers::internal::rerank::{GenericRerankModel, JinaCompatibleRerank};

impl JinaCompatibleRerank for super::client::Llamacpp {
    const PROVIDER_NAME: &'static str = "llamacpp";

    // `llama-server` posts one task per document and waits for all of them,
    // bounded only by memory and the client's patience — there is no
    // documented cap. This is the batching hint
    // [`RerankModel::max_documents`](crate::rerank::RerankModel::max_documents)
    // exposes, set to the same 1024 the shared embeddings driver defaults to
    // rather than to a number the server does not actually enforce.
    const MAX_DOCUMENTS: usize = 1024;

    // The `/v1` prefix comes from `Llamacpp::build_uri`, so the bare path
    // here resolves to `/v1/rerank`.
    fn rerank_path(&self) -> String {
        "/rerank".to_string()
    }
}

/// llama.cpp rerank model, driven by the shared Jina-compatible rerank path.
pub type RerankModel<H = crate::http_client::BoxedHttpClient> =
    GenericRerankModel<super::client::Llamacpp, H>;

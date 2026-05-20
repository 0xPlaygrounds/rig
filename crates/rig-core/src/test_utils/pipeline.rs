//! Pipeline helpers for deterministic tests.

use crate::{
    completion::{CompletionError, Prompt, PromptError},
    message::{self, Message},
    vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndex, request::Filter},
    wasm_compat::WasmCompatSend,
};

/// A prompt model that echoes user text with a stable prefix.
pub struct MockPromptModel;

impl Prompt for MockPromptModel {
    #[allow(refining_impl_trait)]
    async fn prompt(&self, prompt: impl Into<Message>) -> Result<String, PromptError> {
        let msg = prompt.into();
        let prompt = match msg {
            Message::User { content } => match content.first() {
                message::UserContent::Text(message::Text { text, .. }) => text,
                _ => {
                    return Err(PromptError::CompletionError(CompletionError::RequestError(
                        "mock prompt model only accepts text user messages".into(),
                    )));
                }
            },
            _ => {
                return Err(PromptError::CompletionError(CompletionError::RequestError(
                    "mock prompt model only accepts user messages".into(),
                )));
            }
        };

        Ok(format!("Mock response: {prompt}"))
    }
}

/// A vector index that always returns one JSON document containing `{"foo":"bar"}`.
pub struct MockVectorStoreIndex;

impl VectorStoreIndex for MockVectorStoreIndex {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: for<'a> serde::Deserialize<'a> + WasmCompatSend>(
        &self,
        _req: VectorSearchRequest,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let doc = serde_json::from_value(serde_json::json!({
            "foo": "bar",
        }))?;

        Ok(vec![(1.0, "doc1".to_string(), doc)])
    }

    async fn top_n_ids(
        &self,
        _req: VectorSearchRequest,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Ok(vec![(1.0, "doc1".to_string())])
    }
}

/// Document fixture returned by [`MockVectorStoreIndex`] in pipeline tests.
#[derive(Debug, serde::Deserialize, PartialEq)]
pub struct Foo {
    pub foo: String,
}

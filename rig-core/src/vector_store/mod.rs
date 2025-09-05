use futures::future::BoxFuture;
pub use request::VectorSearchRequest;
use reqwest::StatusCode;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

use crate::embeddings::EmbeddingError;
use crate::{embeddings::Embedding, Embed, OneOrMany};

pub mod in_memory_store;
pub mod request;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VectorStoreError {
	#[error("Embedding error: {0}")]
	EmbeddingError(#[from] EmbeddingError),

	/// Json error (e.g.: serialization, deserialization, etc.)
	#[error("Json error: {0}")]
	JsonError(#[from] serde_json::Error),

	#[error("Datastore error: {0}")]
	DatastoreError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

	#[error("Missing Id: {0}")]
	MissingIdError(String),

	#[error("HTTP request error: {0}")]
	ReqwestError(#[from] reqwest::Error),

	#[error("External call to API returned an error. Error code: {0} Message: {1}")]
	ExternalAPIError(StatusCode, String),

	#[error("Error while building VectorSearchRequest: {0}")]
	BuilderError(String),
}

/// Trait for inserting documents into a vector store.
pub trait InsertDocuments: Send + Sync {
	/// Insert documents into the vector store.
	///
	fn insert_documents<Doc: Serialize + Embed + Send>(
		&self,
		documents: Vec<(Doc, OneOrMany<Embedding>)>,
	) -> impl std::future::Future<Output=Result<(), VectorStoreError>> + Send;
}

/// Trait for vector store indexes
pub trait VectorStoreIndex: Send + Sync {
	/// Get the top n documents based on the distance to the given query.
	/// The result is a list of tuples of the form (score, id, document)
	fn top_n<T: for<'a> Deserialize<'a> + Send>(
		&self,
		req: VectorSearchRequest,
	) -> impl std::future::Future<Output=Result<Vec<(f64, String, T)>, VectorStoreError>> + Send;

	/// Same as `top_n` but returns the document ids only.
	fn top_n_ids(
		&self,
		req: VectorSearchRequest,
	) -> impl std::future::Future<Output=Result<Vec<(f64, String)>, VectorStoreError>> + Send;
}

pub type TopNResults = Result<Vec<(f64, String, Value)>, VectorStoreError>;

pub trait VectorStoreIndexDyn: Send + Sync {
	fn top_n<'a>(&'a self, req: VectorSearchRequest) -> BoxFuture<'a, TopNResults>;

	fn top_n_ids<'a>(
		&'a self,
		req: VectorSearchRequest,
	) -> BoxFuture<'a, Result<Vec<(f64, String)>, VectorStoreError>>;
}

impl<I: VectorStoreIndex> VectorStoreIndexDyn for I {
	fn top_n<'a>(
		&'a self,
		req: VectorSearchRequest,
	) -> BoxFuture<'a, Result<Vec<(f64, String, Value)>, VectorStoreError>> {
		Box::pin(async move {
			Ok(self
				.top_n::<serde_json::Value>(req)
				.await?
				.into_iter()
				.map(|(score, id, doc)| (score, id, prune_document(doc).unwrap_or_default()))
				.collect::<Vec<_>>())
		})
	}

	fn top_n_ids<'a>(
		&'a self,
		req: VectorSearchRequest,
	) -> BoxFuture<'a, Result<Vec<(f64, String)>, VectorStoreError>> {
		Box::pin(self.top_n_ids(req))
	}
}

fn prune_document(document: serde_json::Value) -> Option<serde_json::Value> {
	match document {
		Value::Object(mut map) => {
			let new_map = map
				.iter_mut()
				.filter_map(|(key, value)| {
					prune_document(value.take()).map(|value| (key.clone(), value))
				})
				.collect::<serde_json::Map<_, _>>();

			Some(Value::Object(new_map))
		}
		Value::Array(vec) if vec.len() > 400 => None,
		Value::Array(vec) => Some(Value::Array(
			vec.into_iter().filter_map(prune_document).collect(),
		)),
		Value::Number(num) => Some(Value::Number(num)),
		Value::String(s) => Some(Value::String(s)),
		Value::Bool(b) => Some(Value::Bool(b)),
		Value::Null => Some(Value::Null),
	}
}

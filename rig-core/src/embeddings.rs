//! This module provides functionality for working with embeddings and embedding models.
//! Embeddings are numerical representations of documents or other objects, typically used in
//! natural language processing (NLP) tasks such as text classification, information retrieval,
//! and document similarity.
//!
//! The module defines the [EmbeddingModel] trait, which represents an embedding model that can
//! generate embeddings for documents. It also provides an implementation of the [EmbeddingsBuilder]
//! struct, which allows users to build collections of document embeddings using different embedding
//! models and document sources.
//!
//! The module also defines the [Embedding] struct, which represents a single document embedding,
//! and the [DocumentEmbeddings] struct, which represents a document along with its associated
//! embeddings. These structs are used to store and manipulate collections of document embeddings.
//!
//! Finally, the module defines the [EmbeddingError] enum, which represents various errors that
//! can occur during embedding generation or processing.
//!
//! # Example
//! ```rust
//! use rig::providers::openai::{Client, self};
//! use rig::embeddings::{EmbeddingModel, EmbeddingsBuilder};
//!
//! // Initialize the OpenAI client
//! let openai = Client::new("your-openai-api-key");
//!
//! // Create an instance of the `text-embedding-ada-002` model
//! let embedding_model = openai.embedding_model(openai::TEXT_EMBEDDING_ADA_002);
//!
//! // Create an embeddings builder and add documents
//! let embeddings = EmbeddingsBuilder::new(embedding_model)
//!     .simple_document("doc1", "This is the first document.")
//!     .simple_document("doc2", "This is the second document.")
//!     .build()
//!     .await
//!     .expect("Failed to build embeddings.");
//!
//! // Use the generated embeddings
//! // ...
//! ```

use std::{cmp::max, collections::HashMap};

use futures::{stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};

use crate::tool::{ToolEmbedding, ToolSet, ToolType};

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] reqwest::Error),

    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Error processing the document for embedding
    #[error("DocumentError: {0}")]
    DocumentError(String),

    /// Error parsing the completion response
    #[error("ResponseError: {0}")]
    ResponseError(String),

    /// Error returned by the embedding model provider
    #[error("ProviderError: {0}")]
    ProviderError(String),
}

/// Trait for embedding models that can generate embeddings for documents.
pub trait EmbeddingModel: Clone + Sync + Send {
    /// The maximum number of documents that can be embedded in a single request.
    const MAX_DOCUMENTS: usize;

    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize;

    /// Embed a single document
    fn embed_document(
        &self,
        document: &str,
    ) -> impl std::future::Future<Output = Result<Embedding, EmbeddingError>> + Send
    where
        Self: Sync,
    {
        async {
            Ok(self
                .embed_documents(vec![document.to_string()])
                .await?
                .first()
                .cloned()
                .expect("One embedding should be present"))
        }
    }

    /// Embed multiple documents in a single request
    fn embed_documents(
        &self,
        documents: Vec<String>,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + Send;
}

/// Struct that holds a single document and its embedding.
#[derive(Clone, Default, Deserialize, Serialize, Debug)]
pub struct Embedding {
    /// The document that was embedded
    pub document: String,
    /// The embedding vector
    pub vec: Vec<f64>,
}

impl PartialEq for Embedding {
    fn eq(&self, other: &Self) -> bool {
        self.document == other.document
    }
}

impl Eq for Embedding {}

impl Embedding {
    pub fn distance(&self, other: &Self) -> f64 {
        let dot_product: f64 = self
            .vec
            .iter()
            .zip(other.vec.iter())
            .map(|(x, y)| x * y)
            .sum();

        let product_of_lengths = (self.vec.len() * other.vec.len()) as f64;

        dot_product / product_of_lengths
    }
}

/// Struct that holds a document and its embeddings.
///
/// The struct is designed to model any kind of documents that can be serialized to JSON
/// (including a simple string).
///
/// Moreover, it can hold multiple embeddings for the same document, thus allowing a
/// large document to be retrieved from a query that matches multiple smaller and
/// distinct text documents. For example, if the document is a textbook, a summary of
/// each chapter could serve as the book's embeddings.
#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct DocumentEmbeddings {
    #[serde(rename = "_id")]
    pub id: String,
    pub document: serde_json::Value,
    pub embeddings: Vec<Embedding>,
}

type Embeddings = Vec<DocumentEmbeddings>;

/// Builder for creating a collection of embeddings
pub struct EmbeddingsBuilder<M: EmbeddingModel> {
    model: M,
    documents: Vec<(String, serde_json::Value, Vec<String>)>,
}

impl<M: EmbeddingModel> EmbeddingsBuilder<M> {
    /// Create a new embedding builder with the given embedding model
    pub fn new(model: M) -> Self {
        Self {
            model,
            documents: vec![],
        }
    }

    /// Add a simple document to the embedding collection.
    /// The provided document string will be used for the embedding.
    pub fn simple_document(mut self, id: &str, document: &str) -> Self {
        self.documents.push((
            id.to_string(),
            serde_json::Value::String(document.to_string()),
            vec![document.to_string()],
        ));
        self
    }

    /// Add multiple documents to the embedding collection.
    /// Each element of the vector is a tuple of the form (id, document).
    pub fn simple_documents(mut self, documents: Vec<(String, String)>) -> Self {
        self.documents
            .extend(documents.into_iter().map(|(id, document)| {
                (
                    id,
                    serde_json::Value::String(document.clone()),
                    vec![document],
                )
            }));
        self
    }

    /// Add a tool to the embedding collection.
    /// The `tool.context()` corresponds to the document being stored while
    /// `tool.embedding_docs()` corresponds to the documents that will be used to generate the embeddings.
    pub fn tool(mut self, tool: impl ToolEmbedding + 'static) -> Result<Self, EmbeddingError> {
        self.documents.push((
            tool.name(),
            serde_json::to_value(tool.context())?,
            tool.embedding_docs(),
        ));
        Ok(self)
    }

    /// Add the tools from the given toolset to the embedding collection.
    pub fn tools(mut self, toolset: &ToolSet) -> Result<Self, EmbeddingError> {
        for (name, tool) in toolset.tools.iter() {
            if let ToolType::Embedding(tool) = tool {
                self.documents.push((
                    name.clone(),
                    tool.context().map_err(|e| {
                        EmbeddingError::DocumentError(format!(
                            "Failed to generate context for tool {}: {}",
                            name, e
                        ))
                    })?,
                    tool.embedding_docs(),
                ));
            }
        }
        Ok(self)
    }

    /// Add a document to the embedding collection.
    /// `embed_documents` are the documents that will be used to generate the embeddings
    /// for `document`.
    pub fn document<T: Serialize>(
        mut self,
        id: &str,
        document: T,
        embed_documents: Vec<String>,
    ) -> Self {
        self.documents.push((
            id.to_string(),
            serde_json::to_value(document).expect("Document should serialize"),
            embed_documents,
        ));
        self
    }

    /// Add multiple documents to the embedding collection.
    /// Each element of the vector is a tuple of the form (id, document, embed_documents).
    pub fn documents<T: Serialize>(mut self, documents: Vec<(String, T, Vec<String>)>) -> Self {
        self.documents.extend(
            documents
                .into_iter()
                .map(|(id, document, embed_documents)| {
                    (
                        id,
                        serde_json::to_value(document).expect("Document should serialize"),
                        embed_documents,
                    )
                }),
        );
        self
    }

    /// Add a json document to the embedding collection.
    pub fn json_document(
        mut self,
        id: &str,
        document: serde_json::Value,
        embed_documents: Vec<String>,
    ) -> Self {
        self.documents
            .push((id.to_string(), document, embed_documents));
        self
    }

    /// Add multiple json documents to the embedding collection.
    pub fn json_documents(
        mut self,
        documents: Vec<(String, serde_json::Value, Vec<String>)>,
    ) -> Self {
        self.documents.extend(documents);
        self
    }

    /// Generate the embeddings for the given documents
    pub async fn build(self) -> Result<Embeddings, EmbeddingError> {
        // Create a temporary store for the documents
        let documents_map = self
            .documents
            .into_iter()
            .map(|(id, document, docs)| (id, (document, docs)))
            .collect::<HashMap<_, _>>();

        let embeddings = stream::iter(documents_map.iter())
            // Flatten the documents
            .flat_map(|(id, (_, docs))| {
                stream::iter(docs.iter().map(|doc| (id.clone(), doc.clone())))
            })
            // Chunk them into N (the emebdding API limit per request)
            .chunks(M::MAX_DOCUMENTS)
            // Generate the embeddings
            .map(|docs| async {
                let (ids, docs): (Vec<_>, Vec<_>) = docs.into_iter().unzip();
                Ok::<_, EmbeddingError>(
                    ids.into_iter()
                        .zip(self.model.embed_documents(docs).await?.into_iter())
                        .collect::<Vec<_>>(),
                )
            })
            .boxed()
            // Parallelize the embeddings generation over 10 concurrent requests
            .buffer_unordered(max(1, 1024 / M::MAX_DOCUMENTS))
            .try_fold(vec![], |mut acc, mut embeddings| async move {
                Ok({
                    acc.append(&mut embeddings);
                    acc
                })
            })
            .await?;

        // Assemble the DocumentEmbeddings
        let mut document_embeddings: HashMap<String, DocumentEmbeddings> = HashMap::new();
        embeddings.into_iter().for_each(|(id, embedding)| {
            let (document, _) = documents_map.get(&id).expect("Document not found");
            let document_embedding =
                document_embeddings
                    .entry(id.clone())
                    .or_insert_with(|| DocumentEmbeddings {
                        id: id.clone(),
                        document: document.clone(),
                        embeddings: vec![],
                    });

            document_embedding.embeddings.push(embedding);
        });

        Ok(document_embeddings.into_values().collect())
    }
}

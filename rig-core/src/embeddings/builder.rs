//! The module defines the [EmbeddingsBuilder] struct which accumulates objects to be embedded and generates the embeddings for each object when built.
//! Only types that implement the [Embeddable] trait can be added to the [EmbeddingsBuilder].
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

use super::embedding::{ Embedding, EmbeddingError, EmbeddingModel};

/// Struct that holds a document and its embeddings.
///
/// The struct is designed to model any kind of documents that can be serialized to JSON
/// (including a simple string).
///
/// Moreover, it can hold multiple embeddings for the same document, thus allowing a
/// large document to be retrieved from a query that matches multiple smaller and
/// distinct text documents. For example, if the document is a textbook, a summary of
/// each chapter could serve as the book's embeddings.
#[derive(Clone, Eq, PartialEq, Serialize, Deserialize)]
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
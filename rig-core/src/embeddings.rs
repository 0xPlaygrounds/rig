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

use std::{cmp::max, collections::HashMap, marker::PhantomData};

use futures::{stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};

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
#[derive(Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct DocumentEmbeddings {
    #[serde(rename = "_id")]
    pub id: String,
    pub document: serde_json::Value,
    pub embeddings: Vec<Embedding>,
}
pub trait EmbeddingKind {}
pub struct SingleEmbedding;
impl EmbeddingKind for SingleEmbedding {}
pub struct ManyEmbedding;
impl EmbeddingKind for ManyEmbedding {}

pub trait Embeddable {
    type Kind: EmbeddingKind;
    fn embeddable(&self) -> Vec<String>;
}

/// Builder for creating a collection of embeddings
pub struct EmbeddingsBuilder<M: EmbeddingModel, D: Embeddable, K: EmbeddingKind> {
    kind: PhantomData<K>,
    model: M,
    documents: Vec<(D, Vec<String>)>,
}

impl<M: EmbeddingModel, D: Embeddable<Kind = K>, K: EmbeddingKind> EmbeddingsBuilder<M, D, K> {
    /// Create a new embedding builder with the given embedding model
    pub fn new(model: M) -> Self {
        Self {
            kind: PhantomData,
            model,
            documents: vec![],
        }
    }

    pub fn document(mut self, document: D) -> Self {
        let embed_targets = document.embeddable();

        self.documents.push((document, embed_targets));
        self
    }

    pub fn documents(mut self, documents: Vec<D>) -> EmbeddingsBuilder<M, D, D::Kind> {
        documents.into_iter().for_each(|doc| {
            let embed_targets = doc.embeddable();

            self.documents.push((doc, embed_targets));
        });

        self
    }
}

impl<M: EmbeddingModel, D: Embeddable + Send + Sync + Clone>
    EmbeddingsBuilder<M, D, ManyEmbedding>
{
    pub async fn build(&self) -> Result<Vec<(D, Vec<Embedding>)>, EmbeddingError> {
        let documents_map = self
            .documents
            .clone()
            .into_iter()
            .enumerate()
            .map(|(id, (document, _))| (id, document))
            .collect::<HashMap<_, _>>();

        let embeddings = stream::iter(self.documents.clone().into_iter().enumerate())
            // Flatten the documents
            .flat_map(|(i, (_, embed_targets))| {
                stream::iter(
                    embed_targets
                        .into_iter()
                        .map(move |target| (i, target.clone())),
                )
            })
            // Chunk them into N (the emebdding API limit per request)
            .chunks(M::MAX_DOCUMENTS)
            // Generate the embeddings
            .map(|docs| async {
                let (documents, embed_targets): (Vec<_>, Vec<_>) = docs.into_iter().unzip();
                Ok::<_, EmbeddingError>(
                    documents
                        .into_iter()
                        .zip(self.model.embed_documents(embed_targets).await?.into_iter())
                        .collect::<Vec<_>>(),
                )
            })
            .boxed()
            // Parallelize the embeddings generation over 10 concurrent requests
            .buffer_unordered(max(1, 1024 / M::MAX_DOCUMENTS))
            // .try_collect::<Vec<_>>()
            // .await;
            .try_fold(
                HashMap::new(),
                |mut acc: HashMap<_, Vec<_>>, embeddings| async move {
                    embeddings.into_iter().for_each(|(i, embedding)| {
                        acc.entry(i).or_default().push(embedding);
                    });

                    Ok(acc)
                },
            )
            .await?
            .iter()
            .fold(vec![], |mut acc, (i, embeddings_vec)| {
                acc.push((
                    documents_map.get(i).cloned().unwrap(),
                    embeddings_vec.clone(),
                ));
                acc
            });

        Ok(embeddings)
    }
}

impl<M: EmbeddingModel, D: Embeddable + Send + Sync + Clone>
    EmbeddingsBuilder<M, D, SingleEmbedding>
{
    pub async fn build(&self) -> Result<Vec<(D, Embedding)>, EmbeddingError> {
        let embeddings = stream::iter(
            self.documents
                .clone()
                .into_iter()
                .map(|(document, embed_target)| (document, embed_target.first().cloned().unwrap())),
        )
        // Chunk them into N (the emebdding API limit per request)
        .chunks(M::MAX_DOCUMENTS)
        // Generate the embeddings
        .map(|docs| async {
            let (documents, embed_targets): (Vec<_>, Vec<_>) = docs.into_iter().unzip();
            Ok::<_, EmbeddingError>(
                documents
                    .into_iter()
                    .zip(self.model.embed_documents(embed_targets).await?.into_iter())
                    .collect::<Vec<_>>(),
            )
        })
        .boxed()
        // Parallelize the embeddings generation over 10 concurrent requests
        .buffer_unordered(max(1, 1024 / M::MAX_DOCUMENTS))
        .try_fold(vec![], |mut acc, embeddings| async move {
            acc.extend(embeddings);
            Ok(acc)
        })
        .await?;

        Ok(embeddings)
    }
}

impl Embeddable for String {
    type Kind = SingleEmbedding;

    fn embeddable(&self) -> Vec<String> {
        vec![self.clone()]
    }
}

impl Embeddable for i32 {
    type Kind = SingleEmbedding;

    fn embeddable(&self) -> Vec<String> {
        vec![self.to_string()]
    }
}

impl<T: Embeddable> Embeddable for Vec<T> {
    type Kind = ManyEmbedding;

    fn embeddable(&self) -> Vec<String> {
        self.iter().flat_map(|i| i.embeddable()).collect()
    }
}

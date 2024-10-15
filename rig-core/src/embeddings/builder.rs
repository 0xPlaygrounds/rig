//! The module defines the [EmbeddingsBuilder] struct which accumulates objects to be embedded and generates the embeddings for each object when built.
//! Only types that implement the [Embeddable] trait can be added to the [EmbeddingsBuilder].
//!
//! # Example
//! ```rust
//! use std::env;
//!
//! use rig::{
//!    embeddings::EmbeddingsBuilder,
//!    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
//! };
//! use rig_derive::Embed;
//!
//! #[derive(Embed)]
//! struct FakeDefinition {
//!    id: String,
//!    word: String,
//!    #[embed]
//!    definitions: Vec<String>,
//! }
//! // Create OpenAI client
//! let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
//! let openai_client = Client::new(&openai_api_key);
//!
//! let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);
//!
//! let embeddings = EmbeddingsBuilder::new(model.clone())
//!    .documents(vec![
//!        FakeDefinition {
//!            id: "doc0".to_string(),
//!            word: "flurbo".to_string(),
//!            definitions: vec![
//!                "A green alien that lives on cold planets.".to_string(),
//!                "A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
//!            ]
//!        },
//!        FakeDefinition {
//!            id: "doc1".to_string(),
//!            word: "glarb-glarb".to_string(),
//!            definitions: vec![
//!                "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
//!                "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
//!            ]
//!        },
//!        FakeDefinition {
//!            id: "doc2".to_string(),
//!            word: "linglingdong".to_string(),
//!            definitions: vec![
//!                "A term used by inhabitants of the sombrero galaxy to describe humans.".to_string(),
//!                "A rare, mystical instrument crafted by the ancient monks of the Nebulon Mountain Ranges on the planet Quarm.".to_string()
//!            ]
//!        },
//!    ])
//!    .build()
//!    .await?;
//!                                 
//! // Use the generated embeddings
//! // ...
//! ```

use std::{cmp::max, collections::HashMap, marker::PhantomData};

use futures::{stream, StreamExt, TryStreamExt};

use crate::Embeddable;

use super::{
    embeddable::{EmbeddableError, EmbeddingKind, ManyEmbedding, SingleEmbedding},
    embedding::{Embedding, EmbeddingError, EmbeddingModel},
};

/// Builder for creating a collection of embeddings.
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

    /// Add a document that implements `Embeddable` to the builder.
    pub fn document(mut self, document: D) -> Result<Self, EmbeddableError> {
        let embed_targets = document.embeddable()?;

        self.documents.push((document, embed_targets));
        Ok(self)
    }

    /// Add many documents that implement `Embeddable` to the builder.
    pub fn documents(mut self, documents: Vec<D>) -> Result<Self, EmbeddableError> {
        for doc in documents.into_iter() {
            let embed_targets = doc.embeddable()?;

            self.documents.push((doc, embed_targets));
        }

        Ok(self)
    }
}

impl<M: EmbeddingModel, D: Embeddable + Send + Sync + Clone>
    EmbeddingsBuilder<M, D, ManyEmbedding>
{
    /// Generate embeddings for all documents in the builder.
    /// The method only applies when documents in the builder each contain multiple embedding targets.
    /// Returns a vector of tuples, where the first element is the document and the second element is the vector of embeddings.
    pub async fn build(&self) -> Result<Vec<(D, Vec<Embedding>)>, EmbeddingError> {
        // Use this for reference later to merge a document back with its embeddings.
        let documents_map = self
            .documents
            .clone()
            .into_iter()
            .enumerate()
            .map(|(id, (document, _))| (id, document))
            .collect::<HashMap<_, _>>();

        let embeddings = stream::iter(self.documents.iter().enumerate())
            // Merge the embedding targets of each document into a single list of embedding targets.
            .flat_map(|(i, (_, embed_targets))| {
                stream::iter(embed_targets.iter().map(move |target| (i, target.clone())))
            })
            // Chunk them into N (the emebdding API limit per request).
            .chunks(M::MAX_DOCUMENTS)
            // Generate the embeddings for a chunk at a time.
            .map(|docs| async {
                let (document_indices, embed_targets): (Vec<_>, Vec<_>) = docs.into_iter().unzip();

                Ok::<_, EmbeddingError>(
                    document_indices
                        .into_iter()
                        .zip(self.model.embed_documents(embed_targets).await?.into_iter())
                        .collect::<Vec<_>>(),
                )
            })
            .boxed()
            // Parallelize the embeddings generation over 10 concurrent requests
            .buffer_unordered(max(1, 1024 / M::MAX_DOCUMENTS))
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
    /// Generate embeddings for all documents in the builder.
    /// The method only applies when documents in the builder each contain a single embedding target.
    /// Returns a vector of tuples, where the first element is the document and the second element is the embedding.
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

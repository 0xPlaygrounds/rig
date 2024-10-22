//! The module defines the [EmbeddingsBuilder] struct which accumulates objects to be embedded and generates the embeddings for each object when built.
//! Only types that implement the [Embeddable] trait can be added to the [EmbeddingsBuilder].

use std::{cmp::max, collections::HashMap};

use futures::{stream, StreamExt, TryStreamExt};

use crate::{
    embeddings::{Embeddable, Embedding, EmbeddingError, EmbeddingModel},
    OneOrMany,
};

/// Builder for creating a collection of embeddings.
pub struct EmbeddingsBuilder<M: EmbeddingModel, T: Embeddable> {
    model: M,
    documents: Vec<(T, OneOrMany<String>)>,
}

impl<M: EmbeddingModel, T: Embeddable> EmbeddingsBuilder<M, T> {
    /// Create a new embedding builder with the given embedding model
    pub fn new(model: M) -> Self {
        Self {
            model,
            documents: vec![],
        }
    }

    /// Add a document that implements `Embeddable` to the builder.
    pub fn document(mut self, document: T) -> Result<Self, T::Error> {
        let embed_targets = document.embeddable()?;

        self.documents.push((document, embed_targets));
        Ok(self)
    }

    /// Add many documents that implement `Embeddable` to the builder.
    pub fn documents(mut self, documents: Vec<T>) -> Result<Self, T::Error> {
        for doc in documents.into_iter() {
            let embed_targets = doc.embeddable()?;

            self.documents.push((doc, embed_targets));
        }

        Ok(self)
    }
}

/// # Example
/// ```rust
/// use std::env;
///
/// use rig::{
///     embeddings::EmbeddingsBuilder,
///     providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
///     vector_store::{in_memory_store::InMemoryVectorStore, VectorStoreIndex},
///     Embeddable,
/// };
/// use serde::{Deserialize, Serialize};
///
/// // Shape of data that needs to be RAG'ed.
/// // The definition field will be used to generate embeddings.
/// #[derive(Embeddable, Clone, Deserialize, Debug, Serialize, Eq, PartialEq, Default)]
/// struct FakeDefinition {
///     id: String,
///     word: String,
///     #[embed]
///     definitions: Vec<String>,
/// }
///
/// // Create OpenAI client
/// let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
/// let openai_client = Client::new(&openai_api_key);
///
/// let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);
///
/// # tokio_test::block_on(async {
/// let embeddings = EmbeddingsBuilder::new(model.clone())
///     .documents(vec![
///         FakeDefinition {
///             id: "doc0".to_string(),
///             word: "flurbo".to_string(),
///             definitions: vec![
///                 "A green alien that lives on cold planets.".to_string(),
///                 "A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
///             ]
///         },
///         FakeDefinition {
///             id: "doc1".to_string(),
///             word: "glarb-glarb".to_string(),
///             definitions: vec![
///                 "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
///                 "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
///             ]
///         },
///         FakeDefinition {
///             id: "doc2".to_string(),
///             word: "linglingdong".to_string(),
///             definitions: vec![
///                 "A term used by inhabitants of the sombrero galaxy to describe humans.".to_string(),
///                 "A rare, mystical instrument crafted by the ancient monks of the Nebulon Mountain Ranges on the planet Quarm.".to_string()
///             ]
///         },
///     ])
///     .unwrap()
///     .build()
///     .await
///     .unwrap();
///
/// assert_eq!(embeddings.iter().any(|(doc, embeddings)| doc.id == "doc0" && embeddings.len() == 2), true);
/// assert_eq!(embeddings.iter().any(|(doc, embeddings)| doc.id == "doc1" && embeddings.len() == 2), true);
/// assert_eq!(embeddings.iter().any(|(doc, embeddings)| doc.id == "doc2" && embeddings.len() == 2), true);
/// })
/// ```
impl<M: EmbeddingModel, T: Embeddable + Send + Sync + Clone> EmbeddingsBuilder<M, T> {
    /// Generate embeddings for all documents in the builder.
    /// Returns a vector of tuples, where the first element is the document and the second element is the embeddings (either one embedding or many).
    pub async fn build(self) -> Result<Vec<(T, OneOrMany<Embedding>)>, EmbeddingError> {
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
                stream::iter(
                    embed_targets
                        .clone()
                        .into_iter()
                        .map(move |target| (i, target)),
                )
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
                |mut acc: HashMap<_, OneOrMany<Embedding>>, embeddings| async move {
                    embeddings.into_iter().for_each(|(i, embedding)| {
                        acc.entry(i)
                            .and_modify(|embeddings| embeddings.push(embedding.clone()))
                            .or_insert(OneOrMany::one(embedding.clone()));
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

#[cfg(test)]
mod tests {
    use crate::{
        embeddings::{embeddable::EmbeddableError, Embedding, EmbeddingModel},
        Embeddable,
    };

    use super::EmbeddingsBuilder;

    #[derive(Clone)]
    struct FakeModel;

    impl EmbeddingModel for FakeModel {
        const MAX_DOCUMENTS: usize = 5;

        fn ndims(&self) -> usize {
            10
        }

        async fn embed_documents(
            &self,
            documents: Vec<String>,
        ) -> Result<Vec<crate::embeddings::Embedding>, crate::embeddings::EmbeddingError> {
            Ok(documents
                .iter()
                .map(|doc| Embedding {
                    document: doc.to_string(),
                    vec: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                })
                .collect())
        }
    }

    #[derive(Clone, Debug)]
    struct FakeDefinition {
        id: String,
        definitions: Vec<String>,
    }

    impl Embeddable for FakeDefinition {
        type Error = EmbeddableError;

        fn embeddable(&self) -> Result<crate::OneOrMany<String>, Self::Error> {
            crate::OneOrMany::many(self.definitions.clone()).map_err(EmbeddableError::new)
        }
    }

    fn fake_definitions() -> Vec<FakeDefinition> {
        vec![
            FakeDefinition {
                id: "doc0".to_string(),
                definitions: vec![
                    "A green alien that lives on cold planets.".to_string(),
                    "A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
                ]
            },
            FakeDefinition {
                id: "doc1".to_string(),
                definitions: vec![
                    "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
                    "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
                ]
            }
        ]
    }

    fn fake_definitions_2() -> Vec<FakeDefinition> {
        vec![
            FakeDefinition {
                id: "doc2".to_string(),
                definitions: vec!["Another fake definitions".to_string()],
            },
            FakeDefinition {
                id: "doc3".to_string(),
                definitions: vec!["Some fake definition".to_string()],
            },
        ]
    }

    #[derive(Clone, Debug)]
    struct FakeDefinitionSingle {
        id: String,
        definition: String,
    }

    impl Embeddable for FakeDefinitionSingle {
        type Error = EmbeddableError;

        fn embeddable(&self) -> Result<crate::OneOrMany<String>, Self::Error> {
            Ok(crate::OneOrMany::one(self.definition.clone()))
        }
    }

    fn fake_definitions_single() -> Vec<FakeDefinitionSingle> {
        vec![
            FakeDefinitionSingle {
                id: "doc0".to_string(),
                definition: "A green alien that lives on cold planets.".to_string(),
            },
            FakeDefinitionSingle {
                id: "doc1".to_string(),
                definition: "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
            }
        ]
    }

    #[tokio::test]
    async fn test_build_many() {
        let fake_definitions = fake_definitions();

        let fake_model = FakeModel;
        let mut result = EmbeddingsBuilder::new(fake_model)
            .documents(fake_definitions)
            .unwrap()
            .build()
            .await
            .unwrap();

        result.sort_by(|(fake_definition_1, _), (fake_definition_2, _)| {
            fake_definition_1.id.cmp(&fake_definition_2.id)
        });

        assert_eq!(result.len(), 2);

        let first_definition = &result[0];
        assert_eq!(first_definition.0.id, "doc0");
        assert_eq!(first_definition.1.len(), 2);
        assert_eq!(
            first_definition.1.first().document,
            "A green alien that lives on cold planets.".to_string()
        );

        let second_definition = &result[1];
        assert_eq!(second_definition.0.id, "doc1");
        assert_eq!(second_definition.1.len(), 2);
        assert_eq!(
            second_definition.1.rest()[0].document, "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
        )
    }

    #[tokio::test]
    async fn test_build_single() {
        let fake_definitions = fake_definitions_single();

        let fake_model = FakeModel;
        let mut result = EmbeddingsBuilder::new(fake_model)
            .documents(fake_definitions)
            .unwrap()
            .build()
            .await
            .unwrap();

        result.sort_by(|(fake_definition_1, _), (fake_definition_2, _)| {
            fake_definition_1.id.cmp(&fake_definition_2.id)
        });

        assert_eq!(result.len(), 2);

        let first_definition = &result[0];
        assert_eq!(first_definition.0.id, "doc0");
        assert_eq!(first_definition.1.len(), 1);
        assert_eq!(
            first_definition.1.first().document,
            "A green alien that lives on cold planets.".to_string()
        );

        let second_definition = &result[1];
        assert_eq!(second_definition.0.id, "doc1");
        assert_eq!(second_definition.1.len(), 1);
        assert_eq!(
            second_definition.1.first().document, "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string()
        )
    }

    #[tokio::test]
    async fn test_build_many_and_single() {
        let fake_definitions = fake_definitions();
        let fake_definitions_single = fake_definitions_2();

        let fake_model = FakeModel;
        let mut result = EmbeddingsBuilder::new(fake_model)
            .documents(fake_definitions)
            .unwrap()
            .documents(fake_definitions_single)
            .unwrap()
            .build()
            .await
            .unwrap();

        result.sort_by(|(fake_definition_1, _), (fake_definition_2, _)| {
            fake_definition_1.id.cmp(&fake_definition_2.id)
        });

        assert_eq!(result.len(), 4);

        let second_definition = &result[1];
        assert_eq!(second_definition.0.id, "doc1");
        assert_eq!(second_definition.1.len(), 2);
        assert_eq!(
            second_definition.1.first().document, "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string()
        );

        let third_definition = &result[2];
        assert_eq!(third_definition.0.id, "doc2");
        assert_eq!(third_definition.1.len(), 1);
        assert_eq!(
            third_definition.1.first().document,
            "Another fake definitions".to_string()
        )
    }
}

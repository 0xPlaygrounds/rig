//! The module defines the [EmbeddingsBuilder] struct which accumulates objects to be embedded
//! and batch generates the embeddings for each object when built.
//! Only types that implement the [Embed] trait can be added to the [EmbeddingsBuilder].

use std::{cmp::max, collections::HashMap};

use futures::{stream, StreamExt};

use crate::{
    embeddings::{
        embed::TextEmbedder, Embed, EmbedError, Embedding, EmbeddingError, EmbeddingModel,
    },
    OneOrMany,
};

/// Builder for creating embeddings from one or more documents of type `T`.
/// Note: `T` can be any type that implements the [Embed] trait.
///
/// Using the builder is preferred over using [EmbeddingModel::embed_text] directly as
/// it will batch the documents in a single request to the model provider.
///
/// # Example
/// ```rust
/// use std::env;
///
/// use rig::{
///     embeddings::EmbeddingsBuilder,
///     providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
/// };
/// use serde::{Deserialize, Serialize};
///
/// // Create OpenAI client
/// let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
/// let openai_client = Client::new(&openai_api_key);
///
/// let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);
///
/// let embeddings = EmbeddingsBuilder::new(model.clone())
///     .documents(vec![
///         "1. *flurbo* (noun): A green alien that lives on cold planets.".to_string(),
///         "2. *flurbo* (noun): A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
///         "1. *glarb-glarb* (noun): An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
///         "2. *glarb-glarb* (noun): A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
///         "1. *linlingdong* (noun): A term used by inhabitants of the sombrero galaxy to describe humans.".to_string(),
///         "2. *linlingdong* (noun): A rare, mystical instrument crafted by the ancient monks of the Nebulon Mountain Ranges on the planet Quarm.".to_string()
///     ])?
///     .build()
///     .await?;
/// ```
pub struct EmbeddingsBuilder<M: EmbeddingModel, T: Embed> {
    model: M,
    documents: Vec<(T, Vec<String>)>,
}

impl<M: EmbeddingModel, T: Embed> EmbeddingsBuilder<M, T> {
    /// Create a new embedding builder with the given embedding model
    pub fn new(model: M) -> Self {
        Self {
            model,
            documents: vec![],
        }
    }

    /// Add a document to be embedded to the builder. `document` must implement the [Embed] trait.
    pub fn document(mut self, document: T) -> Result<Self, EmbedError> {
        let mut embedder = TextEmbedder::default();
        document.embed(&mut embedder)?;

        self.documents.push((document, embedder.texts));

        Ok(self)
    }

    /// Add multiple documents to be embedded to the builder. `documents` must be iterable
    /// with items that implement the [Embed] trait.
    pub fn documents(self, documents: impl IntoIterator<Item = T>) -> Result<Self, EmbedError> {
        let builder = documents
            .into_iter()
            .try_fold(self, |builder, doc| builder.document(doc))?;

        Ok(builder)
    }
}

impl<M: EmbeddingModel, T: Embed + Send> EmbeddingsBuilder<M, T> {
    /// Generate embeddings for all documents in the builder.
    /// Returns a vector of tuples, where the first element is the document and the second element is the embeddings (either one embedding or many).
    pub async fn build(self) -> Result<Vec<(T, OneOrMany<Embedding>)>, EmbeddingError> {
        use stream::TryStreamExt;

        // Store the documents and their texts in a HashMap for easy access.
        let mut docs = HashMap::new();
        let mut texts = Vec::new();

        // Iterate over all documents in the builder and insert their docs and texts into the lookup stores.
        for (i, (doc, doc_texts)) in self.documents.into_iter().enumerate() {
            docs.insert(i, doc);
            texts.push((i, doc_texts));
        }

        // Compute the embeddings.
        let mut embeddings = stream::iter(texts.into_iter())
            // Merge the texts of each document into a single list of texts.
            .flat_map(|(i, texts)| stream::iter(texts.into_iter().map(move |text| (i, text))))
            // Chunk them into batches. Each batch size is at most the embedding API limit per request.
            .chunks(M::MAX_DOCUMENTS)
            // Generate the embeddings for each batch.
            .map(|text| async {
                let (ids, docs): (Vec<_>, Vec<_>) = text.into_iter().unzip();

                let embeddings = self.model.embed_texts(docs).await?;
                Ok::<_, EmbeddingError>(ids.into_iter().zip(embeddings).collect::<Vec<_>>())
            })
            // Parallelize the embeddings generation over 10 concurrent requests
            .buffer_unordered(max(1, 1024 / M::MAX_DOCUMENTS))
            // Collect the embeddings into a HashMap.
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
            .await?;

        // Merge the embeddings with their respective documents
        Ok(docs
            .into_iter()
            .map(|(i, doc)| {
                (
                    doc,
                    embeddings.remove(&i).expect("Document should be present"),
                )
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        embeddings::{embed::EmbedError, embed::TextEmbedder, Embedding, EmbeddingModel},
        Embed,
    };

    use super::EmbeddingsBuilder;

    #[derive(Clone)]
    struct Model;

    impl EmbeddingModel for Model {
        const MAX_DOCUMENTS: usize = 5;

        fn ndims(&self) -> usize {
            10
        }

        async fn embed_texts(
            &self,
            documents: impl IntoIterator<Item = String> + Send,
        ) -> Result<Vec<crate::embeddings::Embedding>, crate::embeddings::EmbeddingError> {
            Ok(documents
                .into_iter()
                .map(|doc| Embedding {
                    document: doc.to_string(),
                    vec: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                })
                .collect())
        }
    }

    #[derive(Clone, Debug)]
    struct WordDefinition {
        id: String,
        definitions: Vec<String>,
    }

    impl Embed for WordDefinition {
        fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
            for definition in &self.definitions {
                embedder.embed(definition.clone());
            }
            Ok(())
        }
    }

    fn definitions_multiple_text() -> Vec<WordDefinition> {
        vec![
            WordDefinition {
                id: "doc0".to_string(),
                definitions: vec![
                    "A green alien that lives on cold planets.".to_string(),
                    "A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
                ]
            },
            WordDefinition {
                id: "doc1".to_string(),
                definitions: vec![
                    "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
                    "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
                ]
            }
        ]
    }

    fn definitions_multiple_text_2() -> Vec<WordDefinition> {
        vec![
            WordDefinition {
                id: "doc2".to_string(),
                definitions: vec!["Another fake definitions".to_string()],
            },
            WordDefinition {
                id: "doc3".to_string(),
                definitions: vec!["Some fake definition".to_string()],
            },
        ]
    }

    #[derive(Clone, Debug)]
    struct WordDefinitionSingle {
        id: String,
        definition: String,
    }

    impl Embed for WordDefinitionSingle {
        fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
            embedder.embed(self.definition.clone());
            Ok(())
        }
    }

    fn definitions_single_text() -> Vec<WordDefinitionSingle> {
        vec![
            WordDefinitionSingle {
                id: "doc0".to_string(),
                definition: "A green alien that lives on cold planets.".to_string(),
            },
            WordDefinitionSingle {
                id: "doc1".to_string(),
                definition: "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
            }
        ]
    }

    #[tokio::test]
    async fn test_build_multiple_text() {
        let fake_definitions = definitions_multiple_text();

        let fake_model = Model;
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
    async fn test_build_single_text() {
        let fake_definitions = definitions_single_text();

        let fake_model = Model;
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
    async fn test_build_multiple_and_single_text() {
        let fake_definitions = definitions_multiple_text();
        let fake_definitions_single = definitions_multiple_text_2();

        let fake_model = Model;
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

    #[tokio::test]
    async fn test_build_string() {
        let bindings = definitions_multiple_text();
        let fake_definitions = bindings.iter().map(|def| def.definitions.clone());

        let fake_model = Model;
        let mut result = EmbeddingsBuilder::new(fake_model)
            .documents(fake_definitions)
            .unwrap()
            .build()
            .await
            .unwrap();

        result.sort_by(|(fake_definition_1, _), (fake_definition_2, _)| {
            fake_definition_1.cmp(&fake_definition_2)
        });

        assert_eq!(result.len(), 2);

        let first_definition = &result[0];
        assert_eq!(first_definition.1.len(), 2);
        assert_eq!(
            first_definition.1.first().document,
            "A green alien that lives on cold planets.".to_string()
        );

        let second_definition = &result[1];
        assert_eq!(second_definition.1.len(), 2);
        assert_eq!(
            second_definition.1.rest()[0].document, "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
        )
    }
}

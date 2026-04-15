//! The module defines the [EmbeddingsBuilder] struct which accumulates objects to be embedded
//! and batch generates the embeddings for each object when built.
//! Only types that implement the [Embed] trait can be added to the [EmbeddingsBuilder].

use std::{cmp::max, collections::HashMap};

use futures::{StreamExt, stream};

use crate::{
    OneOrMany,
    embeddings::{
        Embed, EmbedError, Embedding, EmbeddingError, EmbeddingModel, embed::TextEmbedder,
    },
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
#[non_exhaustive]
pub struct EmbeddingsBuilder<M, T>
where
    M: EmbeddingModel,
    T: Embed,
{
    model: M,
    documents: Vec<(T, Vec<String>)>,
}

impl<M, T> EmbeddingsBuilder<M, T>
where
    M: EmbeddingModel,
    T: Embed,
{
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

impl<M, T> EmbeddingsBuilder<M, T>
where
    M: EmbeddingModel,
    T: Embed + Send,
{
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

        // Flatten the texts while keeping track of the document index.
        let mut flat_texts = Vec::new();
        for (i, doc_texts) in texts.into_iter() {
            for text in doc_texts {
                flat_texts.push((i, text));
            }
        }

        let max_documents = M::MAX_DOCUMENTS;
        let max_tokens = self.model.max_tokens_per_request().unwrap_or(usize::MAX);

        // Group them into batches.
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_tokens = 0;

        for (i, text) in flat_texts {
            // Simple KISS estimate: bytes = tokens (upper bound)
            let text_tokens = text.len();

            // Check if adding this text would exceed the limit
            if !current_batch.is_empty()
                && (current_batch.len() >= max_documents
                    || current_tokens + text_tokens > max_tokens)
            {
                batches.push(current_batch);
                current_batch = Vec::new();
                current_tokens = 0;
            }

            current_tokens += text_tokens;
            current_batch.push((i, text));
        }
        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        // Compute the embeddings.
        let mut embeddings = stream::iter(batches.into_iter())
            // Generate the embeddings for each batch.
            .map(|batch| async {
                let (ids, docs): (Vec<_>, Vec<_>) = batch.into_iter().unzip();

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
        Embed,
        client::Nothing,
        embeddings::{
            Embedding, EmbeddingModel,
            embed::{EmbedError, TextEmbedder},
        },
    };

    use super::EmbeddingsBuilder;

    #[derive(Clone)]
    struct MockEmbeddingModel;

    impl EmbeddingModel for MockEmbeddingModel {
        const MAX_DOCUMENTS: usize = 5;

        type Client = Nothing;

        fn make(_: &Self::Client, _: impl Into<String>, _: Option<usize>) -> Self {
            Self {}
        }

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

        let fake_model = MockEmbeddingModel;
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

        let fake_model = MockEmbeddingModel;
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

        let fake_model = MockEmbeddingModel;
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

        let fake_model = MockEmbeddingModel;
        let mut result = EmbeddingsBuilder::new(fake_model)
            .documents(fake_definitions)
            .unwrap()
            .build()
            .await
            .unwrap();

        result.sort_by(|(fake_definition_1, _), (fake_definition_2, _)| {
            fake_definition_1.cmp(fake_definition_2)
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

    #[derive(Clone)]
    struct LimitModel;

    impl EmbeddingModel for LimitModel {
        const MAX_DOCUMENTS: usize = 100;

        type Client = Nothing;

        fn make(_: &Self::Client, _: impl Into<String>, _: Option<usize>) -> Self {
            Self
        }

        fn max_tokens_per_request(&self) -> Option<usize> {
            Some(10)
        }

        fn ndims(&self) -> usize {
            10
        }

        async fn embed_texts(
            &self,
            documents: impl IntoIterator<Item = String> + Send,
        ) -> Result<Vec<crate::embeddings::Embedding>, crate::embeddings::EmbeddingError> {
            let docs: Vec<String> = documents.into_iter().collect();
            let total_len: usize = docs.iter().map(|s| s.len()).sum();
            if total_len > 10 {
                return Err(crate::embeddings::EmbeddingError::ProviderError(
                    "Too many tokens".to_string(),
                ));
            }
            Ok(docs
                .iter()
                .map(|d| Embedding {
                    document: d.clone(),
                    vec: vec![0.0; 10],
                })
                .collect())
        }
    }

    #[tokio::test]
    async fn test_build_respects_token_limit() {
        let docs = vec![
            WordDefinitionSingle {
                id: "1".into(),
                definition: "hello".into(),
            },
            WordDefinitionSingle {
                id: "2".into(),
                definition: "world!".into(),
            },
        ];

        let model = LimitModel;
        // This should pass if batching splits "hello" and "world!"
        let result = EmbeddingsBuilder::new(model)
            .documents(docs)
            .unwrap()
            .build()
            .await;

        assert!(result.is_ok(), "Build failed: {:?}", result.err());
        let result = result.unwrap();
        assert_eq!(result.len(), 2);
    }
}

use std::{cmp::max, collections::HashMap};

use anyhow::Result;
use futures::{stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};

use crate::tool::{ToolEmbedding, ToolSet, ToolType};

pub trait EmbeddingModel: Clone + Sync {
    const MAX_DOCUMENTS: usize;

    fn embed_document(
        &self,
        document: &str,
    ) -> impl std::future::Future<Output = Result<Embedding>> + Send
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

    fn embed_documents(
        &self,
        documents: Vec<String>,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>>> + Send;
}

#[derive(Clone, Default, Deserialize, Serialize)]
pub struct Embedding {
    pub document: String,
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

/// Struct to store the document and its embeddings.
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

    /// Add a document to the embedding collection.
    /// The document will be used for the embedding.
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

    pub fn tool(mut self, tool: impl ToolEmbedding + Send + Sync + 'static) -> Result<Self> {
        self.documents.push((
            tool.name(),
            serde_json::to_value(tool.context())?,
            tool.embedding_docs(),
        ));
        Ok(self)
    }

    pub fn tools(mut self, toolset: &ToolSet) -> Result<Self> {
        for (name, tool) in toolset.tools.iter() {
            if let ToolType::Embedding(tool) = tool {
                self.documents
                    .push((name.clone(), tool.context()?, tool.embedding_docs()));
            }
        }
        Ok(self)
    }

    /// Add a document to the embedding collection.
    /// `embed_documents` are the documents that will be used to generate the embeddings
    /// for `document`.
    pub fn document(
        mut self,
        id: &str,
        document: serde_json::Value,
        embed_documents: Vec<String>,
    ) -> Self {
        self.documents
            .push((id.to_string(), document, embed_documents));
        self
    }

    /// Add multiple documents to the embedding collection.
    /// Each element of the vector is a tuple of the form (id, document, embed_documents).
    pub fn documents(mut self, documents: Vec<(String, serde_json::Value, Vec<String>)>) -> Self {
        self.documents.extend(documents);
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
    pub async fn build(self) -> Result<Embeddings> {
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
                anyhow::Ok(
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

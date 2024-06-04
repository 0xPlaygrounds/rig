use anyhow::Result;
use serde::Deserialize;

use crate::embeddings::{DocumentEmbeddings, Embedding};

pub mod in_memory_store;

pub trait VectorStore {
    type Q;

    fn add_documents(
        &mut self,
        documents: Vec<DocumentEmbeddings>,
    ) -> impl std::future::Future<Output = Result<()>> + Send;

    fn get_document_embeddings(
        &self,
        id: &str,
    ) -> impl std::future::Future<Output = Result<Option<DocumentEmbeddings>>> + Send;

    fn get_document<T: for<'a> Deserialize<'a>>(
        &self,
        id: &str,
    ) -> impl std::future::Future<Output = Result<Option<T>>> + Send;

    fn get_document_by_query(
        &self,
        query: Self::Q,
    ) -> impl std::future::Future<Output = Result<Option<DocumentEmbeddings>>> + Send;
}

pub trait VectorStoreIndex: Sync {
    fn embed_document(
        &self,
        document: &str,
    ) -> impl std::future::Future<Output = Result<Embedding>> + Send;

    /// Get the top n documents based on the distance to the given embedding.
    /// The distance is calculated as the cosine distance between the prompt and
    /// the document embedding.
    /// The result is a list of tuples with the distance and the document.
    fn top_n_from_query(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, DocumentEmbeddings)>>> + Send;

    /// Same as `top_n_from_query` but returns the documents without its embeddings.
    /// The documents are deserialized into the given type.
    fn top_n_documents_from_query<T: for<'a> Deserialize<'a>>(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, T)>>> + Send {
        async move {
            let documents = self.top_n_from_query(query, n).await?;
            Ok(documents
                .into_iter()
                .map(|(distance, doc)| (distance, serde_json::from_value(doc.document).unwrap()))
                .collect())
        }
    }

    /// Same as `top_n_from_query` but returns the document ids only.
    fn top_n_ids_from_query(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, String)>>> + Send {
        async move {
            let documents = self.top_n_from_query(query, n).await?;
            Ok(documents
                .into_iter()
                .map(|(distance, doc)| (distance, doc.id))
                .collect())
        }
    }

    /// Get the top n documents based on the distance to the given embedding.
    /// The distance is calculated as the cosine distance between the prompt and
    /// the document embedding.
    /// The result is a list of tuples with the distance and the document.
    fn top_n_from_embedding(
        &self,
        prompt_embedding: &Embedding,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, DocumentEmbeddings)>>> + Send;

    /// Same as `top_n_from_embedding` but returns the documents without its embeddings.
    /// The documents are deserialized into the given type.
    fn top_n_documents_from_embedding<T: for<'a> Deserialize<'a>>(
        &self,
        prompt_embedding: &Embedding,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, T)>>> + Send {
        async move {
            let documents = self.top_n_from_embedding(prompt_embedding, n).await?;
            Ok(documents
                .into_iter()
                .map(|(distance, doc)| (distance, serde_json::from_value(doc.document).unwrap()))
                .collect())
        }
    }

    /// Same as `top_n_from_embedding` but returns the document ids only.
    fn top_n_ids_from_embedding(
        &self,
        prompt_embedding: &Embedding,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, String)>>> + Send {
        async move {
            let documents = self.top_n_from_embedding(prompt_embedding, n).await?;
            Ok(documents
                .into_iter()
                .map(|(distance, doc)| (distance, doc.id))
                .collect())
        }
    }
}

pub struct NoIndex;

impl VectorStoreIndex for NoIndex {
    async fn embed_document(&self, _document: &str) -> Result<Embedding> {
        Ok(Embedding::default())
    }

    async fn top_n_from_query(
        &self,
        _query: &str,
        _n: usize,
    ) -> Result<Vec<(f64, DocumentEmbeddings)>> {
        Ok(vec![])
    }

    async fn top_n_from_embedding(
        &self,
        _prompt_embedding: &Embedding,
        _n: usize,
    ) -> Result<Vec<(f64, DocumentEmbeddings)>> {
        Ok(vec![])
    }
}

use std::sync::Arc;

use lancedb::{
    arrow::arrow_schema::{DataType, Field, Fields, Schema},
    index::Index,
    query::QueryBase,
    DistanceType,
};
use rig::{
    embeddings::EmbeddingModel,
    vector_store::{VectorStore, VectorStoreError, VectorStoreIndex},
};
use serde::{Deserialize, Serialize};
use table_schemas::{document::DocumentRecords, embedding::EmbeddingRecordsBatch, merge};
use utils::{Insert, Query};

mod table_schemas;
mod utils;

pub struct LanceDbVectorStore {
    document_table: lancedb::Table,
    embedding_table: lancedb::Table,
    embedding_dimension: i32,
}

impl LanceDbVectorStore {
    /// Note: Tables are created inside the new function rather than created outside and passed as reference to new function.
    /// This is because a specific schema needs to be enforced on the tables and this is done at creation time.
    pub async fn new(
        db: &lancedb::Connection,
        embedding_dimension: i32,
    ) -> Result<Self, lancedb::Error> {
        Ok(Self {
            document_table: db
                .create_empty_table("documents", Arc::new(Self::document_schema()))
                .execute()
                .await?,
            embedding_table: db
                .create_empty_table(
                    "embeddings",
                    Arc::new(Self::embedding_schema(embedding_dimension)),
                )
                .execute()
                .await?,
            embedding_dimension,
        })
    }

    fn document_schema() -> Schema {
        Schema::new(Fields::from(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("document", DataType::Utf8, false),
        ]))
    }

    fn embedding_schema(dimension: i32) -> Schema {
        Schema::new(Fields::from(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("document_id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float64, true)),
                    dimension,
                ),
                false,
            ),
        ]))
    }

    pub fn index<M: EmbeddingModel>(&self, model: M) -> LanceDbVectorIndex<M> {
        LanceDbVectorIndex::new(
            model,
            self.embedding_table.clone(),
            self.document_table.clone(),
        )
    }
}

fn lancedb_to_rig_error(e: lancedb::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

fn serde_to_rig_error(e: serde_json::Error) -> VectorStoreError {
    VectorStoreError::JsonError(e)
}

impl VectorStore for LanceDbVectorStore {
    type Q = lancedb::query::Query;

    async fn add_documents(
        &mut self,
        documents: Vec<rig::embeddings::DocumentEmbeddings>,
    ) -> Result<(), VectorStoreError> {
        let document_records =
            DocumentRecords::try_from(documents.clone()).map_err(serde_to_rig_error)?;

        self.document_table
            .insert(document_records, Self::document_schema())
            .await
            .map_err(lancedb_to_rig_error)?;

        let embedding_records = EmbeddingRecordsBatch::from(documents);

        self.embedding_table
            .insert(
                embedding_records,
                Self::embedding_schema(self.embedding_dimension),
            )
            .await
            .map_err(lancedb_to_rig_error)?;

        Ok(())
    }

    async fn get_document_embeddings(
        &self,
        id: &str,
    ) -> Result<Option<rig::embeddings::DocumentEmbeddings>, VectorStoreError> {
        let documents: DocumentRecords = self
            .document_table
            .query()
            .only_if(format!("id = {id}"))
            .execute_query()
            .await?;

        let embeddings: EmbeddingRecordsBatch = self
            .embedding_table
            .query()
            .only_if(format!("document_id = {id}"))
            .execute_query()
            .await?;

        Ok(merge(&documents, &embeddings)?.into_iter().next())
    }

    async fn get_document<T: for<'a> serde::Deserialize<'a>>(
        &self,
        id: &str,
    ) -> Result<Option<T>, VectorStoreError> {
        let documents: DocumentRecords = self
            .document_table
            .query()
            .only_if(format!("id = {id}"))
            .execute_query()
            .await?;

        let document = documents
            .as_iter()
            .next()
            .map(|document| serde_json::from_str(&document.document).map_err(serde_to_rig_error))
            .transpose();

        document
    }

    async fn get_document_by_query(
        &self,
        query: Self::Q,
    ) -> Result<Option<rig::embeddings::DocumentEmbeddings>, VectorStoreError> {
        let documents: DocumentRecords = query.execute_query().await?;

        let embeddings: EmbeddingRecordsBatch = self
            .embedding_table
            .query()
            .only_if(format!("document_id IN [{}]", documents.ids().join(",")))
            .execute_query()
            .await?;

        Ok(merge(&documents, &embeddings)?.into_iter().next())
    }
}

/// A vector index for a MongoDB collection.
pub struct LanceDbVectorIndex<M: EmbeddingModel> {
    model: M,
    embedding_table: lancedb::Table,
    document_table: lancedb::Table,
}

impl<M: EmbeddingModel> LanceDbVectorIndex<M> {
    pub fn new(model: M, embedding_table: lancedb::Table, document_table: lancedb::Table) -> Self {
        Self {
            model,
            embedding_table,
            document_table,
        }
    }

    pub async fn create_index(&self, index: Index) -> Result<(), lancedb::Error> {
        self.embedding_table
            .create_index(&["embedding"], index)
            .execute()
            .await?;

        Ok(())
    }
}

/// See [LanceDB vector search](https://lancedb.github.io/lancedb/search/) for more information.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum SearchType {
    // Flat search, also called ENN or kNN.
    Flat,
    /// Approximal Nearest Neighbor search, also called ANN.
    Approximate,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SearchParams {
    /// Always set the distance_type to match the value used to train the index
    /// By default, set to L2
    distance_type: Option<DistanceType>,
    /// By default, ANN will be used if there is an index on the table.
    /// By default, kNN will be used if there is NO index on the table.
    /// To use defaults, set to None.
    search_type: Option<SearchType>,
    /// Set this value only when search type is ANN.
    /// See [LanceDb ANN Search](https://lancedb.github.io/lancedb/ann_indexes/#querying-an-ann-index) for more information
    nprobes: Option<usize>,
    /// Set this value only when search type is ANN.
    /// See [LanceDb ANN Search](https://lancedb.github.io/lancedb/ann_indexes/#querying-an-ann-index) for more information
    refine_factor: Option<u32>,
    /// If set to true, filtering will happen after the vector search instead of before
    /// See [LanceDb pre/post filtering](https://lancedb.github.io/lancedb/sql/#pre-and-post-filtering) for more information
    post_filter: Option<bool>,
}

impl SearchParams {
    pub fn new(
        distance_type: Option<DistanceType>,
        search_type: Option<SearchType>,
        nprobes: Option<usize>,
        refine_factor: Option<u32>,
        post_filter: Option<bool>,
    ) -> Self {
        Self {
            distance_type,
            search_type,
            nprobes,
            refine_factor,
            post_filter,
        }
    }
}

impl<M: EmbeddingModel + std::marker::Sync + Send> VectorStoreIndex for LanceDbVectorIndex<M> {
    async fn top_n_from_query(
        &self,
        query: &str,
        n: usize,
        search_params: Self::SearchParams,
    ) -> Result<Vec<(f64, rig::embeddings::DocumentEmbeddings)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_document(query).await?;
        self.top_n_from_embedding(&prompt_embedding, n, search_params)
            .await
    }

    async fn top_n_from_embedding(
        &self,
        prompt_embedding: &rig::embeddings::Embedding,
        n: usize,
        search_params: Self::SearchParams,
    ) -> Result<Vec<(f64, rig::embeddings::DocumentEmbeddings)>, VectorStoreError> {
        let SearchParams {
            distance_type,
            search_type,
            nprobes,
            refine_factor,
            post_filter,
        } = search_params.clone();

        let query = self
            .embedding_table
            .vector_search(prompt_embedding.vec.clone())
            .map_err(lancedb_to_rig_error)?
            .limit(n);

        if let Some(distance_type) = distance_type {
            query.clone().distance_type(distance_type);
        }

        if let Some(SearchType::Flat) = search_type {
            query.clone().bypass_vector_index();
        }

        if let Some(SearchType::Approximate) = search_type {
            if let Some(nprobes) = nprobes {
                query.clone().nprobes(nprobes);
            }
            if let Some(refine_factor) = refine_factor {
                query.clone().refine_factor(refine_factor);
            }
        }

        if let Some(true) = post_filter {
            query.clone().postfilter();
        }

        let embeddings: EmbeddingRecordsBatch = query.execute_query().await?;

        let documents: DocumentRecords = self
            .document_table
            .query()
            .only_if(format!(
                "id IN ({})",
                embeddings
                    .document_ids()
                    .iter()
                    .map(|id| format!("'{id}'"))
                    .collect::<Vec<_>>()
                    .join(",")
            ))
            .execute_query()
            .await?;

        let document_embeddings = merge(&documents, &embeddings)?;

        Ok(document_embeddings
            .into_iter()
            .map(|doc| {
                let distance = embeddings
                    .get_by_id(&doc.id)
                    .map(|records| {
                        records
                            .as_iter()
                            .next()
                            .map(|record| record.distance.unwrap_or(0.0))
                            .unwrap_or(0.0)
                    })
                    .unwrap_or(0.0);

                (distance as f64, doc)
            })
            .collect())
    }

    type SearchParams = SearchParams;
}

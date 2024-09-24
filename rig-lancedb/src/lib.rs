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
use table_schemas::{document::DocumentRecords, embedding::EmbeddingRecordsBatch, merge};
use utils::{Insert, Query};

mod table_schemas;
mod utils;

fn lancedb_to_rig_error(e: lancedb::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

fn serde_to_rig_error(e: serde_json::Error) -> VectorStoreError {
    VectorStoreError::JsonError(e)
}

pub struct LanceDbVectorStore<M: EmbeddingModel> {
    /// Defines which model is used to generate embeddings for the vector store
    model: M,
    /// Table containing documents only
    document_table: lancedb::Table,
    /// Table containing embeddings only.
    /// Foreign key references the document in document table.
    embedding_table: lancedb::Table,
    /// Vector search params that are used during vector search operations.
    search_params: SearchParams,
}

/// See [LanceDB vector search](https://lancedb.github.io/lancedb/search/) for more information.
#[derive(Debug, Clone)]
pub enum SearchType {
    // Flat search, also called ENN or kNN.
    Flat,
    /// Approximal Nearest Neighbor search, also called ANN.
    Approximate,
}

#[derive(Debug, Clone, Default)]
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
    pub fn distance_type(mut self, distance_type: DistanceType) -> Self {
        self.distance_type = Some(distance_type);
        self
    }

    pub fn search_type(mut self, search_type: SearchType) -> Self {
        self.search_type = Some(search_type);
        self
    }

    pub fn nprobes(mut self, nprobes: usize) -> Self {
        self.nprobes = Some(nprobes);
        self
    }

    pub fn refine_factor(mut self, refine_factor: u32) -> Self {
        self.refine_factor = Some(refine_factor);
        self
    }

    pub fn post_filter(mut self, post_filter: bool) -> Self {
        self.post_filter = Some(post_filter);
        self
    }
}

impl<M: EmbeddingModel> LanceDbVectorStore<M> {
    /// Note: Tables are created inside the new function rather than created outside and passed as reference to new function.
    /// This is because a specific schema needs to be enforced on the tables and this is done at creation time.
    pub async fn new(
        db: &lancedb::Connection,
        model: &M,
        search_params: &SearchParams,
    ) -> Result<Self, lancedb::Error> {
        let document_table = db
            .create_empty_table("documents", Arc::new(Self::document_schema()))
            .execute()
            .await?;

        let embedding_table = db
            .create_empty_table(
                "embeddings",
                Arc::new(Self::embedding_schema(model.ndims() as i32)),
            )
            .execute()
            .await?;

        Ok(Self {
            document_table,
            embedding_table,
            model: model.clone(),
            search_params: search_params.clone(),
        })
    }

    /// Schema of records in document table.
    fn document_schema() -> Schema {
        Schema::new(Fields::from(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("document", DataType::Utf8, false),
        ]))
    }

    /// Schema of records in embeddings table.
    /// Every embedding vector in the table must have the same size.
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

    /// Define index on document table `id` field for search optimization.
    pub async fn create_document_index(&self, index: Index) -> Result<(), lancedb::Error> {
        self.document_table
            .create_index(&["id"], index)
            .execute()
            .await
    }

    /// Define index on embedding table `id` and `document_id` fields for search optimization.
    pub async fn create_embedding_index(&self, index: Index) -> Result<(), lancedb::Error> {
        self.embedding_table
            .create_index(&["id", "document_id"], index)
            .execute()
            .await
    }

    /// Define index on embedding table `embedding` fields for vector search optimization.
    pub async fn create_index(&self, index: Index) -> Result<(), lancedb::Error> {
        self.embedding_table
            .create_index(&["embedding"], index)
            .execute()
            .await?;

        Ok(())
    }
}

impl<M: EmbeddingModel + std::marker::Sync + Send> VectorStore for LanceDbVectorStore<M> {
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
                Self::embedding_schema(self.model.ndims() as i32),
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
            .only_if(format!("id = '{id}'"))
            .execute_query()
            .await?;

        let embeddings: EmbeddingRecordsBatch = self
            .embedding_table
            .query()
            .only_if(format!("document_id = '{id}'"))
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
            .only_if(format!("id = '{id}'"))
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
            .only_if(format!(
                "document_id IN ({})",
                documents
                    .ids()
                    .map(|id| format!("'{id}'"))
                    .collect::<Vec<_>>()
                    .join(",")
            ))
            .execute_query()
            .await?;

        Ok(merge(&documents, &embeddings)?.into_iter().next())
    }
}

impl<M: EmbeddingModel + std::marker::Sync + Send> VectorStoreIndex for LanceDbVectorStore<M> {
    async fn top_n_from_query(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, rig::embeddings::DocumentEmbeddings)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_document(query).await?;
        self.top_n_from_embedding(&prompt_embedding, n).await
    }

    async fn top_n_from_embedding(
        &self,
        prompt_embedding: &rig::embeddings::Embedding,
        n: usize,
    ) -> Result<Vec<(f64, rig::embeddings::DocumentEmbeddings)>, VectorStoreError> {
        let mut query = self
            .embedding_table
            .vector_search(prompt_embedding.vec.clone())
            .map_err(lancedb_to_rig_error)?
            .limit(n);

        let SearchParams {
            distance_type,
            search_type,
            nprobes,
            refine_factor,
            post_filter,
        } = self.search_params.clone();

        if let Some(distance_type) = distance_type {
            query = query.distance_type(distance_type);
        }

        if let Some(SearchType::Flat) = search_type {
            query = query.bypass_vector_index();
        }

        if let Some(SearchType::Approximate) = search_type {
            if let Some(nprobes) = nprobes {
                query = query.nprobes(nprobes);
            }
            if let Some(refine_factor) = refine_factor {
                query = query.refine_factor(refine_factor);
            }
        }

        if let Some(true) = post_filter {
            query = query.postfilter();
        }

        let embeddings: EmbeddingRecordsBatch = query.execute_query().await?;

        let documents: DocumentRecords = self
            .document_table
            .query()
            .only_if(format!("id IN ({})", embeddings.document_ids()))
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
}

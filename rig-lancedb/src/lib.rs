use lancedb::{arrow::arrow_schema::Schema, query::QueryBase, DistanceType};
use rig::{
    embeddings::EmbeddingModel,
    vector_store::{VectorStore, VectorStoreError, VectorStoreIndex},
};
use serde::Deserialize;
use table_schemas::{document::DocumentRecords, embedding::EmbeddingRecordsBatch, merge};
use utils::{Insert, Query};

mod table_schemas;
mod utils;

pub struct LanceDbVectorStore {
    document_table: lancedb::Table,
    document_schema: Schema,

    embedding_table: lancedb::Table,
    embedding_schema: Schema,
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
            .insert(document_records, self.document_schema.clone())
            .await
            .map_err(lancedb_to_rig_error)?;

        let embedding_records = EmbeddingRecordsBatch::from(documents);

        self.embedding_table
            .insert(embedding_records, self.embedding_schema.clone())
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

        Ok(merge(documents, embeddings)?.into_iter().next())
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

        Ok(merge(documents, embeddings)?.into_iter().next())
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
}

/// See [LanceDB vector search](https://lancedb.github.io/lancedb/search/) for more information.
#[derive(Deserialize)]
pub enum SearchType {
    // Flat search, also called ENN or kNN.
    Flat,
    /// Approximal Nearest Neighbor search, also called ANN.
    Approximate,
}

#[derive(Deserialize)]
pub struct SearchParams {
    /// Always set the distance_type to match the value used to train the index
    distance_type: DistanceType,
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

impl<M: EmbeddingModel + std::marker::Sync + Send> VectorStoreIndex for LanceDbVectorIndex<M> {
    async fn top_n_from_query(
        &self,
        query: &str,
        n: usize,
        search_params: Self::SearchParams,
    ) -> Result<Vec<(f64, rig::embeddings::DocumentEmbeddings)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_document(query).await?;
        self.top_n_from_embedding(&prompt_embedding, n, search_params).await
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
        } = search_params;

        let query = self
            .embedding_table
            .vector_search(prompt_embedding.vec.clone())
            .map_err(lancedb_to_rig_error)?
            .distance_type(distance_type)
            .limit(n);

        if let Some(SearchType::Flat) = &search_type {
            query.clone().bypass_vector_index();
        }

        if let Some(SearchType::Approximate) = &search_type {
            if let Some(nprobes) = nprobes {
                query.clone().nprobes(nprobes);
            }
            if let Some(refine_factor) = refine_factor {
                query.clone().refine_factor(refine_factor);
            }
        }

        if let Some(true) = &post_filter {
            query.clone().postfilter();
        }

        let embeddings: EmbeddingRecordsBatch = query.execute_query().await?;

        let documents: DocumentRecords = self
            .document_table
            .query()
            .only_if(format!("id IN [{}]", embeddings.document_ids().join(",")))
            .execute_query()
            .await?;

        // Todo: get distances for each returned vector

        merge(documents, embeddings)?;

        todo!()
    }

    type SearchParams = SearchParams;
}

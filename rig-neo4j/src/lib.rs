use neo4rs::*;
use std::fmt::Debug;

use rig::{
    embeddings::{DocumentEmbeddings, Embedding, EmbeddingModel},
    vector_store::{VectorStore, VectorStoreError, VectorStoreIndex},
};
use serde::Deserialize;

pub struct Neo4jClient {
    pub graph: Graph,
}

/// A Neo4j vector store.
pub struct Neo4jVectorStore {
    //collection: mongodb::Collection<DocumentEmbeddings>,
    pub client: Neo4jClient,
    pub database_name: String,
}

fn neo4j_to_rig_error(e: neo4rs::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

trait ToBoltType {
    fn to_bolt_type(&self) -> BoltType;
}

impl ToBoltType for Embedding {
    fn to_bolt_type(&self) -> BoltType {
        let mut bolt_map = BoltMap::new();
        let bolt_list = BoltType::List(
            self.vec
                .iter()
                .map(|&f| BoltType::Float(BoltFloat::new(f)))
                .collect::<Vec<BoltType>>()
                .into(),
        );
        bolt_map.put(BoltString::new(&self.document), bolt_list);
        BoltType::Map(bolt_map)
    }
}

impl Neo4jClient {
    pub fn new(graph: Graph) -> Self {
        Self { graph }
    }

    pub async fn connect(uri: &str, user: &str, password: &str) -> Result<Self, VectorStoreError> {
        let graph = Graph::new(uri, user, password)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
        Ok(Self { graph })
    }
}

impl VectorStore for Neo4jVectorStore {
    type Q = neo4rs::Query;

    async fn add_documents(
        &mut self,
        documents: Vec<DocumentEmbeddings>,
    ) -> Result<(), VectorStoreError> {
        for doc in documents {
            let query = Query::new("CREATE (d:DocumentEmbeddings {id: $id, document: $document, embeddings: $embeddings})".to_string())
                .param("id", doc.id)
                .param("document", doc.document.to_string())
                .param("embeddings", doc.embeddings.iter().map(|e| e.to_bolt_type()).collect::<Vec<BoltType>>());
            self.client
                .graph
                .run(query)
                .await
                .map_err(neo4j_to_rig_error)?;
        }
        Ok(())
    }

    async fn get_document_embeddings(
        &self,
        _id: &str,
    ) -> Result<Option<DocumentEmbeddings>, VectorStoreError> {
        // TODO: Implement
        Ok(None)
    }

    async fn get_document<T: for<'a> serde::Deserialize<'a>>(
        &self,
        _id: &str,
    ) -> Result<Option<T>, VectorStoreError> {
        // TODO: Implement
        Ok(None)
    }

    async fn get_document_by_query(
        &self,
        _query: Self::Q,
    ) -> Result<Option<DocumentEmbeddings>, VectorStoreError> {
        // TODO: Implement
        Ok(None)
    }
}

impl Neo4jVectorStore {
    pub fn new(client: Neo4jClient, database_name: &str) -> Self {
        Self {
            client,
            database_name: database_name.to_string(),
        }
    }
    /// Creates a `Neo4jVectorIndex` that mirrors an existing Neo4j Vector Index.
    ///
    /// An index (of type "vector") of the same name as `index_name` must already exist for the Neo4j database.
    /// See the Neo4j [documentation (Create vector index)](https://neo4j.com/docs/genai/tutorials/embeddings-vector-indexes/setup/vector-index/) for more information on creating indexes.
    ///
    /// ❗IMPORTANT: The index must be created with the same embedding model that will be used to query the index.
    pub fn index<M: EmbeddingModel>(
        &self,
        model: M,
        index_name: &str,
        search_params: SearchParams,
    ) -> Neo4jVectorIndex<M> {
        Neo4jVectorIndex::new(self.client.graph.clone(), model, index_name, search_params)
    }
}

/// A vector index for a Neo4j graph.
pub struct Neo4jVectorIndex<M: EmbeddingModel> {
    //collection: mongodb::Collection<DocumentEmbeddings>,
    graph: Graph,
    embedding_model: M,
    index_name: String,
    search_params: SearchParams,
}

impl<M: EmbeddingModel> Neo4jVectorIndex<M> {
    const BASE_VECTOR_SEARCH_QUERY: &str = "
    CALL db.index.vector.queryNodes($index_name, $num_candidates, $queryVector)
    YIELD node, score
    ";

    pub fn new(
        graph: Graph,
        embedding_model: M,
        index_name: &str,
        search_params: SearchParams,
    ) -> Self {
        Self {
            graph,
            embedding_model,
            index_name: index_name.to_string(),
            search_params,
        }
    }

    /// Build a Neo4j query that performs a vector search against an index.
    /// See [Query vector index](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/#query-vector-index) for more information.
    pub fn build_vector_search_query(
        &self,
        prompt_embedding: Embedding,
        return_node: bool,
        n: usize,
    ) -> Query {
        let where_clause = match &self.search_params.post_vector_search_filter {
            Some(filter) => format!("WHERE {}", filter),
            None => "".to_string(),
        };

        Query::new(format!(
            "
            {}
            {}
            RETURN score, ID(node) as element_id {}
            ",
            Self::BASE_VECTOR_SEARCH_QUERY,
            where_clause,
            if return_node { ", node as node" } else { "" }
        ))
        .param("queryVector", prompt_embedding.vec)
        .param("num_candidates", n as i64)
        .param("index_name", self.index_name.clone())
    }

    pub async fn execute_and_collect<T: for<'a> Deserialize<'a>>(
        &self,
        query: Query,
    ) -> Result<Vec<T>, VectorStoreError> {
        let mut result = self
            .graph
            .execute(query)
            .await
            .map_err(neo4j_to_rig_error)?;
        let mut results = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            let row_parsed = row
                .to::<T>()
                .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

            results.push(row_parsed);
        }
        Ok(results)
    }
}

/// Search parameters for a vector search. Neo4j currently only supports post-vector-search filtering.
pub struct SearchParams {
    /// Sets the **post-filter** field of the search params. Uses a WHERE clause.
    /// See [Neo4j WHERE clause](https://neo4j.com/docs/cypher-manual/current/clauses/where/) for more information.
    post_vector_search_filter: Option<String>,
}

impl SearchParams {
    /// Initializes a new `SearchParams` with default values.
    pub fn new(filter: Option<String>) -> Self {
        Self {
            post_vector_search_filter: filter,
        }
    }

    pub fn filter(mut self, filter: String) -> Self {
        self.post_vector_search_filter = Some(filter);
        self
    }
}

impl Default for SearchParams {
    fn default() -> Self {
        Self::new(None)
    }
}

impl<M: EmbeddingModel + std::marker::Sync + Send> VectorStoreIndex for Neo4jVectorIndex<M> {
    /// Get the top n nodes and scores matching the query.
    ///
    /// #### Generic Type Parameters
    ///
    /// - `T`: The type used to deserialize the result from the Neo4j query.
    ///        It must implement the `serde::Deserialize` trait.
    ///
    /// #### Returns
    ///
    /// Returns a `Result` containing a vector of tuples. Each tuple contains:
    /// - A `f64` representing the similarity score
    /// - A `String` representing the node ID
    /// - A value of type `T` representing the deserialized node data
    ///
    async fn top_n<T: for<'a> Deserialize<'a> + std::marker::Send>(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let prompt_embedding = self.embedding_model.embed_document(query).await?;
        let query = self.build_vector_search_query(prompt_embedding, true, n);

        #[derive(Debug, Deserialize)]
        struct RowResult<T> {
            node: T,
            score: f64,
            element_id: i64,
        }

        let rows = self.execute_and_collect::<RowResult<T>>(query).await?;

        let results = rows
            .into_iter()
            .map(|row| (row.score, row.element_id.to_string(), row.node))
            .collect::<Vec<_>>();

        Ok(results)
    }

    /// Get the top n ids and scores matching the query. Runs faster than top_n since it doesn't need to transfer and parse
    /// the full nodes and embeddings to the client.
    async fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let prompt_embedding = self.embedding_model.embed_document(query).await?;

        let query = self.build_vector_search_query(prompt_embedding, false, n);

        #[derive(Debug, Deserialize)]
        struct RowResult {
            score: f64,
            element_id: i64,
        }

        let rows = self.execute_and_collect::<RowResult>(query).await?;

        let results = rows
            .into_iter()
            .map(|row| (row.score, row.element_id.to_string()))
            .collect::<Vec<_>>();

        Ok(results)
    }
}

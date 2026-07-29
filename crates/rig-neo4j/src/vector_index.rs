//! A vector index for a Neo4j graph DB.
//!
//! This module provides a way to perform vector searches on a Neo4j graph DB.
//! It uses the [Neo4j vector index](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
//! to search for similar nodes based on a query.

use neo4rs::{Graph, Query};
use rig_core::{
    OneOrMany,
    embeddings::Embedding,
    vector_store::{
        SearchHit, StoreRecord, VectorStoreError,
        request::{SearchFilter, VectorSearchRequest},
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned, de::Error};

use crate::{Neo4jClient, Neo4jSearchFilter, ToBoltType};

/// A vector index over a Neo4j vector index.
///
/// Queries arrive pre-embedded via [`VectorSearchRequest`]; the index never
/// embeds text itself.
pub struct Neo4jVectorIndex {
    graph: Graph,
    index_config: IndexConfig,
}

/// The index name must be unique among both indexes and constraints.
/// A newly created index is not immediately available but is created in the background.
///
/// #### Default Values
/// - `index_name`: "vector_index"
/// - `embedding_property`: "embedding"
/// - `similarity_function`: VectorSimilarityFunction::Cosine
/// - `node_label`: None (inserts default to the `Document` label)
#[derive(Serialize, Deserialize, Clone)]
pub struct IndexConfig {
    pub index_name: String,
    pub embedding_property: String,
    pub similarity_function: VectorSimilarityFunction,
    /// The node label that [`Neo4jVectorIndex::insert`] writes to (and that the index
    /// applies to). Populated from the index's `labelsOrTypes` when loaded via
    /// [`Neo4jClient::get_index`](crate::Neo4jClient::get_index).
    pub node_label: Option<String>,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            index_name: "vector_index".to_string(),
            embedding_property: "embedding".to_string(),
            similarity_function: VectorSimilarityFunction::Cosine,
            node_label: None,
        }
    }
}

impl IndexConfig {
    pub fn new(index_name: impl Into<String>) -> Self {
        Self {
            index_name: index_name.into(),
            embedding_property: "embedding".to_string(),
            similarity_function: VectorSimilarityFunction::Cosine,
            node_label: None,
        }
    }

    pub fn index_name(mut self, index_name: &str) -> Self {
        self.index_name = index_name.to_string();
        self
    }

    pub fn similarity_function(mut self, similarity_function: VectorSimilarityFunction) -> Self {
        self.similarity_function = similarity_function;
        self
    }

    pub fn embedding_property(mut self, embedding_property: &str) -> Self {
        self.embedding_property = embedding_property.to_string();
        self
    }

    /// Sets the node label that [`Neo4jVectorIndex::insert`] writes to.
    pub fn node_label(mut self, node_label: &str) -> Self {
        self.node_label = Some(node_label.to_string());
        self
    }
}

/// Cosine is most commonly used, but Euclidean is also supported.
/// See [Neo4j vector similarity functions](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/#similarity-functions)
/// for more information.
#[derive(Default, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum VectorSimilarityFunction {
    #[default]
    Cosine,
    Euclidean,
}

use std::str::FromStr;

impl FromStr for VectorSimilarityFunction {
    type Err = VectorStoreError;

    fn from_str(s: &str) -> Result<Self, VectorStoreError> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(VectorSimilarityFunction::Cosine),
            "euclidean" => Ok(VectorSimilarityFunction::Euclidean),
            _ => Err(VectorStoreError::JsonError(serde_json::Error::custom(
                format!("Invalid similarity function: {s}"),
            ))),
        }
    }
}

const BASE_VECTOR_SEARCH_QUERY: &str = "
    CALL db.index.vector.queryNodes($index_name, $num_candidates, $queryVector)
    YIELD node, score
";

impl Neo4jVectorIndex {
    pub fn new(graph: Graph, index_config: IndexConfig) -> Self {
        Self {
            graph,
            index_config,
        }
    }

    /// Build a Neo4j query that performs a vector search against an index.
    /// See [Query vector index](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/#query-vector-index) for more information.
    ///
    /// Query template:
    /// ```text
    /// CALL db.index.vector.queryNodes($index_name, $num_candidates, $queryVector)
    /// YIELD node, score
    /// WHERE {where_clause}
    /// RETURN score, ID(node) as element_id, node {.*, embedding:null } as node
    /// ```
    pub fn build_vector_search_query(
        &self,
        return_node: bool,
        req: &VectorSearchRequest<Neo4jSearchFilter>,
    ) -> Query {
        // Neo4j's `db.index.vector.queryNodes` takes a single vector, so use
        // the first query embedding.
        let prompt_embedding = req.query().first();
        let where_clause = match (req.threshold(), req.filter()) {
            (Some(thresh), Some(filt)) => Neo4jSearchFilter::gt("distance", thresh.into())
                .and(filt.clone())
                .render(),
            (Some(thresh), _) => Neo4jSearchFilter::gt("distance", thresh.into()).render(),
            (_, Some(filt)) => filt.clone().render(),
            _ => String::new(),
        };

        // Propertiy containing the embedding vectors are excluded from the returned node
        let query = format!(
            "\
            {}\
            \t{}\n\
            \tRETURN score, ID(node) as element_id {}
            ",
            BASE_VECTOR_SEARCH_QUERY,
            where_clause,
            if return_node {
                format!(
                    ", node {{.*, {}:null }} as node",
                    self.index_config.embedding_property
                )
            } else {
                "".to_string()
            }
        );

        tracing::debug!("Query before params: {}", query);

        Query::new(query)
            .param("queryVector", prompt_embedding.vec)
            .param("num_candidates", req.samples() as i64)
            .param("index_name", self.index_config.index_name.clone())
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

#[derive(Debug, Deserialize)]
pub struct RowResultNode<T> {
    score: f64,
    element_id: i64,
    node: T,
}

#[derive(Debug, Deserialize)]
struct RowResult {
    score: f64,
    element_id: i64,
}

impl Neo4jVectorIndex {
    /// Get the top n nodes and scores matching the pre-embedded query.
    ///
    /// Each [`SearchHit`] contains the similarity score, the node ID, and the
    /// node's properties as a JSON payload (with the embedding property nulled).
    pub async fn top_n(
        &self,
        req: VectorSearchRequest<Neo4jSearchFilter>,
    ) -> Result<Vec<SearchHit>, VectorStoreError> {
        let query = self.build_vector_search_query(true, &req);

        let rows = Neo4jClient::execute_and_collect::<RowResultNode<serde_json::Value>>(
            &self.graph,
            query,
        )
        .await?;

        Ok(rows
            .into_iter()
            .map(|row| SearchHit {
                id: row.element_id.to_string(),
                score: row.score,
                payload: row.node,
            })
            .collect())
    }

    /// Get the top n ids and scores matching the query. Runs faster than top_n since it doesn't need to transfer and parse
    /// the full nodes and embeddings to the client.
    pub async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Neo4jSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let query = self.build_vector_search_query(true, &req);

        let rows = Neo4jClient::execute_and_collect::<RowResult>(&self.graph, query).await?;

        let results = rows
            .into_iter()
            .map(|row| (row.score, row.element_id.to_string()))
            .collect::<Vec<_>>();

        Ok(results)
    }

    /// Get the top n nodes deserialized into `T` as `(score, id, document)`
    /// tuples. Sugar over [`Self::top_n`].
    pub async fn top_n_as<T: DeserializeOwned>(
        &self,
        req: VectorSearchRequest<Neo4jSearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        self.top_n(req)
            .await?
            .into_iter()
            .map(|hit| {
                let doc = serde_json::from_value(hit.payload)?;
                Ok((hit.score, hit.id, doc))
            })
            .collect()
    }

    /// Inserts one node per embedding, flattening the record's JSON payload
    /// fields onto the node alongside the record's `id`, the embedding
    /// (`embedding_property`), and its source text (`embedded_text`). Nodes are
    /// written under the index's `node_label`, defaulting to
    /// [`DEFAULT_NODE_LABEL`].
    pub async fn insert(&self, records: Vec<StoreRecord>) -> Result<(), VectorStoreError> {
        let node_label = self
            .index_config
            .node_label
            .as_deref()
            .unwrap_or(DEFAULT_NODE_LABEL);
        let embedding_property = &self.index_config.embedding_property;

        // Build one parameter map per embedding for a single UNWIND insert.
        let mut items: Vec<neo4rs::BoltType> = Vec::new();
        for record in records {
            for embedding in record.embeddings.iter() {
                let mut props = neo4rs::BoltMap::new();
                if let serde_json::Value::Object(map) = &record.payload {
                    for (key, value) in map {
                        props.put(neo4rs::BoltString::new(key), value.to_bolt_type());
                    }
                } else {
                    props.put(
                        neo4rs::BoltString::new("document"),
                        record.payload.to_bolt_type(),
                    );
                }
                props.put(
                    neo4rs::BoltString::new("id"),
                    neo4rs::BoltType::String(neo4rs::BoltString::new(&record.id)),
                );
                props.put(
                    neo4rs::BoltString::new("embedded_text"),
                    neo4rs::BoltType::String(neo4rs::BoltString::new(&embedding.document)),
                );
                props.put(
                    neo4rs::BoltString::new(embedding_property),
                    embedding.vec.to_bolt_type(),
                );
                items.push(neo4rs::BoltType::Map(props));
            }
        }

        self.graph
            .run(neo4rs::query(&insert_documents_query(node_label)).param("items", items))
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        Ok(())
    }

    /// Serializes each document and inserts it. Sugar over [`Self::insert`].
    pub async fn insert_as<T: Serialize>(
        &self,
        docs: Vec<(String, T, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let records = docs
            .into_iter()
            .map(|(id, doc, embeddings)| StoreRecord::new(id, &doc, embeddings))
            .collect::<Result<Vec<_>, _>>()?;
        self.insert(records).await
    }
}

/// The node label [`Neo4jVectorIndex::insert`] writes to when the index config
/// does not specify one (i.e. `node_label` is `None`).
const DEFAULT_NODE_LABEL: &str = "Document";

/// The Cypher used to bulk-insert nodes from an `$items` parameter list.
fn insert_documents_query(node_label: &str) -> String {
    format!("UNWIND $items AS item CREATE (n:{node_label}) SET n = item")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_label_defaults_to_none_and_builder_sets_it() {
        assert_eq!(IndexConfig::new("idx").node_label, None);
        assert_eq!(
            IndexConfig::new("idx")
                .node_label("Movie")
                .node_label
                .as_deref(),
            Some("Movie"),
        );
    }

    #[test]
    fn insert_documents_query_uses_label_else_default() {
        assert_eq!(
            insert_documents_query("Movie"),
            "UNWIND $items AS item CREATE (n:Movie) SET n = item",
        );
        assert_eq!(
            insert_documents_query(DEFAULT_NODE_LABEL),
            "UNWIND $items AS item CREATE (n:Document) SET n = item",
        );
    }
}

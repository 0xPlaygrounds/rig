//! Vector search over a Neo4j
//! [vector index](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/).

use neo4rs::{Graph, Query};
use rig_core::{
    Embed,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{SearchFilter, VectorSearchRequest},
    },
    wasm_compat::WasmCompatSend,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned, de::Error};

use crate::{Neo4jClient, Neo4jSearchFilter, ToBoltType};

/// Vector index over Neo4j nodes.
///
/// Queries are embedded with the same model `M` that populated the index, so
/// results are meaningless under another model.
pub struct Neo4jVectorIndex<M> {
    graph: Graph,
    embedding_model: M,
    index_config: IndexConfig,
}

/// Identifies the index to query and the shape of the nodes it covers.
///
/// Index names must be unique among indexes and constraints. Defaults are the
/// `vector_index` index, the `embedding` property, cosine similarity, and no
/// explicit node label.
#[derive(Serialize, Deserialize, Clone)]
pub struct IndexConfig {
    pub index_name: String,
    pub embedding_property: String,
    pub similarity_function: VectorSimilarityFunction,
    /// Node label that [`InsertDocuments`] writes to, adopted from the index when
    /// loaded through [`Neo4jClient::get_index`](crate::Neo4jClient::get_index).
    /// Inserts fall back to the `Document` label when unset.
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

    pub fn index_name(mut self, index_name: impl Into<String>) -> Self {
        self.index_name = index_name.into();
        self
    }

    pub fn similarity_function(mut self, similarity_function: VectorSimilarityFunction) -> Self {
        self.similarity_function = similarity_function;
        self
    }

    pub fn embedding_property(mut self, embedding_property: impl Into<String>) -> Self {
        self.embedding_property = embedding_property.into();
        self
    }

    /// Sets the node label that [`InsertDocuments`] writes to.
    pub fn node_label(mut self, node_label: impl Into<String>) -> Self {
        self.node_label = Some(node_label.into());
        self
    }
}

/// Similarity function an index is built with. See
/// [Neo4j vector similarity functions](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/#similarity-functions).
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

impl<M: EmbeddingModel> Neo4jVectorIndex<M> {
    pub fn new(graph: Graph, embedding_model: M, index_config: IndexConfig) -> Self {
        Self {
            graph,
            embedding_model,
            index_config,
        }
    }

    /// Builds the vector search query, returning node ids and scores plus, when
    /// `return_node` is set, the node with its embedding property nulled out.
    ///
    /// A request threshold is rendered as a predicate on the node's `distance`
    /// property rather than the yielded `score`. The filter text and embedding
    /// property are spliced into the Cypher verbatim.
    pub fn build_vector_search_query(
        &self,
        prompt_embedding: Embedding,
        return_node: bool,
        req: &VectorSearchRequest<Neo4jSearchFilter>,
    ) -> Query {
        let where_clause = match (req.threshold(), req.filter()) {
            (Some(thresh), Some(filt)) => Neo4jSearchFilter::gt("distance", thresh.into())
                .and(filt.clone())
                .render(),
            (Some(thresh), _) => Neo4jSearchFilter::gt("distance", thresh.into()).render(),
            (_, Some(filt)) => filt.clone().render(),
            _ => String::new(),
        };

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

    /// Embeds the query and runs the search, deserializing each row as `R`.
    /// Node data is always requested, even when `R` discards it.
    async fn run_search<R: for<'a> Deserialize<'a>>(
        &self,
        req: &VectorSearchRequest<Neo4jSearchFilter>,
    ) -> Result<Vec<R>, VectorStoreError> {
        let prompt_embedding = self.embedding_model.embed_text(req.query()).await?;
        let query = self.build_vector_search_query(prompt_embedding, true, req);

        Neo4jClient::execute_and_collect::<R>(&self.graph, query).await
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

impl<M: EmbeddingModel> VectorStoreIndex for Neo4jVectorIndex<M> {
    type Filter = Neo4jSearchFilter;

    /// Returns matches as `(score, node id, node)`. The node is deserialized as
    /// `T` without its embedding property.
    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<Neo4jSearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let rows = self.run_search::<RowResultNode<T>>(&req).await?;

        Ok(rows
            .into_iter()
            .map(|row| (row.score, row.element_id.to_string(), row.node))
            .collect())
    }

    /// Like `top_n` but returns `(score, node id)` without deserializing nodes.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Neo4jSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let rows = self.run_search::<RowResult>(&req).await?;

        Ok(rows
            .into_iter()
            .map(|row| (row.score, row.element_id.to_string()))
            .collect())
    }
}

/// Node label used by [`InsertDocuments`] when the config specifies none.
const DEFAULT_NODE_LABEL: &str = "Document";

/// Bulk insert statement over an `$items` parameter list. `node_label` is
/// spliced in verbatim.
fn insert_documents_query(node_label: &str) -> String {
    format!("UNWIND $items AS item CREATE (n:{node_label}) SET n = item")
}

impl<M: EmbeddingModel> InsertDocuments for Neo4jVectorIndex<M> {
    /// Inserts one node per embedding, flattening the document's JSON fields
    /// onto the node alongside the embedding (`embedding_property`) and its
    /// source text (`embedded_text`). Nodes are written under the index's
    /// `node_label`, defaulting to the `Document` label.
    async fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, Vec<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let node_label = self
            .index_config
            .node_label
            .as_deref()
            .unwrap_or(DEFAULT_NODE_LABEL);
        let embedding_property = &self.index_config.embedding_property;

        let mut items: Vec<neo4rs::BoltType> = Vec::new();
        for (document, embeddings) in documents {
            let json_doc = serde_json::to_value(&document)?;

            for embedding in embeddings {
                let mut props = neo4rs::BoltMap::new();
                if let serde_json::Value::Object(map) = &json_doc {
                    for (key, value) in map {
                        props.put(neo4rs::BoltString::new(key), value.to_bolt_type());
                    }
                } else {
                    props.put(neo4rs::BoltString::new("document"), json_doc.to_bolt_type());
                }
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
            .map_err(VectorStoreError::datastore)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests;

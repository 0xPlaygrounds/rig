//! A [`GraphStore`] over Neo4j.
//!
//! Entities are `Entity` nodes keyed by their `id` property, and relations are
//! `RELATES_TO` relationships carrying a `relation` label. Storing each node's
//! embedding in its `embedding` property lets a vector index over
//! `Entity.embedding` seed a [`GraphIndex`](rig_core::graph_store::GraphIndex).
//!
//! ```no_run
//! use rig_core::graph_store::{GraphIndex, GraphStore, KnowledgeGraph};
//! use rig_core::providers::openai::{self, OpenAI};
//! use rig_neo4j::{Neo4jClient, vector_index::IndexConfig};
//!
//! # async fn run(graph: KnowledgeGraph) -> Result<(), anyhow::Error> {
//! let model = OpenAI::from_env()?.embedding(openai::TEXT_EMBEDDING_3_SMALL, None);
//! let client = Neo4jClient::connect("neo4j://localhost:7687", "neo4j", "password").await?;
//! let store = client.graph_store().await?;
//! store.insert_graph(graph.clone()).await?;
//! let embedded = rig_core::embeddings::EmbeddingsBuilder::new(model.clone())
//!     .documents(graph.nodes)?
//!     .build()
//!     .await?;
//! store.insert_embedded_nodes(embedded).await?;
//! client
//!     .create_vector_index(IndexConfig::new("entities"), "Entity", model.capabilities().ndims)
//!     .await?;
//! let index = GraphIndex::new(client.get_index(model, "entities").await?, store);
//! # let _ = index;
//! # Ok(())
//! # }
//! ```

use neo4rs::{BoltType, Graph, Query};
use rig_core::{
    embeddings::Embedding,
    graph_store::{Edge, GraphStore, KnowledgeGraph, Node},
    vector_store::VectorStoreError,
};
use serde::{Deserialize, Serialize};

use crate::{Neo4jClient, ToBoltType};

/// Merges `$nodes` as `Entity` nodes: empty names and kinds are filled, and
/// descriptions gain each distinct new text.
const MERGE_NODES: &str = "
UNWIND $nodes AS node
MERGE (n:Entity {id: node.id})
SET n.name = CASE WHEN coalesce(n.name, '') = '' THEN node.name ELSE n.name END,
    n.kind = CASE WHEN coalesce(n.kind, '') = '' THEN node.kind ELSE n.kind END,
    n.description = CASE
        WHEN node.description = '' OR coalesce(n.description, '') CONTAINS node.description
            THEN coalesce(n.description, '')
        WHEN coalesce(n.description, '') = '' THEN node.description
        ELSE n.description + ' ' + node.description
    END";

/// Merges `$edges` as `RELATES_TO` relationships, creating bare endpoints.
const MERGE_EDGES: &str = "
UNWIND $edges AS edge
MERGE (a:Entity {id: edge.source})
ON CREATE SET a.name = '', a.kind = '', a.description = ''
MERGE (b:Entity {id: edge.target})
ON CREATE SET b.name = '', b.kind = '', b.description = ''
MERGE (a)-[r:RELATES_TO {relation: edge.relation}]->(b)
SET r.description = CASE
    WHEN edge.description = '' OR coalesce(r.description, '') CONTAINS edge.description
        THEN coalesce(r.description, '')
    WHEN coalesce(r.description, '') = '' THEN edge.description
    ELSE r.description + ' ' + edge.description
END";

const SEED: &str = "
MATCH (n:Entity {id: $id})
RETURN n.id AS id, coalesce(n.name, '') AS name, coalesce(n.kind, '') AS kind,
    coalesce(n.description, '') AS description";

/// The edges within `hops` of `$id`, nearest first. An edge's distance is the
/// shortest path from the seed that ends with it. Cypher cannot take the
/// path length as a parameter, so `hops` is spliced in.
fn edges_query(hops: usize) -> String {
    format!(
        "
MATCH p = (:Entity {{id: $id}})-[:RELATES_TO*1..{hops}]-(:Entity)
UNWIND relationships(p) AS r
WITH r, min(length(p)) AS distance
WITH r, distance, startNode(r) AS a, endNode(r) AS b
RETURN
    a {{id: a.id, name: coalesce(a.name, ''), kind: coalesce(a.kind, ''),
        description: coalesce(a.description, '')}} AS source,
    b {{id: b.id, name: coalesce(b.name, ''), kind: coalesce(b.kind, ''),
        description: coalesce(b.description, '')}} AS target,
    r.relation AS relation,
    coalesce(r.description, '') AS description,
    distance
ORDER BY distance, source.id, relation, target.id
LIMIT $limit"
    )
}

#[derive(Deserialize)]
struct EdgeRow {
    source: Node,
    target: Node,
    relation: String,
    description: String,
}

#[derive(Serialize)]
struct EmbeddedNode {
    #[serde(flatten)]
    node: Node,
    embedding: Option<Vec<f64>>,
}

/// A knowledge graph stored in Neo4j.
#[derive(Clone)]
pub struct Neo4jGraphStore {
    graph: Graph,
}

impl Neo4jGraphStore {
    /// A store over `graph`. Creates the uniqueness constraint on
    /// `Entity.id` if it is missing, so concurrent inserts merge rather
    /// than duplicate entities.
    pub async fn new(graph: Graph) -> Result<Self, VectorStoreError> {
        graph
            .run(neo4rs::query(
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            ))
            .await
            .map_err(VectorStoreError::datastore)?;
        Ok(Self { graph })
    }

    /// Merges each node as [`GraphStore::insert_graph`] does and stores its
    /// first embedding in the `embedding` property. Nodes without an
    /// embedding are merged unchanged.
    pub async fn insert_embedded_nodes(
        &self,
        nodes: Vec<(Node, Vec<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let nodes: Vec<BoltType> = nodes
            .into_iter()
            .map(|(node, embeddings)| {
                EmbeddedNode {
                    node,
                    embedding: embeddings.into_iter().next().map(|embedding| embedding.vec),
                }
                .to_bolt_type()
            })
            .collect();
        self.graph
            .run(
                neo4rs::query(&format!(
                    "{MERGE_NODES}, n.embedding = coalesce(node.embedding, n.embedding)"
                ))
                .param("nodes", nodes),
            )
            .await
            .map_err(VectorStoreError::datastore)
    }
}

impl Neo4jClient {
    /// A [`Neo4jGraphStore`] over this client's database.
    pub async fn graph_store(&self) -> Result<Neo4jGraphStore, VectorStoreError> {
        Neo4jGraphStore::new(self.graph.clone()).await
    }
}

impl GraphStore for Neo4jGraphStore {
    /// Merges nodes, then edges, in one transaction.
    async fn insert_graph(&self, graph: KnowledgeGraph) -> Result<(), VectorStoreError> {
        let mut txn = self
            .graph
            .start_txn()
            .await
            .map_err(VectorStoreError::datastore)?;
        txn.run_queries([
            neo4rs::query(MERGE_NODES).param("nodes", graph.nodes.to_bolt_type()),
            neo4rs::query(MERGE_EDGES).param("edges", graph.edges.to_bolt_type()),
        ])
        .await
        .map_err(VectorStoreError::datastore)?;
        txn.commit().await.map_err(VectorStoreError::datastore)
    }

    /// Edges at equal distance are ordered by source id, relation and target
    /// id. Path enumeration grows quickly with `hops` in dense graphs.
    async fn neighbourhood(
        &self,
        id: &str,
        hops: usize,
        limit: usize,
    ) -> Result<KnowledgeGraph, VectorStoreError> {
        let mut graph = KnowledgeGraph {
            nodes: Neo4jClient::execute_and_collect::<Node>(
                &self.graph,
                Query::new(SEED.to_string()).param("id", id),
            )
            .await?,
            edges: Vec::new(),
        };
        if graph.nodes.is_empty() || hops == 0 || limit == 0 {
            return Ok(graph);
        }
        let rows = Neo4jClient::execute_and_collect::<EdgeRow>(
            &self.graph,
            Query::new(edges_query(hops))
                .param("id", id)
                .param("limit", i64::try_from(limit).unwrap_or(i64::MAX)),
        )
        .await?;
        for row in rows {
            graph.edges.push(Edge {
                source: row.source.id.clone(),
                target: row.target.id.clone(),
                relation: row.relation,
                description: row.description,
            });
            for node in [row.source, row.target] {
                if !graph.nodes.iter().any(|known| known.id == node.id) {
                    graph.nodes.push(node);
                }
            }
        }
        Ok(graph)
    }
}

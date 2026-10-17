//! Knowledge-graph storage and graph retrieval.
//!
//! A [`GraphStore`] holds entities ([`Node`]) and the relations between them
//! ([`Edge`]). A [`GraphIndex`] retrieves from it: a vector index over the
//! entities finds the ones closest to the query, and the store expands each into
//! its neighbourhood. `GraphIndex` is a [`VectorStoreIndex`], so it serves as an
//! agent's dynamic context or search tool like any other index. [`ExtractedGraph`]
//! is the schema a language model fills when extracting a graph from text.
//!
//! ```
//! use rig_core::graph_store::{Edge, GraphStore, KnowledgeGraph, Node, in_memory::InMemoryGraphStore};
//!
//! # tokio_test::block_on(async {
//! let store = InMemoryGraphStore::default();
//! store.insert_graph(KnowledgeGraph {
//!     nodes: vec![Node::new("Ada Lovelace", "person", "Mathematician.")],
//!     edges: vec![Edge::new("Ada Lovelace", "worked with", "Charles Babbage", "")],
//! }).await?;
//! let around_ada = store.neighbourhood("ada lovelace", 1, 10).await?;
//! assert_eq!(around_ada.edges.len(), 1);
//! # Ok::<(), rig_core::vector_store::VectorStoreError>(())
//! # })?;
//! # Ok::<(), rig_core::vector_store::VectorStoreError>(())
//! ```

use std::collections::HashSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    Embed,
    embeddings::{EmbedError, TextEmbedder},
    vector_store::{
        VectorSearchIdResult, VectorSearchRequest, VectorSearchResult, VectorStoreError,
        VectorStoreIndex,
    },
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

pub mod in_memory;

/// An entity in a knowledge graph.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    /// Identity: stores merge nodes with equal ids.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Entity type, such as `person` or `organization`.
    pub kind: String,
    /// What the source text says about the entity.
    pub description: String,
}

impl Node {
    /// A node whose id is [`entity_id`] of `name`.
    pub fn new(
        name: impl Into<String>,
        kind: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        let name = name.into();
        Self {
            id: entity_id(&name),
            name,
            kind: kind.into(),
            description: description.into(),
        }
    }
}

/// Embeds `name (kind): description`, so a vector index over nodes can find
/// the entities a query is about.
impl Embed for Node {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(format!(
            "{} ({}): {}",
            self.name, self.kind, self.description
        ));
        Ok(())
    }
}

/// A directed relation between two nodes, by node id.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    /// Id of the node the relation starts at.
    pub source: String,
    /// Id of the node the relation points to.
    pub target: String,
    /// Relation label, such as `works at`.
    pub relation: String,
    /// What the source text says about the relation.
    pub description: String,
}

impl Edge {
    /// An edge between the nodes named `source` and `target`, with ids and
    /// relation label normalized by [`entity_id`].
    pub fn new(source: &str, relation: &str, target: &str, description: impl Into<String>) -> Self {
        Self {
            source: entity_id(source),
            target: entity_id(target),
            relation: entity_id(relation),
            description: description.into(),
        }
    }
}

/// Nodes and the edges between them.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// The entities.
    pub nodes: Vec<Node>,
    /// The relations.
    pub edges: Vec<Edge>,
}

/// The id of the entity called `name`: lowercase, with runs of whitespace
/// collapsed to one space, so differently written mentions of a name merge.
pub fn entity_id(name: &str) -> String {
    name.split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Merges `new` into a stored description, keeping each distinct text once.
pub fn merge_description(existing: &mut String, new: &str) {
    if new.is_empty() || existing.contains(new) {
        return;
    }
    if !existing.is_empty() {
        existing.push(' ');
    }
    existing.push_str(new);
}

/// Storage for a knowledge graph.
///
/// Errors use [`VectorStoreError`], so a store's failures surface unchanged
/// through [`GraphIndex`].
pub trait GraphStore: WasmCompatSend + WasmCompatSync {
    /// Adds `graph`. A node whose id is already stored is merged: its empty
    /// name and kind are filled, and its description gains the new text via
    /// [`merge_description`]. Edges merge the same way by `(source, relation, target)`. An
    /// edge whose endpoint is not stored creates a node with only an id.
    fn insert_graph(
        &self,
        graph: KnowledgeGraph,
    ) -> impl std::future::Future<Output = Result<(), VectorStoreError>> + WasmCompatSend;

    /// The node `id` and what lies within `hops` edges of it, following edges
    /// in both directions. Nodes start with `id`; edges come nearest first,
    /// at most `limit` of them, with each edge's endpoints among the nodes.
    /// An unknown `id` gives an empty graph.
    fn neighbourhood(
        &self,
        id: &str,
        hops: usize,
        limit: usize,
    ) -> impl std::future::Future<Output = Result<KnowledgeGraph, VectorStoreError>> + WasmCompatSend;
}

/// Graph retrieval: seeds from a vector index over nodes, expanded through a
/// [`GraphStore`].
///
/// Each seed's document in the index must carry the node's `id` field, as a
/// stored [`Node`] does. A search returns one result per seed, in the seed
/// index's order and with its score; the document is that seed's
/// [`KnowledgeGraph`] neighbourhood, without the nodes and edges an earlier
/// result already holds. Seeds with nothing new are left out.
pub struct GraphIndex<I, G> {
    seeds: I,
    graph: G,
    hops: usize,
    limit: usize,
}

impl<I, G> GraphIndex<I, G> {
    /// Searches `seeds` for entities and expands them through `graph`, two
    /// hops deep and at most 32 edges per seed.
    pub fn new(seeds: I, graph: G) -> Self {
        Self {
            seeds,
            graph,
            hops: 2,
            limit: 32,
        }
    }

    /// Expand each seed this many edges deep.
    pub fn hops(mut self, hops: usize) -> Self {
        self.hops = hops;
        self
    }

    /// Return at most this many edges per seed.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// The graph store.
    pub fn graph(&self) -> &G {
        &self.graph
    }
}

/// The part of a seed document [`GraphIndex`] reads.
#[derive(Deserialize)]
struct Seed {
    id: String,
}

impl<I, G> GraphIndex<I, G>
where
    I: VectorStoreIndex,
    G: GraphStore,
{
    async fn seeds(
        &self,
        req: VectorSearchRequest<I::Filter>,
    ) -> Result<Vec<VectorSearchIdResult>, VectorStoreError> {
        Ok(self
            .seeds
            .top_n::<Seed>(req)
            .await?
            .into_iter()
            .map(|result| VectorSearchIdResult {
                score: result.score,
                id: result.document.id,
            })
            .collect())
    }
}

impl<I, G> VectorStoreIndex for GraphIndex<I, G>
where
    I: VectorStoreIndex,
    G: GraphStore,
{
    type Filter = I::Filter;

    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<VectorSearchResult<T>>, VectorStoreError> {
        let mut seen_nodes = HashSet::new();
        let mut seen_edges = HashSet::new();
        let mut results = Vec::new();
        for seed in self.seeds(req).await? {
            let mut graph = self
                .graph
                .neighbourhood(&seed.id, self.hops, self.limit)
                .await?;
            graph
                .nodes
                .retain(|node| seen_nodes.insert(node.id.clone()));
            graph.edges.retain(|edge| {
                seen_edges.insert((
                    edge.source.clone(),
                    edge.relation.clone(),
                    edge.target.clone(),
                ))
            });
            if graph.nodes.is_empty() && graph.edges.is_empty() {
                continue;
            }
            results.push(VectorSearchResult {
                score: seed.score,
                id: seed.id,
                document: serde_json::from_value(serde_json::to_value(graph)?)?,
            });
        }
        Ok(results)
    }

    /// The seed node ids and scores, without expanding them.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<VectorSearchIdResult>, VectorStoreError> {
        self.seeds(req).await
    }
}

/// A graph extracted from text by a language model: the schema of an
/// extractor's output, converted into a [`KnowledgeGraph`] with
/// [`From`], which derives every id with [`entity_id`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ExtractedGraph {
    /// Every distinct entity the text names: people, organizations, places,
    /// products, events and concepts.
    pub entities: Vec<ExtractedEntity>,
    /// Every relation the text states between two of those entities.
    pub relations: Vec<ExtractedRelation>,
}

/// An entity in an [`ExtractedGraph`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ExtractedEntity {
    /// The entity's full name as written in the text.
    pub name: String,
    /// The entity's type, as one lowercase word such as `person` or `city`.
    pub kind: String,
    /// One sentence on what the text says about the entity.
    pub description: String,
}

/// A relation in an [`ExtractedGraph`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ExtractedRelation {
    /// Name of the entity the relation starts at, exactly as in `entities`.
    pub source: String,
    /// Short lowercase verb phrase, such as `founded` or `located in`.
    pub relation: String,
    /// Name of the entity the relation points to, exactly as in `entities`.
    pub target: String,
    /// One sentence on what the text says about the relation.
    pub description: String,
}

impl From<ExtractedGraph> for KnowledgeGraph {
    fn from(extracted: ExtractedGraph) -> Self {
        Self {
            nodes: extracted
                .entities
                .into_iter()
                .map(|entity| Node::new(entity.name, entity.kind, entity.description))
                .collect(),
            edges: extracted
                .relations
                .into_iter()
                .map(|relation| {
                    Edge::new(
                        &relation.source,
                        &relation.relation,
                        &relation.target,
                        relation.description,
                    )
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests;

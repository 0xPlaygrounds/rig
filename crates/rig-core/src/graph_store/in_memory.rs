//! A [`GraphStore`] held in memory.
//!
//! ```
//! use rig_core::graph_store::in_memory::InMemoryGraphStore;
//!
//! let store = InMemoryGraphStore::default();
//! # let _ = store;
//! ```

use std::{
    collections::{HashMap, HashSet},
    sync::{PoisonError, RwLock},
};

use super::{Edge, GraphStore, KnowledgeGraph, Node, merge_description};
use crate::vector_store::VectorStoreError;

/// A knowledge graph in memory, traversed breadth first. Neighbourhoods list
/// edges in insertion order within each hop.
#[derive(Debug, Default)]
pub struct InMemoryGraphStore {
    state: RwLock<State>,
}

#[derive(Debug, Default)]
struct State {
    nodes: HashMap<String, Node>,
    edges: Vec<Edge>,
    edge_ids: HashMap<(String, String, String), usize>,
    /// Indices into `edges` touching each node id.
    adjacency: HashMap<String, Vec<usize>>,
}

impl State {
    fn insert_node(&mut self, node: Node) {
        let Some(existing) = self.nodes.get_mut(&node.id) else {
            self.nodes.insert(node.id.clone(), node);
            return;
        };
        if existing.name.is_empty() {
            existing.name = node.name;
        }
        if existing.kind.is_empty() {
            existing.kind = node.kind;
        }
        merge_description(&mut existing.description, &node.description);
    }

    fn insert_edge(&mut self, edge: Edge) {
        let key = (
            edge.source.clone(),
            edge.relation.clone(),
            edge.target.clone(),
        );
        if let Some(existing) = self
            .edge_ids
            .get(&key)
            .and_then(|&index| self.edges.get_mut(index))
        {
            merge_description(&mut existing.description, &edge.description);
            return;
        }
        for id in [&edge.source, &edge.target] {
            if !self.nodes.contains_key(id) {
                self.insert_node(Node {
                    id: id.clone(),
                    ..Node::default()
                });
            }
        }
        let index = self.edges.len();
        self.adjacency
            .entry(edge.source.clone())
            .or_default()
            .push(index);
        if edge.target != edge.source {
            self.adjacency
                .entry(edge.target.clone())
                .or_default()
                .push(index);
        }
        self.edge_ids.insert(key, index);
        self.edges.push(edge);
    }

    fn neighbourhood(&self, id: &str, hops: usize, limit: usize) -> KnowledgeGraph {
        let Some(seed) = self.nodes.get(id) else {
            return KnowledgeGraph::default();
        };
        let mut graph = KnowledgeGraph {
            nodes: vec![seed.clone()],
            edges: Vec::new(),
        };
        let mut seen_nodes = HashSet::from([id.to_owned()]);
        let mut seen_edges = HashSet::new();
        let mut frontier = vec![id.to_owned()];
        for _ in 0..hops {
            let mut next = Vec::new();
            for node_id in &frontier {
                for &index in self.adjacency.get(node_id).into_iter().flatten() {
                    if graph.edges.len() >= limit {
                        return graph;
                    }
                    let Some(edge) = self.edges.get(index) else {
                        continue;
                    };
                    if !seen_edges.insert(index) {
                        continue;
                    }
                    let other = if edge.source == *node_id {
                        &edge.target
                    } else {
                        &edge.source
                    };
                    if seen_nodes.insert(other.clone()) {
                        graph.nodes.extend(self.nodes.get(other).cloned());
                        next.push(other.clone());
                    }
                    graph.edges.push(edge.clone());
                }
            }
            frontier = next;
        }
        graph
    }
}

impl GraphStore for InMemoryGraphStore {
    async fn insert_graph(&self, graph: KnowledgeGraph) -> Result<(), VectorStoreError> {
        let mut state = self.state.write().unwrap_or_else(PoisonError::into_inner);
        for node in graph.nodes {
            state.insert_node(node);
        }
        for edge in graph.edges {
            state.insert_edge(edge);
        }
        Ok(())
    }

    async fn neighbourhood(
        &self,
        id: &str,
        hops: usize,
        limit: usize,
    ) -> Result<KnowledgeGraph, VectorStoreError> {
        Ok(self
            .state
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .neighbourhood(id, hops, limit))
    }
}

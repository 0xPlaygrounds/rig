use serde_json::Value;

use super::{in_memory::InMemoryGraphStore, *};
use crate::vector_store::request::Filter;

/// A seed index answering every search with fixed nodes and scores.
struct Seeds(Vec<(f64, Node)>);

impl VectorStoreIndex for Seeds {
    type Filter = Filter<Value>;

    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<VectorSearchResult<T>>, VectorStoreError> {
        self.0
            .iter()
            .take(req.samples() as usize)
            .enumerate()
            .map(|(index, (score, node))| {
                Ok(VectorSearchResult {
                    score: *score,
                    // Backend ids need not be node ids; the document's `id` is.
                    id: format!("row{index}"),
                    document: serde_json::from_value(serde_json::to_value(node)?)?,
                })
            })
            .collect()
    }

    async fn top_n_ids(
        &self,
        _req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<VectorSearchIdResult>, VectorStoreError> {
        Ok(Vec::new())
    }
}

/// Ada worked with Babbage, who designed the Analytical Engine, which was
/// built in London; Grace is unconnected.
fn sample() -> KnowledgeGraph {
    KnowledgeGraph::from(ExtractedGraph {
        entities: vec![
            ExtractedEntity {
                name: "Ada  Lovelace".into(),
                kind: "person".into(),
                description: "Mathematician.".into(),
            },
            ExtractedEntity {
                name: "Charles Babbage".into(),
                kind: "person".into(),
                description: "Inventor.".into(),
            },
            ExtractedEntity {
                name: "Grace Hopper".into(),
                kind: "person".into(),
                description: "Admiral.".into(),
            },
        ],
        relations: vec![
            ExtractedRelation {
                source: "Ada Lovelace".into(),
                relation: "Worked with".into(),
                target: "charles babbage".into(),
                description: "Collaborators.".into(),
            },
            ExtractedRelation {
                source: "Charles Babbage".into(),
                relation: "designed".into(),
                target: "Analytical Engine".into(),
                description: String::new(),
            },
            ExtractedRelation {
                source: "Analytical Engine".into(),
                relation: "planned in".into(),
                target: "London".into(),
                description: String::new(),
            },
        ],
    })
}

fn ids(graph: &KnowledgeGraph) -> Vec<&str> {
    graph.nodes.iter().map(|node| node.id.as_str()).collect()
}

#[tokio::test]
async fn in_memory_store_merges_and_expands_by_hop() {
    let store = InMemoryGraphStore::default();
    store.insert_graph(sample()).await.expect("insert");
    store
        .insert_graph(KnowledgeGraph {
            nodes: vec![
                Node::new("Ada Lovelace", "", "Wrote the first program."),
                Node::new("Analytical Engine", "machine", "A mechanical computer."),
            ],
            edges: vec![Edge::new(
                "Ada Lovelace",
                "worked with",
                "Charles Babbage",
                "Collaborators.",
            )],
        })
        .await
        .expect("insert");

    let one_hop = store
        .neighbourhood("charles babbage", 1, 10)
        .await
        .expect("neighbourhood");
    let two_hops = store
        .neighbourhood("ada lovelace", 2, 10)
        .await
        .expect("neighbourhood");
    let limited = store
        .neighbourhood("ada lovelace", 3, 2)
        .await
        .expect("neighbourhood");
    let unknown = store
        .neighbourhood("nobody", 2, 10)
        .await
        .expect("neighbourhood");

    assert_eq!(
        ids(&one_hop),
        ["charles babbage", "ada lovelace", "analytical engine"]
    );
    assert_eq!(one_hop.edges.len(), 2);
    let ada = one_hop.nodes.get(1).cloned().expect("node");
    assert_eq!(ada.name, "Ada  Lovelace");
    assert_eq!(ada.kind, "person");
    assert_eq!(ada.description, "Mathematician. Wrote the first program.");
    let engine = one_hop.nodes.get(2).cloned().expect("node");
    assert_eq!(engine.kind, "machine");

    assert_eq!(
        ids(&two_hops),
        ["ada lovelace", "charles babbage", "analytical engine"]
    );
    assert_eq!(
        two_hops
            .edges
            .first()
            .map(|edge| (edge.relation.as_str(), edge.description.as_str())),
        Some(("worked with", "Collaborators."))
    );
    assert_eq!(limited.edges.len(), 2);
    assert_eq!(unknown, KnowledgeGraph::default());
}

#[tokio::test]
async fn graph_index_returns_each_seeds_new_neighbourhood() {
    let store = InMemoryGraphStore::default();
    store.insert_graph(sample()).await.expect("insert");
    let nodes = sample().nodes;
    let seeds = Seeds(vec![
        (0.9, nodes.get(1).cloned().expect("node")),
        (0.8, nodes.first().cloned().expect("node")),
        (0.7, nodes.get(2).cloned().expect("node")),
    ]);
    let index = GraphIndex::new(seeds, store).hops(1);
    let req = VectorSearchRequest::builder()
        .query("Who did Ada work with?")
        .samples(3)
        .build();

    let results = index
        .top_n::<KnowledgeGraph>(req.clone())
        .await
        .expect("top_n");
    let ids_only = index.top_n_ids(req).await.expect("top_n_ids");

    // Ada's only edge is already in Babbage's neighbourhood, so Ada is left out.
    let summary: Vec<_> = results
        .iter()
        .map(|result| {
            (
                result.id.as_str(),
                result.score,
                ids(&result.document),
                result.document.edges.len(),
            )
        })
        .collect();
    assert_eq!(
        summary,
        [
            (
                "charles babbage",
                0.9,
                vec!["charles babbage", "ada lovelace", "analytical engine"],
                2
            ),
            ("grace hopper", 0.7, vec!["grace hopper"], 0),
        ]
    );
    assert_eq!(
        ids_only
            .iter()
            .map(|result| result.id.as_str())
            .collect::<Vec<_>>(),
        ["charles babbage", "ada lovelace", "grace hopper"]
    );
}

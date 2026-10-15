//! Graph retrieval: extracts a knowledge graph from a few paragraphs with an LLM,
//! then answers a multi-hop question with the graph as dynamic context.
//! Requires `OPENAI_API_KEY`.
//!
//! No paragraph answers the question alone: the agent has to follow the
//! Lumen cell to its inventor, her employer, and the employer's headquarters.

use rig::embeddings::EmbeddingsBuilder;
use rig::extractor::ExtractorBuilder;
use rig::graph_store::{GraphIndex, GraphStore, KnowledgeGraph, in_memory::InMemoryGraphStore};
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};
use rig::vector_store::{
    VectorSearchRequest, VectorStoreIndex, in_memory_store::InMemoryVectorStore,
};

const PARAGRAPHS: [&str; 4] = [
    "The Lumen cell is a solid-state battery that charges in ninety seconds. \
     It was invented by the materials scientist Mira Okonkwo.",
    "Mira Okonkwo leads the energy storage lab at Halvard Dynamics, \
     which she joined after finishing her doctorate in Lagos.",
    "Halvard Dynamics was founded in 2011 by Petter Halvard. \
     The company is headquartered in Tromso, Norway.",
    "Jonas Brekke invented the Corvid drone. He works at Skarv Robotics, \
     a company based in Bergen.",
];

const QUESTION: &str =
    "In which city is the company that employs the inventor of the Lumen cell headquartered?";

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let openai = OpenAI::from_env()?;
    let completion = openai.completion(openai::GPT_5_4_MINI);
    let embedding = openai
        .embedding(openai::TEXT_EMBEDDING_3_SMALL, None)
        .erase();

    // Extract entities and relations from each paragraph into one graph.
    let extractor = ExtractorBuilder::knowledge_graph(completion.clone())
        .retries(1)
        .build();
    let store = InMemoryGraphStore::default();
    let mut nodes = Vec::new();
    for paragraph in PARAGRAPHS {
        let graph = KnowledgeGraph::from(extractor.extract(paragraph).await?.output);
        println!(
            "Extracted {} entities and {} relations",
            graph.nodes.len(),
            graph.edges.len()
        );
        nodes.extend(graph.nodes.clone());
        store.insert_graph(graph).await?;
    }

    // Seed retrieval with a vector index over the entities, keyed by node id.
    let embedded = EmbeddingsBuilder::new(embedding.clone())
        .documents(nodes)?
        .build()
        .await?;
    let seeds = InMemoryVectorStore::from_documents_with_id_f(embedded, |node| node.id.clone())
        .index(embedding);
    let index = GraphIndex::new(seeds, store).hops(3);

    let request = VectorSearchRequest::builder()
        .query(QUESTION)
        .samples(2)
        .build();
    for result in index.top_n::<KnowledgeGraph>(request).await? {
        println!("\nContext seeded at `{}`:", result.id);
        for edge in result.document.edges {
            println!("  {} --{}--> {}", edge.source, edge.relation, edge.target);
        }
    }

    let agent = AgentBuilder::new(completion)
        .preamble(
            "Answer from the knowledge-graph context only. \
             Name the chain of facts you followed.",
        )
        .dynamic_context(2, index)
        .build();
    let answer = agent.prompt(QUESTION).await?.output();
    println!("\nQ: {QUESTION}\nA: {answer}");
    Ok(())
}

//! This example shows how to perform a vector search on a Neo4j database.
//! It is based on the [Neo4j Embeddings & Vector Index Tutorial](https://neo4j.com/docs/genai/tutorials/embeddings-vector-indexes/).
//! The tutorial uses the `recommendations` dataset and the `moviePlots` index, which is created in the tutorial.
//! They both need to be configured and the database running before running this example.

use std::env;

use rig::{
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::VectorStoreIndex,
};

use neo4rs::*;
use rig_neo4j::{Neo4jClient, Neo4jVectorStore, SearchParams};
use serde::{Deserialize, Serialize};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let neo4j_uri = env::var("NEO4J_URI").expect("NEO4J_URI not set");
    let neo4j_username = env::var("NEO4J_USERNAME").expect("NEO4J_USERNAME not set");
    let neo4j_password = env::var("NEO4J_PASSWORD").expect("NEO4J_PASSWORD not set");

    let neo4j_client = Neo4jClient::connect(&neo4j_uri, &neo4j_username, &neo4j_password).await?;
    let vector_store = Neo4jVectorStore::new(neo4j_client, "neo4j");

    // // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Define the properties that will be retrieved from querying the graph nodes
    #[derive(Debug, Deserialize, Serialize)]
    struct Movie {
        title: String,
        plot: String,
    }

    // Create a vector index on our vector store
    // ❗IMPORTANT: Reuse the same model that was used to generate the embeddings
    let index = vector_store.index(
        model,
        "moviePlots",
        SearchParams::new(Some("node.year > 1990".to_string())),
    );

    // Query the index
    let results = index
        .top_n::<Movie>("a historical movie on quebec", 5)
        .await?
        .into_iter()
        .map(|(score, id, doc)| (doc.title, id, doc.plot, score))
        .collect::<Vec<_>>();

    print_results(results);

    let id_results = index
        .top_n_ids("What is a linglingdong?", 1)
        .await?
        .into_iter()
        .map(|(score, id)| (score, id))
        .collect::<Vec<_>>();

    println!("ID results: {:?}", id_results);

    Ok(())
}

fn print_results(results: Vec<(String, String, String, f64)>) {
    println!("Results: {:#?}", results);

    println!("{:<40} {:<10} {:<100}", "Title", "ID", "Plot");
    println!("{}", "-".repeat(150));
    for (title, id, plot, _) in results {
        let wrapped_title = textwrap::fill(&title, 40);
        let wrapped_plot = textwrap::fill(&plot, 100);
        let title_lines: Vec<&str> = wrapped_title.lines().collect();
        let plot_lines: Vec<&str> = wrapped_plot.lines().collect();
        let max_lines = title_lines.len().max(plot_lines.len());

        for i in 0..max_lines {
            let title_line = title_lines.get(i).unwrap_or(&"");
            let plot_line = plot_lines.get(i).unwrap_or(&"");
            if i == 0 {
                println!("{:<40} {:<10} {:<100}", title_line, id, plot_line);
            } else {
                println!("{:<40} {:<10} {:<100}", title_line, "", plot_line);
            }
        }
        println!();
    }
    println!();
}

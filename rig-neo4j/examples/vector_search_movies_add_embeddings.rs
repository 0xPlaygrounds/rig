//! This example shows how to perform a vector search on a Neo4j database.
//! It is based on the [Neo4j Embeddings & Vector Index Tutorial](https://neo4j.com/docs/genai/tutorials/embeddings-vector-indexes/).
//! The tutorial uses the `recommendations` dataset and the `moviePlots` index, which is created in the tutorial.
//! See the Neo4j tutorial for more information on how to import the dataset.
//!
//! ❗IMPORTANT: The `recommendations` database has 28k nodes, so this example will take a while to run.

use std::env;

use rig::{
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::{
        VectorStoreIndex,
        request::{SearchFilter, VectorSearchRequest},
    },
};

use neo4rs::*;
use rig::client::EmbeddingsClient;
use rig_neo4j::{Neo4jClient, ToBoltType, vector_index::IndexConfig};
use serde::{Deserialize, Serialize};

#[path = "./display/lib.rs"]
mod display;

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Movie {
    title: String,
    plot: String,
    to_encode: Option<String>,
}

const NODE_LABEL: &str = "Movie";
const INDEX_NAME: &str = "moviePlots";

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let neo4j_uri = env::var("NEO4J_URI").expect("NEO4J_URI not set");
    let neo4j_username = env::var("NEO4J_USERNAME").expect("NEO4J_USERNAME not set");
    let neo4j_password = env::var("NEO4J_PASSWORD").expect("NEO4J_PASSWORD not set");

    let neo4j_client = Neo4jClient::connect(&neo4j_uri, &neo4j_username, &neo4j_password).await?;

    // Add embeddings to the Neo4j database
    let batch_size = 1000;
    let mut batch_n = 1;
    let mut movies_batch = Vec::<Movie>::new();

    let mut result = neo4j_client
        .graph
        .execute(Query::new(
            format!("MATCH (m:{NODE_LABEL}) RETURN m.plot AS plot, m.title AS title").to_string(),
        ))
        .await?;

    while let Some(row) = result.next().await? {
        let title: Option<BoltString> = row.get("title")?;
        let plot: Option<BoltString> = row.get("plot")?;

        if let (Some(title), Some(plot)) = (title, plot) {
            movies_batch.push(Movie {
                title: title.to_string(),
                plot: plot.to_string(),
                to_encode: Some(format!("Title: {title}\nPlot: {plot}")),
            });
        }

        // Import a batch; flush buffer
        if movies_batch.len() == batch_size {
            import_batch(&neo4j_client.graph, &movies_batch, batch_n).await?;
            movies_batch.clear();
            batch_n += 1;
        }
    }

    // Import any remaining movies
    if !movies_batch.is_empty() {
        import_batch(&neo4j_client.graph, &movies_batch, batch_n).await?;
    }

    // Show counters
    let mut result = neo4j_client
        .graph
        .execute(Query::new(
            format!("MATCH (m:{NODE_LABEL} WHERE m.embedding IS NOT NULL) RETURN count(*) AS countMoviesWithEmbeddings, size(m.embedding) AS embeddingSize").to_string(),
        ))
        .await?;

    if let Some(row) = result.next().await? {
        let count: i64 = row.get("countMoviesWithEmbeddings")?;
        let size: i64 = row.get("embeddingSize")?;
        println!(
            "Embeddings generated and attached to nodes.\n\
             Movie nodes with embeddings: {count}.\n\
             Embedding size: {size}."
        );
    }

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Since we are starting from scratch, we need to create the DB vector index
    neo4j_client
        .create_vector_index(IndexConfig::new(INDEX_NAME), NODE_LABEL, &model)
        .await?;

    // ❗IMPORTANT: Reuse the same model that was used to generate the embeddings
    let index = neo4j_client.get_index(model, INDEX_NAME).await?;

    let query = "a historical movie on quebec";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(5)
        .filter(SearchFilter::gt("node.year".into(), 1990.into()))
        .build()?;

    // Query the index
    let results = index
        .top_n::<Movie>(req)
        .await?
        .into_iter()
        .map(|(score, id, doc)| display::SearchResult {
            title: doc.title,
            id,
            description: doc.plot,
            score,
        })
        .collect::<Vec<_>>();

    println!("{:#}", display::SearchResults(&results));

    let query = "What is a linglingdong?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(1)
        .build()?;

    let id_results = index.top_n_ids(req).await?.into_iter().collect::<Vec<_>>();

    println!("ID results: {id_results:?}");

    Ok(())
}

async fn import_batch(graph: &Graph, nodes: &[Movie], batch_n: i32) -> Result<(), anyhow::Error> {
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let to_encode_list: Vec<String> = nodes
        .iter()
        .map(|node| node.to_encode.clone().unwrap())
        .collect();

    graph.run(
        Query::new(format!(
            "CALL genai.vector.encodeBatch($to_encode_list, 'OpenAI', {{ token: $token }}) YIELD index, vector
             MATCH (m:{NODE_LABEL} {{title: $movies[index].title, plot: $movies[index].plot}})
             CALL db.create.setNodeVectorProperty(m, 'embedding', vector)").to_string()
        )
        .param("movies", nodes.to_bolt_type())
        .param("to_encode_list", to_encode_list)
        .param("token", openai_api_key)
    ).await?;

    println!("Processed batch {batch_n}");
    Ok(())
}

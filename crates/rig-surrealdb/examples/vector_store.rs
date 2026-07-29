use rig_core::client::{EmbeddingsClient, ProviderClient};
use rig_core::providers::openai;
use rig_core::vector_store::request::VectorSearchRequest;
use rig_core::{
    Embed,
    embeddings::{EmbeddingModel, EmbeddingsBuilder},
};
use rig_surrealdb::{Mem, SurrealVectorStore};
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;

// A vector search is performed on the `description` field, so we derive `Embed`
// and mark that field with `#[embed]`.
#[derive(Embed, Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Default)]
struct TopicDefinition {
    topic: String,
    #[serde(skip)] // used for embeddings but not persisted in the example document payload
    #[embed]
    description: String,
}

impl std::fmt::Display for TopicDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.topic)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_client = openai::Client::from_env()?;
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    let surreal = Surreal::new::<Mem>(()).await?;

    surreal.use_ns("example").use_db("example").await?;

    let topics = vec![
        TopicDefinition {
            topic: "pasta carbonara".to_string(),
            description: "A traditional Roman pasta dish made with eggs, pecorino romano, black pepper, and guanciale.".to_string(),
        },
        TopicDefinition {
            topic: "green tea".to_string(),
            description: "A drink made by steeping unoxidized tea leaves in hot water for a light, grassy flavor.".to_string(),
        },
        TopicDefinition {
            topic: "solar eclipse".to_string(),
            description: "An event where the moon passes between Earth and the sun, temporarily blocking the sun's light.".to_string(),
        },
    ];

    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(topics)?
        .build()
        .await?;

    // The store receives precomputed vectors; embedding happens outside it.
    let vector_store = SurrealVectorStore::with_defaults(surreal);

    vector_store
        .insert_as(
            documents
                .into_iter()
                .enumerate()
                .map(|(i, (doc, embeddings))| (format!("topic{i}"), doc, embeddings))
                .collect(),
        )
        .await?;

    let query = "Which dish is a Roman pasta recipe made with eggs, pecorino romano, black pepper, and guanciale?";
    println!("Attempting vector search with query: {query}");

    let query_embedding = model.embed_text(query).await?;
    let req = VectorSearchRequest::builder()
        .query(query_embedding.clone())
        .samples(3)
        .build();

    let results = vector_store.top_n_as::<TopicDefinition>(req).await?;

    anyhow::ensure!(
        results.len() == 3,
        "expected three unfiltered results, got {}",
        results.len()
    );
    let Some(first_result) = results.first() else {
        return Err(anyhow::anyhow!("expected at least one result"));
    };
    anyhow::ensure!(
        first_result.2.topic == "pasta carbonara",
        "expected first result to be pasta carbonara, got {}",
        first_result.2.topic
    );

    println!("{} results for query: {}", results.len(), query);
    for (distance, _id, doc) in results.iter() {
        println!("Result distance {distance} for topic: {doc}");
    }

    let Some(second_result) = results.get(1) else {
        return Err(anyhow::anyhow!("expected at least two results"));
    };
    let midpoint = (first_result.0 + second_result.0) / 2.0;

    println!(
        "Attempting vector search with cosine similarity threshold of {midpoint} and query: {query}"
    );
    let req = VectorSearchRequest::builder()
        .query(query_embedding)
        .samples(1)
        .threshold(midpoint)
        .build();

    let results = vector_store.top_n_as::<TopicDefinition>(req).await?;

    println!("{} results for query: {}", results.len(), query);
    anyhow::ensure!(
        results.len() == 1,
        "expected one filtered result, got {}",
        results.len()
    );
    let Some(filtered_result) = results.first() else {
        return Err(anyhow::anyhow!("expected one filtered result"));
    };
    anyhow::ensure!(
        filtered_result.2.topic == "pasta carbonara",
        "expected filtered result to be pasta carbonara, got {}",
        filtered_result.2.topic
    );

    for (distance, _id, doc) in results.iter() {
        println!("Result distance {distance} for topic: {doc}");
    }

    Ok(())
}

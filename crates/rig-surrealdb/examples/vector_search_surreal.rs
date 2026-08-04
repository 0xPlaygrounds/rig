use rig_core::Embed;
use rig_core::OneOrMany;
use rig_core::embeddings::EmbeddingJob;
use rig_core::http_runtime::HttpRuntime;
use rig_core::providers::openai;
use rig_core::vector_store::request::VectorSearchRequest;
use rig_surrealdb::{Mem, SurrealVectorStore};
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;

// A vector search needs to be performed on the `definitions` field, so we derive the `Embed` trait for `WordDefinition`
// and tag that field with `#[embed]`.
// We are not going to store the definitions on our database so we skip the `Serialize` trait
#[derive(Embed, Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Default)]
struct WordDefinition {
    word: String,
    #[serde(skip)] // we don't want to serialize this field, we use only to create embeddings
    #[embed]
    definition: String,
}

impl std::fmt::Display for WordDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.word)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Embedding configuration is plain data plus a shared HTTP runtime.
    let embed_cfg = openai::functions::EmbeddingConfig::from_env(openai::TEXT_EMBEDDING_ADA_002)?;
    let rt = HttpRuntime::new();

    let surreal = Surreal::new::<Mem>(()).await?;

    surreal.use_ns("example").use_db("example").await?;

    // create test documents with mocked embeddings
    let words = vec![
        WordDefinition {
            word: "flurbo".to_string(),
            definition: "1. *flurbo* (name): A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
        },
        WordDefinition {
            word: "glarb-glarb".to_string(),
            definition: "1. *glarb-glarb* (noun): A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
        },
        WordDefinition {
            word: "linglingdong".to_string(),
            definition: "1. *linglingdong* (noun): A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        }];

    let documents = EmbeddingJob::new()
        .documents(words)
        .for_provider(&openai::functions::DESCRIPTOR)
        .run(|texts| openai::functions::embed(&embed_cfg, &rt, texts))
        .await?;

    // init vector store; embedding happens *outside* the store
    let vector_store = SurrealVectorStore::with_defaults(surreal);

    vector_store
        .insert_as(
            documents
                .into_iter()
                .enumerate()
                .map(|(i, (doc, embeddings))| (format!("doc{i}"), doc, embeddings))
                .collect(),
        )
        .await?;

    // query vector: embed the query, then send the pre-embedded request
    let query = "What does \"glarb-glarb\" mean?";
    println!("Attempting vector search with query: {query}");

    let query_embedding = openai::functions::embed(&embed_cfg, &rt, vec![query.to_string()])
        .await?
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no embedding returned for the query"))?;
    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding.clone()), 2);

    let results = vector_store.top_n_as::<WordDefinition>(req).await?;

    println!("{} results for query: {}", results.len(), query);
    for (distance, _id, doc) in results.iter() {
        println!("Result distance {distance} for word: {doc}");
    }

    // Use the midpoint as similarity threshold to guarantee exactly one result is returned.
    let Some(first_result) = results.first() else {
        return Err(anyhow::anyhow!("expected at least one result"));
    };
    let Some(second_result) = results.get(1) else {
        return Err(anyhow::anyhow!("expected at least two results"));
    };
    let midpoint = (first_result.0 + second_result.0) / 2.0;

    println!(
        "Attempting vector search with cosine similarity threshold of {midpoint} and query: {query}"
    );
    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 1).with_threshold(midpoint);

    let results = vector_store.top_n_as::<WordDefinition>(req).await?;

    println!("{} results for query: {}", results.len(), query);
    anyhow::ensure!(
        results.len() == 1,
        "expected one result after threshold filtering, got {}",
        results.len()
    );

    for (distance, _id, doc) in results.iter() {
        println!("Result distance {distance} for word: {doc}");
    }

    Ok(())
}

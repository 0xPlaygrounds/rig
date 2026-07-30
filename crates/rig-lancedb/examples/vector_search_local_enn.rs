use fixture::{as_record_batch, words};
use rig_core::OneOrMany;
use rig_core::embeddings::{default_concurrency, embed_documents};
use rig_core::http_runtime::HttpRuntime;
use rig_core::providers::openai;
use rig_core::vector_store::request::VectorSearchRequest;
use rig_lancedb::{LanceDbVectorIndex, SearchParams};

#[path = "./fixtures/lib.rs"]
mod fixture;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Embeddings come from a free function over plain configuration plus a
    // shared HTTP runtime — there is no client or model object.
    let embed_cfg = openai::functions::EmbeddingConfig::from_env(openai::TEXT_EMBEDDING_ADA_002)?;
    // The config knows its model's native width; no need to restate it.
    let embedding_dims = embed_cfg
        .ndims()
        .ok_or_else(|| anyhow::anyhow!("text-embedding-ada-002 has a known vector width"))?;
    let rt = HttpRuntime::new();
    let max_documents = openai::functions::DESCRIPTOR
        .max_embedding_documents
        .unwrap_or(usize::MAX);

    // Generate embeddings for the test data.
    let embeddings = embed_documents(
        words(),
        max_documents,
        default_concurrency(max_documents),
        |texts| openai::functions::embed(&embed_cfg, &rt, texts),
    )
    .await?;

    // Define search_params params that will be used by the vector store to perform the vector search.
    let search_params = SearchParams::default();

    // Initialize LanceDB locally.
    let db = lancedb::connect("data/lancedb-store").execute().await?;

    let table = if db
        .table_names()
        .execute()
        .await?
        .contains(&"definitions".to_string())
    {
        db.open_table("definitions").execute().await?
    } else {
        db.create_table(
            "definitions",
            vec![as_record_batch(embeddings, embedding_dims)?],
        )
        .execute()
        .await?
    };

    let vector_store = LanceDbVectorIndex::new(table, "id", search_params).await?;

    // Queries are pre-embedded: embed the query text with the embedding model
    // and pass the embedding to the search request.
    let query = "My boss says I zindle too much, what does that mean?";
    let query_embedding = openai::functions::embed(&embed_cfg, &rt, vec![query.to_string()])
        .await?
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no embedding returned for the query"))?;
    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 1);

    // Query the index
    let results = vector_store.top_n_ids(req).await?;

    println!("Results: {results:?}");

    Ok(())
}

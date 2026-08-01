use fixture::{Word, as_record_batch, words};
use lancedb::index::vector::IvfPqIndexBuilder;
use rig_agent::prelude::*;
use rig_core::OneOrMany;
use rig_core::embeddings::EmbeddingJob;
use rig_core::providers::openai;
use rig_core::vector_store::request::VectorSearchRequest;
use rig_lancedb::{LanceDbVectorIndex, SearchParams};

#[path = "./fixtures/lib.rs"]
mod fixture;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = openai::CompletionsClient::from_env()?;
    let embed_cfg = client.embedding_config(openai::TEXT_EMBEDDING_ADA_002);
    // The config knows its model's native width; no need to restate it.
    let embedding_dims = embed_cfg
        .ndims()
        .ok_or_else(|| anyhow::anyhow!("text-embedding-ada-002 has a known vector width"))?;
    let rt = client.http_runtime();

    // Initialize LanceDB locally.
    let db = lancedb::connect("data/lancedb-store").execute().await?;

    // Generate embeddings for the test data.
    let mut corpus = words();
    // Note: need at least 256 rows in order to create an index so copy the definition 256 times for testing purposes.
    corpus.extend((0..256).map(|i| Word {
        id: format!("doc{i}"),
        definition: "Definition of *flumbuzzle (noun)*: A sudden, inexplicable urge to rearrange or reorganize small objects, such as desk items or books, for no apparent reason.".to_string(),
    }));

    let embeddings = EmbeddingJob::new()
        .documents(corpus)
        .for_provider(&openai::functions::DESCRIPTOR)
        .run(|texts| openai::functions::embed(&embed_cfg, &rt, texts))
        .await?;

    let top_k = embeddings.len();

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

    // See [LanceDB indexing](https://lancedb.github.io/lancedb/concepts/index_ivfpq/#product-quantization) for more information
    if table.index_stats("embedding").await?.is_none() {
        table
            .create_index(
                &["embedding"],
                lancedb::index::Index::IvfPq(IvfPqIndexBuilder::default()),
            )
            .execute()
            .await?;
    }

    // Define search_params params that will be used by the vector store to perform the vector search.
    let search_params = SearchParams::default();
    let vector_store_index = LanceDbVectorIndex::new(table, "id", search_params).await?;

    let query = "My boss says I zindle too much, what does that mean?";

    // Queries are pre-embedded: embed the query text with the embedding model,
    // retrieve the most relevant documents, and pass them to the agent as
    // context.
    let query_embedding = openai::functions::embed(&embed_cfg, &rt, vec![query.to_string()])
        .await?
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no embedding returned for the query"))?;
    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), top_k as u64);
    let hits = vector_store_index.top_n(req).await?;

    let mut agent_builder = client
        .agent(openai::GPT_4O)
        .temperature(0.5)
        .preamble("You are a helpful AI assistant.");
    for hit in &hits {
        agent_builder = agent_builder.context(&hit.payload.to_string());
    }
    let agent = agent_builder.build();

    let response = agent.prompt(query).await?;

    println!("Response: {}", response);

    Ok(())
}

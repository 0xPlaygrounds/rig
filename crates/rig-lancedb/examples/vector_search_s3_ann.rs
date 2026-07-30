use fixture::{Word, as_record_batch, words};
use lancedb::{DistanceType, index::vector::IvfPqIndexBuilder};
use rig_core::OneOrMany;
use rig_core::embeddings::{default_concurrency, embed_documents};
use rig_core::http_runtime::HttpRuntime;
use rig_core::providers::openai;
use rig_core::vector_store::request::VectorSearchRequest;
use rig_lancedb::{LanceDbVectorIndex, SearchParams};

#[path = "./fixtures/lib.rs"]
mod fixture;

// Note: see docs to deploy LanceDB on other cloud providers such as google and azure.
// https://lancedb.github.io/lancedb/guides/storage/
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

    // Initialize LanceDB on S3.
    // Note: see below docs for more options and IAM permission required to read/write to S3.
    // https://lancedb.github.io/lancedb/guides/storage/#aws-s3
    let db = lancedb::connect("s3://lancedb-test-829666124233")
        .execute()
        .await?;

    // Generate embeddings for the test data.
    let mut corpus = words();
    // Note: need at least 256 rows in order to create an index so copy the definition 256 times for testing purposes.
    corpus.extend((0..256).map(|i| Word {
        id: format!("doc{i}"),
        definition: "Definition of *flumbuzzle (noun)*: A sudden, inexplicable urge to rearrange or reorganize small objects, such as desk items or books, for no apparent reason.".to_string(),
    }));

    let embeddings = embed_documents(
        corpus,
        max_documents,
        default_concurrency(max_documents),
        |texts| openai::functions::embed(&embed_cfg, &rt, texts),
    )
    .await?;

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
                lancedb::index::Index::IvfPq(
                    IvfPqIndexBuilder::default()
                        // This overrides the default distance type of L2.
                        // Needs to be the same distance type as the one used in search params.
                        .distance_type(DistanceType::Cosine),
                ),
            )
            .execute()
            .await?;
    }

    // Define search_params params that will be used by the vector store to perform the vector search.
    let search_params = SearchParams::default().distance_type(DistanceType::Cosine);

    let vector_store = LanceDbVectorIndex::new(table, "id", search_params).await?;

    // Queries are pre-embedded: embed the query text with the embedding model
    // and pass the embedding to the search request.
    let query = "I'm always looking for my phone, I always seem to forget it in the most counterintuitive places. What's the word for this feeling?";
    let query_embedding = openai::functions::embed(&embed_cfg, &rt, vec![query.to_string()])
        .await?
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no embedding returned for the query"))?;

    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 1);

    // Query the index
    let results = vector_store.top_n_as::<Word>(req).await?;

    println!("Results: {results:?}");

    Ok(())
}

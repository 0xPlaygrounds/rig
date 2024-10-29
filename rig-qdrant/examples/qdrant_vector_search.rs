// To run this example:
//
// export OPENAI_API_KEY=<YOUR-API-KEY>
//
// docker run -p 6333:63333 -p 6334:6334 qdrant/qdrant
//
// cargo run --release --example qdrant_vector_search

use std::env;

use qdrant_client::{
    qdrant::{
        CreateCollectionBuilder, Distance, PointStruct, QueryPointsBuilder, UpsertPointsBuilder,
        VectorParamsBuilder,
    },
    Payload, Qdrant,
};
use rig::{
    embeddings::EmbeddingsBuilder,
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::VectorStoreIndex,
};
use rig_qdrant::QdrantVectorStore;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    const COLLECTION_NAME: &str = "rig-collection";

    let client = Qdrant::from_url("http://localhost:6334").build()?;

    // Create a collection with 1536 dimensions if it doesn't exist
    if !client.collection_exists(COLLECTION_NAME).await? {
        client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION_NAME)
                    .vectors_config(VectorParamsBuilder::new(1536, Distance::Cosine)),
            )
            .await?;
    }

    // Initialize OpenAI client.
    // Get your API key from https://platform.openai.com/api-keys
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    let documents = EmbeddingsBuilder::new(model.clone())
        .simple_document("0981d983-a5f8-49eb-89ea-f7d3b2196d2e", "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .simple_document("62a36d43-80b6-4fd6-990c-f75bb02287d1", "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .simple_document("f9e17d59-32e5-440c-be02-b2759a654824", "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
        .build()
        .await?;

    let points: Vec<PointStruct> = documents
        .into_iter()
        .map(|d| {
            let vec: Vec<f32> = d.embeddings[0].vec.iter().map(|&x| x as f32).collect();
            PointStruct::new(
                d.id,
                vec,
                Payload::try_from(json!({
                    "document": d.document,
                }))
                .unwrap(),
            )
        })
        .collect();

    client
        .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME, points))
        .await?;

    let query_params = QueryPointsBuilder::new(COLLECTION_NAME).with_payload(true);
    let vector_store = QdrantVectorStore::new(client, model, query_params.build());

    let results = vector_store
        .top_n::<serde_json::Value>("What is a linglingdong?", 1)
        .await?;

    println!("Results: {:?}", results);

    Ok(())
}

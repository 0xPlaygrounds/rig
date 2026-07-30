// To run this example:
//
// export OPENAI_API_KEY=<YOUR-API-KEY>
// docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
// cargo run --release --example qdrant_vector_search
//
// You can view the data at http://localhost:6333/dashboard

use anyhow::anyhow;
use qdrant_client::{
    Qdrant,
    qdrant::{CreateCollectionBuilder, Distance, QueryPointsBuilder, VectorParamsBuilder},
};
use rig_core::OneOrMany;
use rig_core::{
    Embed,
    client::ProviderClient,
    embeddings::{EmbeddingModel, EmbeddingsBuilder},
    providers::openai::{self, Client},
    vector_store::{StoreRecord, request::SearchFilter},
};
use rig_core::{client::EmbeddingsClient, vector_store::request::VectorSearchRequest};
use rig_qdrant::{QdrantFilter, QdrantVectorStore};

#[derive(Embed, serde::Deserialize, serde::Serialize, Debug)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    const COLLECTION_NAME: &str = "rig-collection";

    let client = Qdrant::from_url("http://localhost:6334").build()?;

    // Create a collection with 1536 dimensions if it doesn't exist
    // Note: Make sure the dimensions match the size of the embeddings returned by the
    // model you are using
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
    let openai_client = Client::from_env()?;

    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    // Embedding happens *outside* the store: the store only ever sees
    // precomputed vectors.
    let documents = EmbeddingsBuilder::new(model.clone())
        .document(Word {
            id: "0981d983-a5f8-49eb-89ea-f7d3b2196d2e".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        })?
        .document(Word {
            id: "62a36d43-80b6-4fd6-990c-f75bb02287d1".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        })?
        .document(Word {
            id: "f9e17d59-32e5-440c-be02-b2759a654824".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        })?
        .build()
        .await?;

    let query_params = QueryPointsBuilder::new(COLLECTION_NAME).with_payload(true);
    let vector_store = QdrantVectorStore::new(client, query_params.build());

    let records = documents
        .into_iter()
        .map(|(word, embeddings)| StoreRecord::new(word.id.clone(), &word, embeddings))
        .collect::<Result<Vec<_>, _>>()?;

    vector_store
        .insert(records)
        .await
        .map_err(|err| anyhow!("Couldn't insert documents: {err}"))?;

    // Embed the query, then create a pre-embedded request.
    let query = "What is a linglingdong?";
    let query_embedding = model.embed_text(query).await?;

    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding.clone()), 1);

    let results = vector_store.top_n_as::<Word>(req).await?;

    println!("Results: {results:?}");

    let filtered_req = VectorSearchRequest::<QdrantFilter>::new(OneOrMany::one(query_embedding), 1)
        .with_filter(QdrantFilter::eq(
            "id",
            serde_json::json!("f9e17d59-32e5-440c-be02-b2759a654824"),
        ));

    let filtered_results = vector_store.top_n_as::<Word>(filtered_req).await?;

    println!("Filtered results: {filtered_results:?}");

    Ok(())
}

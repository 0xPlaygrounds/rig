use rig_core::{
    Embed,
    embeddings::EmbeddingsBuilder,
    providers::openai::{self, OpenAI},
    vector_store::{InsertDocuments, VectorStoreIndex, request::VectorSearchRequest},
};
use rig_falkordb::{FalkorDBClient, vector_index::IndexConfig};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Embed, Serialize, Deserialize)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = OpenAI::from_env()?.embedding(openai::TEXT_EMBEDDING_3_SMALL, None);
    let client = FalkorDBClient::connect(&std::env::var("FALKORDB_URI")?, "rig_documents").await?;
    client
        .create_vector_index(IndexConfig::new("Document"), model.capabilities().ndims)
        .await?;
    let index = client
        .get_index(model.clone(), "Document", "embedding")
        .await?;

    let documents = EmbeddingsBuilder::new(model)
        .documents([
            Word {
                id: "doc0".into(),
                definition: "A flurbo is a green alien that lives on cold planets.".into(),
            },
            Word {
                id: "doc1".into(),
                definition: "A glarb-glarb is an ancient farming tool from planet Jiro.".into(),
            },
            Word {
                id: "doc2".into(),
                definition: "A linglingdong is a lunar term for a human.".into(),
            },
        ])?
        .build()
        .await?;
    index.insert_documents(documents).await?;

    let request = VectorSearchRequest::builder()
        .query("What is a glarb?")
        .samples(1)
        .build();
    println!("{:#?}", index.top_n::<Word>(request.clone()).await?);
    println!("{:#?}", index.top_n_ids(request).await?);
    Ok(())
}

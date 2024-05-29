use aws_config::{meta::region::RegionProviderChain, BehaviorVersion};
use mongodb::{bson::doc, options::ClientOptions, Client as MongoClient, Collection};
use mongodb_utils::mongo_connection_string;
use std::env;

use llm::{
    embeddings::{DocumentEmbeddings, EmbeddingsBuilder},
    providers::openai::Client,
    vector_store::{mongodb_store::MongoDbVectorStore, VectorStore, VectorStoreIndex},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let aws_config = aws_config::defaults(BehaviorVersion::latest())
        .region(
            RegionProviderChain::default_provider()
                .or_else("us-east-1")
                .or_default_provider(),
        )
        .load()
        .await;

    // Init MongoDB client
    let options = ClientOptions::parse(mongo_connection_string(&aws_config).await)
        .await
        .expect("MongoDB connection string should be valid");

    let mongodb_client =
        MongoClient::with_options(options).expect("MongoDB client options should be valid");

    let model = openai_client.embedding_model("text-embedding-ada-002");

    let collection: Collection<DocumentEmbeddings> = mongodb_client
        .database("knowledgebase")
        .collection("context");

    let mut vector_store = MongoDbVectorStore::new(collection);

    let embeddings = EmbeddingsBuilder::new(model.clone())
        .simple_document("doc0", "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .simple_document("doc1", "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .simple_document("doc2", "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
        .build()
        .await?;

    match vector_store.add_documents(embeddings).await {
        Ok(_) => println!("Documents added successfully"),
        Err(e) => println!("Error adding documents: {:?}", e),
    }

    let index = vector_store.index(model, "context_vector_index", doc! {})?;

    let results = index
        .top_n_from_query("What is a linglingdong?", 1)
        .await?
        .into_iter()
        .map(|(score, doc)| (score, doc.id, doc.document))
        .collect::<Vec<_>>();

    println!("Results: {:?}", results);

    Ok(())
}

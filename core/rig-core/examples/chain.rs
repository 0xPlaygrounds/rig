use rig::prelude::*;
use rig::providers::openai;
use rig::{
    embeddings::EmbeddingsBuilder,
    parallel,
    pipeline::{self, Op, agent_ops::lookup, passthrough},
    providers::openai::Client,
    vector_store::in_memory_store::InMemoryVectorStore,
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().init();
    // Create OpenAI client
    let openai_client = Client::from_env();
    let embedding_model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    // Create embeddings for our documents
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .document("Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")?
        .document("Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")?
        .document("Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")?
        .build()
        .await?;

    // Create vector store with the embeddings
    let vector_store = InMemoryVectorStore::from_documents(embeddings);

    // Create vector store index
    let index = vector_store.index(embedding_model);
    let agent = openai_client.agent(openai::GPT_4)
        .preamble("
            You are a dictionary assistant here to assist the user in understanding the meaning of words.
        ")
        .build();

    let chain = pipeline::new()
        // Chain a parallel operation to the current chain. The parallel operation will
        // perform a lookup operation to retrieve additional context from the user prompt
        // while simultaneously applying a passthrough operation. The latter will allow
        // us to forward the initial prompt to the next operation in the chain.
        .chain(parallel!(
            passthrough::<&str>(),
            lookup::<_, _, String>(index, 1), // Required to specify document type
        ))
        // Chain a "map" operation to the current chain, which will combine the user
        // prompt with the retrieved context documents to create the final prompt.
        // If an error occurs during the lookup operation, we will log the error and
        // simply return the initial prompt.
        .map(|(prompt, maybe_docs)| match maybe_docs {
            Ok(docs) => format!(
                "Non standard word definitions:\n{}\n\n{}",
                docs.into_iter()
                    .map(|(_, _, doc)| doc)
                    .collect::<Vec<_>>()
                    .join("\n"),
                prompt,
            ),
            Err(err) => {
                println!("Error: {err}! Prompting without additional context");
                prompt.to_string()
            }
        })
        // Chain a "prompt" operation which will prompt out agent with the final prompt
        .prompt(agent);

    // Prompt the agent and print the response
    let response = chain.call("What does \"glarb-glarb\" mean?").await?;
    println!("{response}");

    Ok(())
}

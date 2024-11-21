use std::env;

use rig::{
    embeddings::EmbeddingsBuilder,
    parallel,
    pipeline::{self, agent_ops::lookup, passthrough, Op},
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::{in_memory_store::InMemoryVectorStore, VectorStore},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let embedding_model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Create vector store, compute embeddings and load them in the store
    let mut vector_store = InMemoryVectorStore::default();

    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .simple_document("doc0", "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .simple_document("doc1", "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .simple_document("doc2", "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
        .build()
        .await?;

    vector_store.add_documents(embeddings).await?;

    // Create vector store index
    let index = vector_store.index(embedding_model);

    let agent = openai_client.agent("gpt-4")
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
            passthrough(),
            lookup::<_, _, String>(index, 1), // Required to specify document type
        ))
        // Chain a "map" operation to the current chain, which will combine the user
        // prompt with the retrieved context documents to create the final prompt.
        // If an error occurs during the lookup operation, we will log the error and
        // simply return the initial prompt.
        .map(|(prompt, maybe_docs)| match maybe_docs {
            Ok(docs) => format!(
                "Non standard word definitions:\n{}\n\n{}",
                docs.join("\n"),
                prompt,
            ),
            Err(err) => {
                println!("Error: {}! Prompting without additional context", err);
                format!("{prompt}")
            }
        })
        // Chain a "prompt" operation which will prompt out agent with the final prompt
        .prompt(agent);

    // Prompt the agent and print the response
    let response = chain.call("What does \"glarb-glarb\" mean?").await?;

    println!("{:?}", response);

    Ok(())
}

// trait Foo<T> {
//     fn foo(&self);
// }

// impl<F, T, Out> Foo<(T,)> for F
// where
//     F: Fn(T) -> Out,
// {
//     fn foo(&self) {
//         todo!()
//     }
// }

// impl<F, T1, T2, Out> Foo<(T1, T2)> for F
// where
//     F: Fn(T1, T2) -> Out,
// {
//     fn foo(&self) {
//         todo!()
//     }
// }

// impl<F, T1, T2, T3, Out> Foo<(T1, T2, T3)> for F
// where
//     F: Fn(T1, T2, T3) -> Out,
// {
//     fn foo(&self) {
//         todo!()
//     }
// }

// impl<F, T, Fut> Foo<((Fut, T,),)> for F
// where
//     F: Fn(T) -> Fut,
//     Fut: Future,
// {
//     fn foo(&self) {
//         todo!()
//     }
// }

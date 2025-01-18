use std::env;

use rig::{
    embeddings::EmbeddingsBuilder,
    parallel,
    pipeline::{self, agent_ops::lookup, passthrough, Op, TryOp},
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::in_memory_store::InMemoryVectorStore,
};
use schemars::JsonSchema;
use rig::pipeline::agent_ops::extract;

#[derive(serde::Deserialize, JsonSchema, serde::Serialize)]
struct DocumentScore {
    /// The score of the document
    score: f32,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let manipulation_agent = openai_client.extractor::<DocumentScore>("gpt-4")
        .preamble("
            Your role is to score a user's statement on how manipulative it sounds between 0 and 1.
        ")
        .build();

    let depression_agent = openai_client.extractor::<DocumentScore>("gpt-4")
        .preamble("
            Your role is to score a user's statement on how depressive it sounds between 0 and 1.
        ")
        .build();

    let intelligent_agent = openai_client.extractor::<DocumentScore>("gpt-4")
        .preamble("
            Your role is to score a user's statement on how intelligent it sounds between 0 and 1.
        ")
        .build();

    let statement_agent = openai_client.agent("gpt-4")
        .preamble("
            Your role is to make a statement about the user based on the original statement, as well as the provided manipulation, depression and intelligence scores.
        ")
        .build();

    let chain = pipeline::new()
        .chain(parallel!(passthrough(), extract(manipulation_agent), extract(depression_agent), extract(intelligent_agent)))
        .map(|(statement, manip_score, dep_score, int_score)| {
            format!("
                Original statement: {statement}
                Manipulation sentiment score: {}
                Depression sentiment score: {}
                Intelligence sentiment score: {}
                ",
                manip_score.unwrap().score,
                dep_score.unwrap().score,
                int_score.unwrap().score
        )
        });

    // Prompt the agent and print the response
    let response = chain.try_call("I hate swimming. The water always gets in my eyes.").await;

    match response {
        Ok(res) => println!("Successful pipeline run: {res:?}"),
        Err(e) => println!("Unsuccessful pipeline run: {e:?}")
    }

    Ok(())
}

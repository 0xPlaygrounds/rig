use rig::prelude::*;

use rig::providers::openai;
use rig::providers::openai::client::Client;

use schemars::JsonSchema;

#[derive(Debug, serde::Deserialize, JsonSchema, serde::Serialize)]
struct DocumentScore {
    /// The score of the document
    score: f32,
}
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_client = Client::from_env()?;

    let manipulation_agent = openai_client
        .extractor::<DocumentScore>(openai::GPT_4)
        .preamble(
            "
            Your role is to score a user's statement on how manipulative it sounds between 0 and 1.
        ",
        )
        .build();

    let depression_agent = openai_client
        .extractor::<DocumentScore>(openai::GPT_4)
        .preamble(
            "
            Your role is to score a user's statement on how depressive it sounds between 0 and 1.
        ",
        )
        .build();

    let intelligent_agent = openai_client
        .extractor::<DocumentScore>(openai::GPT_4)
        .preamble(
            "
            Your role is to score a user's statement on how intelligent it sounds between 0 and 1.
        ",
        )
        .build();

    // Score the statement on three dimensions concurrently. `join!` (unlike
    // `try_join!`) awaits all three and keeps each `Result`, so one failed
    // extraction doesn't discard the others — the same behaviour the old
    // `parallel!` op provided.
    let statement = "I hate swimming. The water always gets in my eyes.";
    let (manip_score, dep_score, int_score) = futures::join!(
        manipulation_agent.extract(statement),
        depression_agent.extract(statement),
        intelligent_agent.extract(statement),
    );

    let response = match (manip_score, dep_score, int_score) {
        (Ok(manip_score), Ok(dep_score), Ok(int_score)) => format!(
            "
                    Original statement: {statement}
                    Manipulation sentiment score: {}
                    Depression sentiment score: {}
                    Intelligence sentiment score: {}
                    ",
            manip_score.score, dep_score.score, int_score.score
        ),
        (manip_score, dep_score, int_score) => format!(
            "
                    Original statement: {statement}
                    Manipulation sentiment score: {manip_score:?}
                    Depression sentiment score: {dep_score:?}
                    Intelligence sentiment score: {int_score:?}
                    "
        ),
    };

    println!("Pipeline run: {response:?}");

    Ok(())
}

use rig::extractor::ExtractorBuilder;
use rig::prelude::*;
use std::future::IntoFuture;

use rig::providers::openai::{self, OpenAI};

use schemars::JsonSchema;

#[derive(Debug, serde::Deserialize, JsonSchema, serde::Serialize)]
struct DocumentScore {
    /// The score of the document
    score: f32,
}
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Bind the OpenAI Responses API to the default transport
    let openai_client = OpenAI::from_env()?.bound()?;
    let model = openai_client.completion(openai::GPT_4);

    let manipulation_agent = ExtractorBuilder::<DocumentScore>::new(model.clone())
        .append_preamble(
            "
            Your role is to score a user's statement on how manipulative it sounds between 0 and 1.
        ",
        )
        .build();

    let depression_agent = ExtractorBuilder::<DocumentScore>::new(model.clone())
        .append_preamble(
            "
            Your role is to score a user's statement on how depressive it sounds between 0 and 1.
        ",
        )
        .build();

    let intelligent_agent = ExtractorBuilder::<DocumentScore>::new(model)
        .append_preamble(
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
        manipulation_agent.extract(statement).into_future(),
        depression_agent.extract(statement).into_future(),
        intelligent_agent.extract(statement).into_future(),
    );

    let response = match (manip_score, dep_score, int_score) {
        (Ok(manip_score), Ok(dep_score), Ok(int_score)) => format!(
            "
                    Original statement: {statement}
                    Manipulation sentiment score: {}
                    Depression sentiment score: {}
                    Intelligence sentiment score: {}
                    ",
            manip_score.output.score, dep_score.output.score, int_score.output.score
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

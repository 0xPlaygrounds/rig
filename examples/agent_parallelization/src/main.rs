//! Fan one statement out to three structured "scorers" concurrently.
//!
//! Requires `OPENAI_API_KEY`.
use rig::agent::Agent;
use rig::extract::ExtractOptions;
use rig::prelude::*;

use rig::providers::openai;

use schemars::JsonSchema;

#[derive(Debug, serde::Deserialize, JsonSchema, serde::Serialize)]
struct DocumentScore {
    /// The score of the document
    score: f32,
}

/// One scorer: the classic extraction protocol with the role instructions
/// appended to the extraction preamble (what `ExtractorBuilder::preamble` did).
fn scorer(role: &str) -> String {
    let classic = ExtractOptions::classic_extractor();
    let preamble = classic.preamble.clone().unwrap_or_default();
    format!("{preamble}\n=============== ADDITIONAL INSTRUCTIONS ===============\n{role}")
}

async fn score(
    agent: &Agent,
    preamble: String,
    statement: &str,
) -> Result<DocumentScore, anyhow::Error> {
    Ok(agent.extractor(statement).preamble(preamble).run().await?)
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = openai::Client::from_env()?;
    let agent = client.agent(openai::GPT_4).build();

    let manipulation_scorer = scorer(
        "
            Your role is to score a user's statement on how manipulative it sounds between 0 and 1.
        ",
    );

    let depression_scorer = scorer(
        "
            Your role is to score a user's statement on how depressive it sounds between 0 and 1.
        ",
    );

    let intelligent_scorer = scorer(
        "
            Your role is to score a user's statement on how intelligent it sounds between 0 and 1.
        ",
    );

    // Score the statement on three dimensions concurrently. `join!` (unlike
    // `try_join!`) awaits all three and keeps each `Result`, so one failed
    // extraction doesn't discard the others — the same behaviour the old
    // `parallel!` op provided.
    let statement = "I hate swimming. The water always gets in my eyes.";
    let (manip_score, dep_score, int_score) = futures::join!(
        score(&agent, manipulation_scorer, statement),
        score(&agent, depression_scorer, statement),
        score(&agent, intelligent_scorer, statement),
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

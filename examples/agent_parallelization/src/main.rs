//! Fan one statement out to three structured "scorers" concurrently.
//!
//! The extractor builder is gone: a scorer is now just the plain data one
//! [`extract_with_options`] call needs — an [`AgentConfig`] carrying the role
//! preamble plus the classic extractor protocol from
//! [`ExtractOptions::classic_extractor()`], aimed at a [`ProviderConfig`] built
//! from `openai::functions::Config::from_env(MODEL)`. The parallelization
//! itself is unchanged.
//!
//! Requires `OPENAI_API_KEY`.
use std::sync::Arc;

use rig::agent::AgentConfig;
use rig::extract::{ExtractOptions, extract_with_options};
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
fn scorer(role: &str) -> ExtractOptions {
    let classic = ExtractOptions::classic_extractor();
    let preamble = classic.preamble.clone().unwrap_or_default();
    classic.with_preamble(format!(
        "{preamble}\n=============== ADDITIONAL INSTRUCTIONS ===============\n{role}"
    ))
}

async fn score(
    provider: &ProviderConfig,
    rt: &Arc<Runtime>,
    options: ExtractOptions,
    statement: &str,
) -> Result<DocumentScore, anyhow::Error> {
    let outcome = extract_with_options::<DocumentScore>(
        AgentConfig::new(),
        provider.clone(),
        rt.clone(),
        statement,
        options,
    )
    .await?;
    Ok(outcome.value)
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // The provider is plain data: a config that names the model.
    let provider = ProviderConfig::OpenAi(openai::functions::Config::from_env(openai::GPT_4)?);
    let rt = Arc::new(Runtime::new());

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
        score(&provider, &rt, manipulation_scorer, statement),
        score(&provider, &rt, depression_scorer, statement),
        score(&provider, &rt, intelligent_scorer, statement),
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

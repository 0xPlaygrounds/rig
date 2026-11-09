//! Cassette-backed TypeSafe Jev evaluation routed through OpenRouter's
//! `/systemone`, whose `usage` adds the charge as a bare number of USD and
//! whose request can carry `provider` preferences.

use rig::completion::Cost;
use rig::providers::openrouter::extension::{DataCollection, ProviderPreferences};
use rig::typesafeai::{Evaluate, Noul, Query};

use super::super::support::with_openrouter_jev_cassette_result;
use crate::cassettes::CassetteMode;

/// OpenRouter reports `usage.cost` as a number, not rig's `Cost` object. The
/// answer decodes and the cost is the reported figure.
#[tokio::test]
async fn evaluation_reads_the_reported_cost() -> anyhow::Result<()> {
    with_openrouter_jev_cassette_result(
        "jev/evaluation_reads_the_reported_cost",
        |config| config,
        |jev| async move {
            let question =
                Noul::new("Does the customer report a billing problem?")?.named("billing")?;
            let state = serde_json::json!({"message": "I was charged twice for one order."});
            let result = jev.evaluation().evaluate(&state, question).await?;
            anyhow::ensure!((0.0..=1.0).contains(&result.answers.noul));

            let cost = result.usage.and_then(|usage| usage.cost);
            anyhow::ensure!(
                cost.is_some_and(|cost| cost.total > 0.0),
                "OpenRouter reports what the evaluation cost"
            );
            if CassetteMode::current() == CassetteMode::Replay {
                let recorded = crate::cassettes::recorded_interaction_bodies(
                    "openrouter",
                    "jev/evaluation_reads_the_reported_cost",
                );
                let [(_, body)] = recorded.as_slice() else {
                    anyhow::bail!("the scenario records one exchange");
                };
                let body: serde_json::Value = serde_json::from_str(body)?;
                let recorded = body["usage"]["cost"].as_f64().map(Cost::from_total);
                anyhow::ensure!(cost == recorded, "the cost is the recorded figure");
            }
            Ok(())
        },
    )
    .await
}

/// The request body carries zero-data-retention preferences beside the model,
/// and the recorded reply decodes into an answer.
#[tokio::test]
async fn evaluation_sends_provider_preferences() -> anyhow::Result<()> {
    with_openrouter_jev_cassette_result(
        "jev/evaluation_sends_provider_preferences",
        |config| {
            config.provider(
                ProviderPreferences::new()
                    .zdr(true)
                    .data_collection(DataCollection::Deny),
            )
        },
        |jev| async move {
            let question =
                Noul::new("Does the customer report a billing problem?")?.named("billing")?;
            let state = serde_json::json!({"message": "I was charged twice for one order."});
            let result = jev.evaluation().evaluate(&state, question).await?;
            anyhow::ensure!((0.0..=1.0).contains(&result.answers.noul));
            Ok(())
        },
    )
    .await
}

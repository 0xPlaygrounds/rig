//! Cassette-backed TypeSafe Jev evaluation routed through OpenRouter's
//! `/systemone`, whose `usage` adds the charge as a bare number of USD.

use rig::completion::Cost;
use rig::typesafeai::{Evaluate, Noul, Query};

use super::super::support::with_openrouter_jev_cassette_result;
use crate::cassettes::CassetteMode;

/// rig#2782: OpenRouter's numeric `usage.cost` failed the whole evaluation,
/// since it was decoded as rig's `Cost` object. The answer decodes and the
/// cost is the reported figure.
#[tokio::test]
async fn evaluation_reads_the_reported_cost() -> anyhow::Result<()> {
    with_openrouter_jev_cassette_result(
        "jev/evaluation_reads_the_reported_cost",
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

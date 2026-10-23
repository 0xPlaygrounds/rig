//! Cassette-backed TypeSafe Jev evaluation routed through OpenRouter's
//! `/systemone`, whose `usage` adds the charge as a bare number of USD and
//! whose routing follows the request's `provider` preferences.

use rig::completion::Cost;
use rig::error::ProviderError;
use rig::providers::openrouter::extension::{DataCollection, ProviderPreferences};
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

/// Zero-data-retention preferences travel beside the model, and OpenRouter
/// serves the evaluation from an endpoint that meets them.
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

/// OpenRouter applies the preferences: a route no provider of the model can
/// serve is refused before any evaluation.
#[tokio::test]
async fn evaluation_with_an_unservable_route_is_refused() -> anyhow::Result<()> {
    with_openrouter_jev_cassette_result(
        "jev/evaluation_with_an_unservable_route_is_refused",
        |config| {
            config.provider(
                ProviderPreferences::new()
                    .only(["openai"])
                    .allow_fallbacks(false),
            )
        },
        |jev| async move {
            let question =
                Noul::new("Does the customer report a billing problem?")?.named("billing")?;
            let state = serde_json::json!({"message": "I was charged twice for one order."});
            let Err(ProviderError::ProviderResponse(response)) =
                jev.evaluation().evaluate(&state, question).await
            else {
                anyhow::bail!("the route must be refused");
            };
            anyhow::ensure!(response.status.map(|status| status.as_u16()) == Some(404));
            anyhow::ensure!(
                response.body.contains("No allowed providers"),
                "{}",
                response.body
            );
            Ok(())
        },
    )
    .await
}

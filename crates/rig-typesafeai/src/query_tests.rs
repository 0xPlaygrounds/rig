use crate::{ChoiceAnswer, Evaluate, Jev, NoulAnswer, Query, ScoreAnswer, types};
use anyhow::ensure;
use rig_core::driver::Bind;
use rig_core::error::ProviderError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Route {
    Billing,
    Technical,
    Other,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
enum Urgency {
    NoDeadline,
    WithinWeek,
    Today,
}

#[derive(Debug, Serialize, Deserialize)]
struct Assessment<R = ChoiceAnswer<Route>, U = ScoreAnswer<Urgency>, F = NoulAnswer> {
    route: R,
    urgency: U,
    refund_requested: F,
}

type AssessmentQuery = Assessment<crate::Choice<Route>, crate::Score<Urgency>, crate::Noul>;

impl AssessmentQuery {
    fn new() -> Result<Self, rig_core::error::ProviderError> {
        Ok(Self {
            route: crate::Choice::<Route>::new(
                "Which team should handle the customer message?",
                [
                    (Route::Billing, "Charges, payments, and refunds"),
                    (Route::Technical, "Software faults"),
                    (Route::Other, "Anything else"),
                ],
            )?,
            urgency: crate::Score::<Urgency>::new(
                "How soon does the customer ask for action?",
                [
                    (Urgency::NoDeadline, "No deadline"),
                    (Urgency::WithinWeek, "Within a week"),
                    (Urgency::Today, "Today"),
                ],
            )?,
            refund_requested: crate::Noul::new("Does the customer explicitly request a refund?")?,
        })
    }
}

impl<R: crate::Query, U: crate::Query, F: crate::Query> crate::Query for Assessment<R, U, F> {
    type Response = Assessment<R::Response, U::Response, F::Response>;
    type Output = Assessment<R::Output, U::Output, F::Output>;

    fn decode(
        &self,
        response: Self::Response,
    ) -> Result<Self::Output, rig_core::error::ProviderError> {
        Ok(Assessment {
            route: self.route.decode(response.route)?,
            urgency: self.urgency.decode(response.urgency)?,
            refund_requested: self.refund_requested.decode(response.refund_requested)?,
        })
    }
}

#[derive(Deserialize)]
struct Fixture {
    request: Value,
    response: Value,
}

/// The named query must reproduce the same recorded API request as the builders.
#[tokio::test]
async fn named_query_replays_and_assessment_roundtrips() -> anyhow::Result<()> {
    let fixture: Fixture = serde_json::from_str(include_str!("../fixtures/triage.json"))?;
    let schema = AssessmentQuery::new()?;
    let encoded = serde_json::to_value(&schema)?;
    ensure!(Some(&encoded) == fixture.request.get("questions"));
    let server = httpmock::MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/systemone")
                .json_body(fixture.request.clone());
            then.status(200)
                .header("content-type", "application/json")
                .json_body(fixture.response.clone());
        })
        .await;
    let client = Jev::new("test-token")
        .with_endpoint(server.url("/v1/systemone"))
        .bind(rig_reqwest::ReqwestClient::default());
    let state = fixture
        .request
        .get("state")
        .ok_or_else(|| anyhow::anyhow!("missing fixture state"))?;
    let result = client.evaluate(state, &schema).await?;
    ensure!(result.answers.route.choice == Route::Billing);
    ensure!(result.answers.urgency.score == 2.0);
    ensure!(result.answers.urgency.probabilities.get(&Urgency::Today) == Some(&1.0));
    ensure!(result.answers.refund_requested.noul == 0.99);
    let saved = serde_json::to_value(&result.answers)?;
    ensure!(saved.get("refund_requested").is_some());
    let restored: Assessment = serde_json::from_value(saved)?;
    ensure!(restored.refund_requested.noul == result.answers.refund_requested.noul);
    mock.assert_async().await;
    Ok(())
}

/// Named queries share the runtime validation boundary; typed fields do not trust remote JSON.
#[test]
fn named_query_rejects_mismatched_answer_types() -> anyhow::Result<()> {
    let fixture: Fixture = serde_json::from_str(include_str!("../fixtures/triage.json"))?;
    let mut answers = fixture
        .response
        .get("answers")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing answers"))?;
    let route = answers
        .get_mut("route")
        .ok_or_else(|| anyhow::anyhow!("missing route"))?;
    *route = serde_json::json!({"type":"noul","noul":0.8});
    let schema = AssessmentQuery::new()?;
    ensure!(matches!(
        schema.decode(serde_json::from_value(answers)?),
        Err(ProviderError::Response(_))
    ));
    Ok(())
}

#[test]
fn typed_score_uses_supplied_order_and_validates_the_rubric() -> anyhow::Result<()> {
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    enum Severity {
        Low = 5,
        High = 20,
    }
    let query = crate::Score::new(
        serde_json::json!({"question":"Impact?"}),
        [
            (
                Severity::High,
                serde_json::json!({"what":"Blocked", "not_for":"Workaround available", "examples":["Checkout down"]}),
            ),
            (Severity::Low, serde_json::json!("Routine")),
        ],
    )?;
    let mut response: types::Answer = serde_json::from_value(serde_json::json!({
            "type":"score", "score":0.75, "confidence":0.5,
            "probabilities":{"0":0.25, "1":0.75},
            "legend":{"0":{"what":"Blocked", "not_for":"Workaround available", "examples":["Checkout down"]},"1":"Routine"}
    }))?;
    let answer: crate::ScoreAnswer<Severity> = query.decode(response.clone())?;
    ensure!(answer.probabilities.get(&Severity::High) == Some(&0.25));
    ensure!(answer.probabilities.get(&Severity::Low) == Some(&0.75));
    let types::Answer::Score { legend, .. } = &mut response else {
        anyhow::bail!("missing score");
    };
    legend.insert("0".into(), Value::String("Wrong description".into()));
    ensure!(query.decode(response).is_err());
    ensure!(
        crate::Score::new(
            "Impact?",
            [(Severity::High, "High"), (Severity::High, "Duplicate")]
        )
        .is_err()
    );
    ensure!(crate::Score::try_from_iter("Impact?", [(Severity::High, "High")]).is_err());
    ensure!(crate::Score::try_from_iter("Impact?", (0..11).map(|i| (i, "level"))).is_err());
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct Renamed<Q> {
    #[serde(rename = "is_ready")]
    ready: Q,
}
impl<Q: Query> Query for Renamed<Q> {
    type Response = Renamed<Q::Response>;
    type Output = Renamed<Q::Output>;
    fn decode(
        &self,
        response: Self::Response,
    ) -> Result<Self::Output, rig_core::error::ProviderError> {
        Ok(Renamed {
            ready: self.ready.decode(response.ready)?,
        })
    }
}

#[test]
fn serde_names_and_joined_runtime_questions_decode_independently() -> anyhow::Result<()> {
    let named = Renamed {
        ready: crate::Noul::new("Ready?")?,
    };
    let dynamic =
        std::collections::BTreeMap::from([("review".to_owned(), crate::Noul::new("Review?")?)]);
    let query = named.join(dynamic);
    let encoded = serde_json::value::to_raw_value(&query)?;
    let ids = crate::validation::request_ids(&encoded)?;
    ensure!(ids.into_iter().collect::<Vec<_>>() == ["is_ready", "review"]);
    let (named, dynamic) = query.decode(serde_json::json!({
        "is_ready": {"type":"noul", "noul":0.9},
        "review": {"type":"noul", "noul":0.2}
    }))?;
    ensure!(named.ready.noul == 0.9);
    ensure!(dynamic.get("review").map(|answer| answer.noul) == Some(0.2));
    Ok(())
}

#[test]
fn serde_boundary_rejects_duplicate_keys_and_invalid_request_shapes() -> anyhow::Result<()> {
    let query = Renamed {
        ready: crate::Noul::new("Ready?")?,
    }
    .join(crate::Noul::new("Again?")?.named("is_ready")?);
    let raw = serde_json::value::to_raw_value(&query)?;
    ensure!(matches!(
        crate::validation::request_ids(&raw),
        Err(ProviderError::Request(_))
    ));
    for json in [
        "{}",
        "[]",
        r#"{"":{"type":"noul","instructions":"Ready?"}}"#,
        r#"{"nested":{"ready":{"type":"noul","instructions":"Ready?"}}}"#,
    ] {
        let raw = serde_json::value::RawValue::from_string(json.into())?;
        ensure!(matches!(
            crate::validation::request_ids(&raw),
            Err(ProviderError::Request(_))
        ));
    }
    let raw = serde_json::value::RawValue::from_string(
        r#"{"ready":{"type":"noul","noul":0.1},"ready":{"type":"noul","noul":0.9}}"#.into(),
    )?;
    ensure!(matches!(
        crate::validation::response_ids(&raw),
        Err(ProviderError::Response(_))
    ));
    Ok(())
}

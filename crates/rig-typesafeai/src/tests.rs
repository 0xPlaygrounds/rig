use super::*;
use crate::decode::DecodeAnswer;
use anyhow::ensure;
use rig_core::driver::Bind;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Route {
    Billing,
    Technical,
    Other,
}

#[derive(Serialize, Deserialize)]
struct Batch<R, U, F> {
    route: R,
    urgency: U,
    #[serde(rename = "refund_requested")]
    refund: F,
}
impl<R: Query, U: Query, F: Query> Query for Batch<R, U, F> {
    type Response = Batch<R::Response, U::Response, F::Response>;
    type Output = Batch<R::Output, U::Output, F::Output>;
    fn decode(&self, response: Self::Response) -> Result<Self::Output, ProviderError> {
        Ok(Batch {
            route: self.route.decode(response.route)?,
            urgency: self.urgency.decode(response.urgency)?,
            refund: self.refund.decode(response.refund)?,
        })
    }
}

fn route() -> Result<Choice<Route>, ProviderError> {
    Choice::new(
        "Which team should handle the customer message?",
        [
            (Route::Billing, "Charges, payments, and refunds"),
            (Route::Technical, "Software faults"),
            (Route::Other, "Anything else"),
        ],
    )
}
#[derive(Deserialize)]
struct Fixture {
    request: Value,
    response: Value,
}
fn fixture() -> Result<Fixture, serde_json::Error> {
    serde_json::from_str(include_str!("../fixtures/triage.json"))
}

/// Replays actual Jev traffic and asserts the outbound request as well as typed answers.
#[tokio::test]
async fn recorded_mixed_batch() -> anyhow::Result<()> {
    let fixture = fixture()?;
    let server = httpmock::MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/systemone")
                .header("authorization", "Bearer test-token")
                .json_body(fixture.request.clone());
            then.status(200)
                .header("content-type", "application/json")
                .header("x-typesafe-request-id", "test-request-id")
                .json_body(fixture.response.clone());
        })
        .await;
    let route = route()?;
    let urgency = DynamicScore::new(
        "How soon does the customer ask for action?",
        ["No deadline", "Within a week", "Today"],
    )?;
    let refund = Noul::new("Does the customer explicitly request a refund?")?;
    let client = Jev::new("test-token")
        .with_endpoint(server.url("/v1/systemone"))
        .bind(rig_reqwest::ReqwestClient::default());
    let result = client
        .evaluate(
            &fixture
                .request
                .get("state")
                .ok_or_else(|| anyhow::anyhow!("missing fixture state"))?,
            Batch {
                route: &route,
                urgency: &urgency,
                refund: &refund,
            },
        )
        .await?;
    ensure!(result.answers.route.choice == Route::Billing);
    ensure!(result.answers.route.probabilities.get(&Route::Billing) == Some(&1.0));
    ensure!(result.answers.urgency.score == 2.0);
    ensure!(result.answers.refund.noul == 0.99);
    ensure!(result.model == "jev-1.13.0");
    ensure!(result.provider_request_id.as_deref() == Some("test-request-id"));
    mock.assert_async().await;
    Ok(())
}

/// Local schema invariants cannot be established by provider recordings.
#[test]
fn rejects_duplicate_ids_and_choices() -> anyhow::Result<()> {
    let q = route()?;
    ensure!(matches!(
        validation::request_ids(&serde_json::value::to_raw_value(
            &(&q).named("route")?.join((&q).named("route")?)
        )?),
        Err(ProviderError::Request(_))
    ));
    ensure!(Choice::new("route", [(Route::Billing, "one"), (Route::Billing, "two")]).is_err());
    ensure!(DynamicScore::try_from_iter("rate", ["only one"]).is_err());
    Ok(())
}

/// Mutations test the validation boundary rather than claiming provider behavior.
#[test]
fn rejects_corrupt_answers() -> anyhow::Result<()> {
    let q = route()?;
    for replacement in [
        json!({"type":"noul","noul":0.5}),
        json!({"type":"choice","choice":"unknown","confidence":1.0,"probabilities":{"unknown":1.0}}),
        json!({"type":"choice","choice":"billing","confidence":1.0,"probabilities":{"billing":0.5,"technical":0.0,"other":0.0}}),
        json!({"type":"choice","choice":"billing","confidence":1.2,"probabilities":{"billing":1.0,"technical":0.0,"other":0.0}}),
    ] {
        ensure!(matches!(
            q.decode(serde_json::from_value(replacement)?),
            Err(ProviderError::Response(_))
        ));
    }
    Ok(())
}

/// Credentials must stay redacted even when users debug provider configuration.
#[test]
fn redacts_configuration() -> anyhow::Result<()> {
    let wire = Jev::new("sensitive-test-value");
    ensure!(!format!("{wire:?}").contains("sensitive-test-value"));
    ensure!(!serde_json::to_string(&wire)?.contains("sensitive-test-value"));
    Ok(())
}

/// Asymmetric Serde labels can reverse routing before the model is involved.
#[test]
fn rejects_labels_that_decode_to_a_different_variant() {
    #[derive(Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
    enum Asymmetric {
        #[serde(rename(serialize = "first", deserialize = "second"))]
        First,
        #[serde(rename(serialize = "second", deserialize = "first"))]
        Second,
    }
    assert!(matches!(
        Choice::new(
            "choose",
            [(Asymmetric::First, "one"), (Asymmetric::Second, "two")]
        ),
        Err(ProviderError::Request(_))
    ));
}

/// Replays structured criteria accepted by Jev, including an undescribed choice.
#[tokio::test]
async fn recorded_structured_batch_into_named_answers() -> anyhow::Result<()> {
    #[derive(Serialize)]
    struct Description<'a> {
        description: &'a str,
        examples: [&'a str; 1],
    }
    #[derive(Serialize)]
    struct Deadline<'a> {
        deadline: &'a str,
    }
    struct Assessment {
        route: ChoiceAnswer<Route>,
        urgency: DynamicScoreAnswer,
        refund: NoulAnswer,
    }
    let fixture: Fixture = serde_json::from_str(include_str!("../fixtures/structured.json"))?;
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
    let route = Choice::new(
        json!({"question":"Which team should handle this request?","focus":"Use the customer message"}),
        [
            (
                Route::Billing,
                Some(Description {
                    description: "Charges and refunds",
                    examples: ["duplicate charge"],
                }),
            ),
            (
                Route::Technical,
                Some(Description {
                    description: "Software faults",
                    examples: ["page crashes"],
                }),
            ),
            (Route::Other, None),
        ],
    )?;
    let urgency = DynamicScore::new(
        "How soon does the customer ask for action?",
        [
            Deadline { deadline: "none" },
            Deadline {
                deadline: "within a week",
            },
            Deadline { deadline: "today" },
        ],
    )?;
    let refund = Noul::new(json!({"question":"Does the customer explicitly request a refund?"}))?
        .criteria(
        Description {
            description: "Explicit refund request",
            examples: ["please refund"],
        },
        Description {
            description: "No explicit refund request",
            examples: ["explain this charge"],
        },
    )?;
    let questions = Batch {
        route: &route,
        urgency: &urgency,
        refund: &refund,
    }
    .map(|answer| Assessment {
        route: answer.route,
        urgency: answer.urgency,
        refund: answer.refund,
    });
    let client = Jev::new("test-token")
        .with_endpoint(server.url("/v1/systemone"))
        .bind(rig_reqwest::ReqwestClient::default());
    let state = fixture
        .request
        .get("state")
        .ok_or_else(|| anyhow::anyhow!("missing state"))?;
    // Borrowing allows a static schema to be reused across independent states.
    let result = client.evaluate(state, &questions).await?;
    ensure!(result.answers.route.choice == Route::Billing);
    ensure!(result.answers.urgency.score == 2.0);
    ensure!(result.answers.refund.noul == 0.99);
    let raw: types::Response = serde_json::from_value(fixture.response)?;
    ensure!(
        questions
            .decode(serde_json::from_str(raw.answers.get())?)?
            .route
            .choice
            == Route::Billing
    );
    mock.assert_async().await;
    Ok(())
}

/// Description shape and dynamic counts are local construction invariants.
#[test]
fn validates_description_shapes_and_dynamic_counts() -> anyhow::Result<()> {
    let empty: Vec<(Route, &str)> = Vec::new();
    ensure!(Choice::try_from_iter("choose", empty).is_err());
    ensure!(DynamicScore::try_from_iter("rate", vec!["level"; 11]).is_err());
    ensure!(Noul::new(42).is_err());
    ensure!(
        Noul::new(Value::Null)?
            .criteria(Value::Null, json!(["no evidence"]))
            .is_ok()
    );
    ensure!(questions::state(Value::Null).is_err());
    ensure!(questions::state(json!({"message":"hello"})).is_ok());
    Ok(())
}

/// Replays the full captured decision matrix whose rounded score weights sum to 0.99.
#[tokio::test]
async fn recorded_rounded_probability_matrix() -> anyhow::Result<()> {
    let fixture: Fixture = serde_json::from_str(include_str!("../fixtures/rounded.json"))?;
    let definitions = serde_json::from_value(
        fixture
            .request
            .get("questions")
            .ok_or_else(|| anyhow::anyhow!("missing questions"))?
            .clone(),
    )?;
    let schema = DynamicQuery::new(definitions)?;
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
        .ok_or_else(|| anyhow::anyhow!("missing state"))?;
    let result = client.evaluate(state, &schema).await?;
    let score = DynamicScoreAnswer::decode(
        result
            .answers
            .get("a0_progress")
            .ok_or_else(|| anyhow::anyhow!("missing progress"))?,
    )?;
    ensure!(score.probabilities == BTreeMap::from([(0, 0.01), (1, 0.23), (2, 0.67), (3, 0.08)]));
    ensure!((score.probabilities.values().sum::<f64>() - 0.99).abs() < 1e-12);
    let actual = serde_json::to_value(&result.answers)?;
    let actual_answers = &actual;
    let recorded_answers = fixture
        .response
        .get("answers")
        .ok_or_else(|| anyhow::anyhow!("missing recorded answers"))?;
    ensure!(actual_answers == recorded_answers);
    mock.assert_async().await;
    Ok(())
}

/// Synthetic edge cases bound the rounding exception without inventing provider traffic.
#[test]
fn bounds_probability_rounding_tolerance() -> anyhow::Result<()> {
    let distribution = |weights: &[f64]| {
        questions::distribution(
            &weights
                .iter()
                .enumerate()
                .map(|(index, weight)| (index.to_string(), *weight))
                .collect(),
        )
    };
    for weights in [
        [0.01, 0.23, 0.67, 0.08],
        [0.02, 0.23, 0.67, 0.09],
        [0.24, 0.24, 0.25, 0.25],
        [0.25, 0.25, 0.26, 0.26],
    ] {
        distribution(&weights)?;
    }
    for weights in [
        vec![0.5, 0.0, 0.0],
        vec![0.24, 0.24, 0.24, 0.25],
        vec![0.241, 0.239, 0.25, 0.26],
        vec![0.0; 255],
        vec![0.01; 105],
        vec![f64::NAN, 1.0],
        vec![f64::INFINITY, 0.0],
        vec![-0.01, 1.01],
    ] {
        ensure!(distribution(&weights).is_err());
    }
    Ok(())
}

#[test]
fn rejects_non_string_choice_serialization() {
    #[derive(PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    #[serde(tag = "kind")]
    enum Tagged {
        One,
        Two,
    }
    assert!(matches!(
        Choice::new("Choose", [(Tagged::One, "One"), (Tagged::Two, "Two")]),
        Err(ProviderError::Request(_))
    ));
    assert!(matches!(
        Choice::new("Choose", [(1u8, "One"), (2u8, "Two")]),
        Err(ProviderError::Request(_))
    ));
}

#[tokio::test]
async fn preserves_http_error_metadata() -> anyhow::Result<()> {
    for status in [429, 529] {
        let server = httpmock::MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/v1/systemone");
                then.status(status)
                    .header("content-type", "application/json")
                    .header("x-typesafe-request-id", "error-request")
                    .body(r#"{"error":"overloaded"}"#);
            })
            .await;
        let client = Jev::new("test-token")
            .with_endpoint(server.url("/v1/systemone"))
            .bind(rig_reqwest::ReqwestClient::default());
        let error = client
            .evaluate(&"state", Noul::new("Ready?")?.named("ready")?)
            .await
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected provider error"))?;
        let ProviderError::ProviderResponse(response) = &error else {
            anyhow::bail!("wrong error: {error}");
        };
        ensure!(response.status.map(|s| s.as_u16()) == Some(status));
        ensure!(response.provider_request_id.as_deref() == Some("error-request"));
        ensure!(response.body == r#"{"error":"overloaded"}"#);
        ensure!(error.report().retryable);
        mock.assert_async().await;
    }
    Ok(())
}

#[tokio::test]
async fn rejects_missing_and_extra_response_ids() -> anyhow::Result<()> {
    for answers in [
        json!({}),
        json!({"ready":{"type":"noul","noul":0.8},"extra":{"type":"noul","noul":0.5}}),
    ] {
        let server = httpmock::MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/v1/systemone");
                then.status(200)
                    .header("content-type", "application/json")
                    .json_body(json!({"model":"test", "answers": answers}));
            })
            .await;
        let client = Jev::new("test-token")
            .with_endpoint(server.url("/v1/systemone"))
            .bind(rig_reqwest::ReqwestClient::default());
        ensure!(
            matches!(client.evaluate(&"state", Noul::new("Ready?")?.named("ready")?).await,
            Err(ProviderError::Response(message)) if message == "response question IDs differ from request")
        );
        mock.assert_async().await;
    }
    Ok(())
}

#[test]
fn dynamic_query_rejects_invalid_runtime_definitions() -> anyhow::Result<()> {
    ensure!(DynamicQuery::new(BTreeMap::new()).is_err());
    for (id, definition) in [
        ("", json!({"type":"noul","instructions":"Ready?"})),
        (
            "score",
            json!({"type":"score","instructions":"Rate", "criteria":[]}),
        ),
        (
            "score",
            json!({"type":"score","instructions":"Rate", "criteria":["one"]}),
        ),
        (
            "choice",
            json!({"type":"choice","instructions":"Choose", "criteria":{"":"bad", "a":"good"}}),
        ),
        (
            "choice",
            json!({"type":"choice","instructions":"Choose", "criteria":{"a":true, "b":"good"}}),
        ),
        ("gate", json!({"type":"noul","instructions":3})),
        (
            "gate",
            json!({"type":"noul","instructions":"Ready?", "criteria":{"yes":"yes", "no":"no"}}),
        ),
    ] {
        let definition = serde_json::from_value(definition)?;
        ensure!(matches!(
            DynamicQuery::new(BTreeMap::from([(id.into(), definition)])),
            Err(ProviderError::Request(_))
        ));
    }
    Ok(())
}

/// Synthetic contradictions and rounding boundaries cannot be captured as valid provider traffic.
#[test]
fn validates_score_against_distribution() -> anyhow::Result<()> {
    let dynamic = DynamicScore::new("Rate", ["low", "medium", "high", "highest"])?;
    let typed = Score::new(
        "Rate",
        [(0, "low"), (1, "medium"), (2, "high"), (3, "highest")],
    )?;
    for (weights, score, valid) in [
        ([0.0, 0.0, 1.0, 0.0], 0.0, false),
        ([0.0, 0.0, 1.0, 0.0], 2.0, true),
        ([0.01, 0.23, 0.67, 0.08], 1.83, true),
        ([0.02, 0.23, 0.67, 0.09], 1.82 / 1.01, true),
        ([0.0, 0.0, 0.0, 0.99], 3.0, true),
        ([0.0, 0.0, 0.0, 1.0], 2.9, false),
        ([0.0, 0.14, 0.76, 0.1], 1.95, true),
        ([0.01, 0.23, 0.67, 0.08], 2.0, false),
        ([0.1234, 0.2345, 0.3456, 0.2965], 1.82, true),
        ([0.1234, 0.2345, 0.3456, 0.2965], 1.84, false),
    ] {
        let answer = types::Answer::Score {
            score,
            probabilities: weights
                .into_iter()
                .enumerate()
                .map(|(index, weight)| (index.to_string(), weight))
                .collect(),
            legend: ["low", "medium", "high", "highest"]
                .into_iter()
                .enumerate()
                .map(|(index, description)| (index.to_string(), json!(description)))
                .collect(),
            confidence: 0.5,
        };
        ensure!(
            dynamic.decode(answer.clone()).is_ok() == valid,
            "dynamic: {weights:?}, {score}"
        );
        ensure!(
            typed.decode(answer).is_ok() == valid,
            "typed: {weights:?}, {score}"
        );
    }
    Ok(())
}

/// A request the endpoint URL cannot carry fails as a request that could not
/// be built.
#[test]
fn an_unbuildable_request_is_a_request_failure() -> anyhow::Result<()> {
    use rig_core::error::{ErrorKind, ProviderError};
    use rig_core::wire::{Mode, Wire};

    let request = serde_json::from_value(json!({"state": 1, "questions": {}}))?;
    let Err(error) = Jev::new("token")
        .with_endpoint("http://bad host")
        .encode(request, Mode::Unary)
    else {
        anyhow::bail!("the request must not build");
    };
    let report = ProviderError::from(error).report();
    ensure!(report.kind == ErrorKind::Request);
    ensure!(!report.retryable);
    ensure!(report.message == "RequestError: invalid uri character");
    Ok(())
}

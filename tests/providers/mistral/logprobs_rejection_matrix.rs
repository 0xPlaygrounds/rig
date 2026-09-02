//! Live Mistral logprobs rejection controls.
//!
//! Mistral's current API reference exposes no chat-completions logprobs
//! contract, and both accessible chat families reject the OpenAI-compatible
//! request field with error code 3051 (`Logprobs are not enabled for this
//! model`). The 24-cell terminal-metadata matrix replaces the unsupported
//! successful-logprobs matrix; these controls preserve the live evidence for
//! that substitution and are not counted toward its 24 cells.
//!
//! The finite recordable space is 2 transports × 2 accessible model families
//! = 4 cells. There are no pruned or unit-only cells: `top_logprobs` has no
//! additional reachable state once `logprobs: true` is rejected before
//! generation, so every cell fixes it at `2` and asserts the same typed error.
//! Each explicit test maps to
//! `tests/cassettes/mistral/logprobs_rejection_matrix/<test-name>.yaml`.
//!
//! | dimension | values |
//! |---|---|
//! | transport | blocking, streaming request |
//! | model | `mistral-small-latest`, `ministral-3b-latest` |
//! | request | `logprobs: true`, `top_logprobs: 2` |
//!
//! | recorded cells | exact fixture set |
//! |---|---|
//! | all 4 | `tests/cassettes/mistral/logprobs_rejection_matrix/{blocking,streaming}_{mistral_small,ministral_3b}.yaml` |

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use futures::StreamExt as _;
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::mistral;
use serde_json::{Value, json};

use super::support::with_mistral_logprobs_rejection_cassette_result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transport {
    Blocking,
    Streaming,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    MistralSmall,
    Ministral3b,
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    transport: Transport,
    model: Model,
}

type SharedError = Arc<Mutex<Option<String>>>;

fn model_name(model: Model) -> &'static str {
    match model {
        Model::MistralSmall => "mistral-small-latest",
        Model::Ministral3b => "ministral-3b-latest",
    }
}

async fn run_cell(client: mistral::Client, cell: Cell, observed: SharedError) -> Result<()> {
    let model = client.completion_model(model_name(cell.model));
    let request = model
        .completion_request("Reply with exactly: cobalt")
        .additional_params(json!({ "logprobs": true, "top_logprobs": 2 }))
        .max_tokens(8)
        .build();

    // The blocking path fails with the provider's `CompletionError`; a stream
    // fails in-band with the `ErrorReport` it was mapped to. Both display the
    // preserved Mistral body, which is what the matrix asserts on.
    let error = match cell.transport {
        Transport::Blocking => match model.raw_completion(request).await {
            Ok(_) => bail!("Mistral unexpectedly accepted blocking logprobs"),
            Err(error) => error.to_string(),
        },
        Transport::Streaming => match model.stream(request).await {
            Err(error) => error.to_string(),
            Ok(mut stream) => loop {
                match stream.next().await {
                    Some(Err(error)) => break error.to_string(),
                    Some(Ok(_)) => continue,
                    None => bail!("Mistral stream ended without rejecting logprobs"),
                }
            },
        },
    };

    *observed.lock().expect("error observation mutex poisoned") = Some(error);
    Ok(())
}

fn recorded_request(scenario: &str) -> Value {
    crate::cassettes::recorded_json_request("mistral", scenario)
}

fn recorded_response(scenario: &str) -> Value {
    crate::cassettes::recorded_json_response("mistral", scenario)
}

fn assert_cell(scenario: &str, cell: Cell, observed: SharedError) {
    let request = recorded_request(scenario);
    assert_eq!(request["model"], model_name(cell.model), "{scenario}");
    assert_eq!(request["logprobs"], true, "{scenario}: request premise");
    assert_eq!(
        request["top_logprobs"], 2,
        "{scenario}: top-logprobs premise"
    );
    assert_eq!(request["max_tokens"], 8, "{scenario}: bounded probe");
    assert_eq!(
        request["stream"].as_bool().unwrap_or(false),
        cell.transport == Transport::Streaming,
        "{scenario}: transport"
    );

    let response = recorded_response(scenario);
    assert_eq!(response["object"], "error", "{scenario}: error envelope");
    assert_eq!(response["code"], "3051", "{scenario}: rejection code");
    let message = response["message"]
        .as_str()
        .context("recorded Mistral error should have a message")
        .expect("recorded Mistral error message");
    assert!(
        message.contains("Logprobs are not enabled"),
        "{scenario}: rejection message: {message}"
    );

    let observed = observed
        .lock()
        .expect("error observation mutex poisoned")
        .take()
        .expect("test body should capture the provider error");
    assert!(observed.contains("3051"), "{scenario}: {observed}");
    assert!(
        observed.contains("Logprobs are not enabled"),
        "{scenario}: {observed}"
    );
}

async fn execute(scenario: &'static str, cell: Cell, observed: SharedError) {
    assert_cell(scenario, cell, observed);
}

#[tokio::test]
async fn blocking_mistral_small() -> Result<()> {
    const S: &str = "logprobs_rejection_matrix/blocking_mistral_small";
    let cell = Cell {
        transport: Transport::Blocking,
        model: Model::MistralSmall,
    };
    let observed = SharedError::default();
    let capture = Arc::clone(&observed);
    with_mistral_logprobs_rejection_cassette_result(
        "logprobs_rejection_matrix/blocking_mistral_small",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    execute(S, cell, observed).await;
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b() -> Result<()> {
    const S: &str = "logprobs_rejection_matrix/blocking_ministral_3b";
    let cell = Cell {
        transport: Transport::Blocking,
        model: Model::Ministral3b,
    };
    let observed = SharedError::default();
    let capture = Arc::clone(&observed);
    with_mistral_logprobs_rejection_cassette_result(
        "logprobs_rejection_matrix/blocking_ministral_3b",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    execute(S, cell, observed).await;
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small() -> Result<()> {
    const S: &str = "logprobs_rejection_matrix/streaming_mistral_small";
    let cell = Cell {
        transport: Transport::Streaming,
        model: Model::MistralSmall,
    };
    let observed = SharedError::default();
    let capture = Arc::clone(&observed);
    with_mistral_logprobs_rejection_cassette_result(
        "logprobs_rejection_matrix/streaming_mistral_small",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    execute(S, cell, observed).await;
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b() -> Result<()> {
    const S: &str = "logprobs_rejection_matrix/streaming_ministral_3b";
    let cell = Cell {
        transport: Transport::Streaming,
        model: Model::Ministral3b,
    };
    let observed = SharedError::default();
    let capture = Arc::clone(&observed);
    with_mistral_logprobs_rejection_cassette_result(
        "logprobs_rejection_matrix/streaming_ministral_3b",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    execute(S, cell, observed).await;
    Ok(())
}

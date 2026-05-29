//! Copilot agent completion smoke test.

use rig::client::{CompletionClient, ModelListingClient};
use rig::completion::Prompt;

use crate::copilot::{LIVE_MODEL, live_client};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn completion_smoke() {
    let agent = live_client()
        .agent(LIVE_MODEL)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}

/// Command to run
/// cargo test -p rig --test copilot all_models_completion_smoke -- --ignored --nocapture
#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn all_models_completion_smoke() {
    let client = live_client();

    let models = client
        .list_models()
        .await
        .expect("listing Copilot models should succeed");

    println!(
        "Found {} models; testing completion on each...",
        models.len()
    );

    assert!(
        !models.is_empty(),
        "expected Copilot to return at least one model"
    );

    let mut succeeded = Vec::new();
    let mut failed = Vec::new();

    for model in models.iter() {
        println!("Testing {:#?}...", model.id);
        let agent = client
            .agent(model.id.as_str())
            .preamble(BASIC_PREAMBLE)
            .build();

        match agent.prompt(BASIC_PROMPT).await {
            Ok(response) if !response.is_empty() => succeeded.push(model.id.clone()),
            Ok(_) => failed.push(format!("{}: empty response", model.id)),
            Err(e) => failed.push(format!("{}: {e}", model.id)),
        }
    }

    println!("\n=== Completion results ({} models) ===", models.len());
    println!("Succeeded ({}):", succeeded.len());
    for id in &succeeded {
        println!("  + {id}");
    }
    println!("Failed ({}):", failed.len());
    for entry in &failed {
        println!("  - {entry}");
    }
}

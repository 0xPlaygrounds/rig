//! Submits two small requests as one batch through a provider's batch API,
//! polls it until it finishes, and prints each request's outcome. A batch
//! still running after `BATCH_MAX_WAIT_SECS` (default 1200) is cancelled.
//!
//! ```sh
//! cargo run -p batch_completions -- anthropic   # or openai, gemini
//! ```
//!
//! Needs the provider's API key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or
//! `GEMINI_API_KEY`).

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use rig::completion::CompletionRequest;
use rig::completion::batch::{BatchItem, BatchOutcome, Batching};
use rig::driver::Model;
use rig::providers::{anthropic, gemini, openai};

const POLL: Duration = Duration::from_secs(15);

#[tokio::main]
async fn main() -> Result<()> {
    let max_wait = std::env::var("BATCH_MAX_WAIT_SECS")
        .ok()
        .and_then(|secs| secs.parse().ok())
        .map_or(Duration::from_secs(1200), Duration::from_secs);
    match std::env::args().nth(1).as_deref() {
        Some("anthropic") => {
            let client = anthropic::Anthropic::from_env()?;
            run(client.completion(anthropic::CLAUDE_HAIKU_4_5), max_wait).await
        }
        Some("openai") => {
            let client = openai::OpenAI::from_env()?;
            run(client.completion(openai::GPT_4_1_NANO), max_wait).await
        }
        Some("gemini") => {
            let client = gemini::Gemini::from_env()?;
            let model = gemini::completion::GEMINI_2_5_FLASH;
            run(client.completion(model), max_wait).await
        }
        _ => bail!("usage: batch_completions <anthropic|openai|gemini>"),
    }
}

/// The same code drives every provider's batch API.
async fn run<W: Batching>(model: Model<W>, max_wait: Duration) -> Result<()> {
    let ask = |question: &str| CompletionRequest::new(question).max_tokens(512);
    let batch = model
        .submit_batch(vec![
            BatchItem::new("capital", ask("The capital of France, in one word?")),
            BatchItem::new("sum", ask("What is 2 + 3? Answer with the number only.")),
        ])
        .await
        .context("submitting the batch")?;
    println!("submitted batch {}", batch.id);

    let started = Instant::now();
    loop {
        let state = model.batch_status(&batch.id).await?;
        println!(
            "{:>4}s  {:?}  pending {} succeeded {} failed {}",
            started.elapsed().as_secs(),
            state.status,
            state.counts.pending,
            state.counts.succeeded,
            state.counts.failed
        );
        if state.status.is_finished() {
            break;
        }
        if started.elapsed() >= max_wait {
            let state = model.cancel_batch(&batch.id).await?;
            println!("waited too long: cancel requested, now {:?}", state.status);
            return Ok(());
        }
        tokio::time::sleep(POLL).await;
    }

    for (id, outcome) in model.batch_results(&batch).await? {
        match outcome {
            BatchOutcome::Succeeded(response) => {
                let text = response
                    .choice
                    .iter()
                    .filter_map(|content| match content {
                        rig::message::AssistantContent::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                let cost = response.usage.cost.map(|cost| cost.total);
                println!("{id}: {text:?} (cost {cost:?} USD)");
            }
            BatchOutcome::Failed(error) => println!("{id}: failed: {error}"),
            BatchOutcome::Canceled => println!("{id}: canceled"),
            BatchOutcome::Expired => println!("{id}: expired"),
        }
    }
    Ok(())
}

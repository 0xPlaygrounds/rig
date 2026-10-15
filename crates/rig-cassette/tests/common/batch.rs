//! The batch-completion cell every provider with a batch API records: two
//! requests submitted as one batch, polled on the session's clock until the
//! batch ends, and each reply read back through the live decoder.

use std::time::Duration;

use rig::completion::CompletionRequest;
use rig::completion::batch::{BatchItem, BatchOutcome, BatchStatus, Batching};
use rig::driver::Model;
use rig::message::AssistantContent;

use crate::cassettes::CassetteClock;

/// How long a recording waits between polls, and how many polls it makes
/// before it gives up.
const POLL: Duration = Duration::from_secs(30);
const POLLS: usize = 40;

/// Submit two questions as one batch to `model`, wait for it to end, and
/// check each answer.
pub async fn round_trip<W: Batching>(model: &Model<W>, clock: &CassetteClock) {
    let ask = |question: &str| CompletionRequest::new(question).max_tokens(512);
    let batch = model
        .submit_batch(vec![
            BatchItem::new("capital", ask("The capital of France, in one word?")),
            BatchItem::new("sum", ask("What is 2 + 3? Answer with the number only.")),
        ])
        .await
        .expect("the provider accepts the batch");
    let mut state = model
        .batch_status(&batch.id)
        .await
        .expect("the batch reads back");
    for _ in 0..POLLS {
        if state.status.is_finished() {
            break;
        }
        clock.pause(POLL).await;
        state = model
            .batch_status(&batch.id)
            .await
            .expect("the batch reads back");
    }
    assert_eq!(state.status, BatchStatus::Completed, "{state:?}");
    assert_eq!((state.counts.succeeded, state.counts.failed), (2, 0));

    let results = model
        .batch_results(&batch)
        .await
        .expect("an ended batch has results");
    let answer = |id: &str| match results.get(id) {
        Some(BatchOutcome::Succeeded(response)) => {
            assert!(
                response.usage.cost.is_some(),
                "a batch reply is priced: {response:?}"
            );
            response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<String>()
        }
        other => panic!("`{id}` should succeed: {other:?}"),
    };
    assert!(answer("capital").contains("Paris"));
    assert!(answer("sum").contains('5'));
}

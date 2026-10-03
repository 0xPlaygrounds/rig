//! Raw response capture on Cohere's streamed Chat Completions route.
//!
//! A streamed reply's `raw` is the terminal record the decoder assembled
//! from the stream; its usage is the one usage chunk the stream carried.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_terminal_reproduces_the_usage_chunk` | terminal record | the terminal reproduces the recorded usage chunk's id, model and usage | recorded |

use rig::completion::CompletionRequest;

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};
use crate::raw_capture::{assert_no_request_id, capture_text_and_terminal, chat};
use crate::support::Observed;

const PROVIDER: &str = "cohere";

#[tokio::test]
async fn stream_terminal_reproduces_the_usage_chunk() {
    let scenario = "raw_stream_capture_matrix/stream_terminal_reproduces_the_usage_chunk";
    let sink = Observed::default();
    let parked = sink.clone();
    with_cohere_cassette(
        "raw_stream_capture_matrix/stream_terminal_reproduces_the_usage_chunk",
        |client| async move {
            capture_text_and_terminal(
                client.completion(CASSETTE_MODEL),
                CompletionRequest::new(
                    "Reply with exactly this one word and nothing else: streamed",
                )
                .temperature(0.0)
                .max_tokens(16),
                parked,
            )
            .await
            .expect("the recorded stream replays");
        },
    )
    .await;
    let (text, terminal) = sink.take();
    assert!(!text.trim().is_empty(), "the stream carried text");
    let frame = chat::recorded_sole_usage_frame(PROVIDER, scenario);
    chat::assert_terminal_reproduces_frame(&terminal, PROVIDER, &frame, scenario);
    assert_no_request_id(terminal.provider_request_id.as_deref(), PROVIDER);
}

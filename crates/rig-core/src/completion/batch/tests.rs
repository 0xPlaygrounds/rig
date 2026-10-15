use super::*;
use crate::providers::anthropic::{AnthropicConfig, CLAUDE_HAIKU_4_5};
use crate::test_utils::SequencedHttpClient;

/// An id valid for one provider is valid for every provider, and a refused
/// batch sends nothing.
#[tokio::test]
async fn a_batch_needs_items_with_distinct_portable_ids() {
    let http = SequencedHttpClient::new([]);
    let model = AnthropicConfig::new("key")
        .connect(http.clone())
        .completion(CLAUDE_HAIKU_4_5);
    let item = |id: &str| BatchItem::new(id, CompletionRequest::new("Hi"));
    for items in [
        Vec::new(),
        vec![item("")],
        vec![item("has space")],
        vec![item(&"x".repeat(MAX_ID_LEN + 1))],
        vec![item("same"), item("same")],
    ] {
        assert!(model.submit_batch(items).await.is_err());
    }
    assert!(http.requests().is_empty());
}

#[test]
fn a_batch_id_cannot_retarget_its_request() {
    assert!(path_id("msgbatch_01-x").is_ok());
    for id in ["", "a/b", "a?b", "a#b", "../a"] {
        assert!(path_id(id).is_err(), "{id}");
    }
}

#[test]
fn a_finished_batch_stops_changing() {
    use BatchStatus::*;
    let finished: Vec<bool> = [
        Pending, Running, Cancelling, Completed, Cancelled, Expired, Failed,
    ]
    .map(BatchStatus::is_finished)
    .to_vec();
    assert_eq!(finished, [false, false, false, true, true, true, true]);
}

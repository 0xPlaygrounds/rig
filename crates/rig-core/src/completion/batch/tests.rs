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

/// A tier that discounts some kinds of token and not others prices each
/// part at its own share.
#[test]
fn a_batch_price_applies_per_kind_of_token() {
    use crate::completion::Cost;
    let price = BatchPrice {
        input: 0.5,
        output: 0.5,
        cache_read: 1.0,
        cache_write: 1.0,
    };
    let cost = Cost::from_parts(2.0, 4.0, 1.0, 0.0).at(price);
    assert_eq!(
        (cost.input, cost.output, cost.cache_read, cost.total),
        (Some(1.0), Some(2.0), Some(1.0), 4.0)
    );
    assert_eq!(Cost::from_total(3.0).at(BatchPrice::HALF).total, 1.5);
    let partly_known = Cost::from_total(1.0).input(0.25).at(BatchPrice::HALF);
    assert_eq!(partly_known.total, 0.5);
}

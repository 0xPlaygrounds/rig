use super::*;

/// Every decision type a host may cache in serializable state round-trips.
#[test]
fn decision_types_round_trip_through_serde() {
    for action in [
        InvalidToolCallAction::fail(),
        InvalidToolCallAction::retry("try add"),
        InvalidToolCallAction::repair("add"),
        InvalidToolCallAction::skip("nope"),
        InvalidToolCallAction::stop("done"),
    ] {
        let json = serde_json::to_string(&action).expect("serialize action");
        assert_eq!(
            serde_json::from_str::<InvalidToolCallAction>(&json).expect("deserialize action"),
            action
        );
    }

    let retry = RetryRequest::Feedback("again".to_string());
    let json = serde_json::to_string(&retry).expect("serialize retry");
    assert_eq!(
        serde_json::from_str::<RetryRequest>(&json).expect("deserialize retry"),
        retry
    );
}

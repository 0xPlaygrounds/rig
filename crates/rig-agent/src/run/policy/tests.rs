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

/// The malformed-arguments reason carries the parser's error, or names the
/// JSON value that is not an object.
#[test]
fn malformed_arguments_name_what_the_parser_rejected() {
    let InvalidToolCallReason::MalformedArguments { error } =
        InvalidToolCallReason::malformed_arguments("{\"x\":")
    else {
        panic!("a malformed-arguments reason");
    };
    assert!(error.contains("line 1 column"), "{error}");
    for (raw, kind) in [
        ("null", "null"),
        ("true", "a boolean"),
        ("1", "a number"),
        ("\"x\"", "a string"),
        ("[1]", "an array"),
    ] {
        assert_eq!(
            InvalidToolCallReason::malformed_arguments(raw),
            InvalidToolCallReason::MalformedArguments {
                error: format!("expected a JSON object, found {kind}")
            }
        );
    }
    assert_eq!(json_kind(&serde_json::json!({})), "an object");
}

#[test]
fn reasons_round_trip_through_serde() {
    for reason in [
        InvalidToolCallReason::UnknownTool,
        InvalidToolCallReason::DisallowedByToolChoice,
        InvalidToolCallReason::malformed_arguments("{"),
    ] {
        let json = serde_json::to_value(&reason).expect("serialize reason");
        assert!(json["reason"].is_string(), "{json}");
        assert_eq!(
            serde_json::from_value::<InvalidToolCallReason>(json).expect("deserialize reason"),
            reason
        );
    }
}

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

fn tool_names(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

#[test]
fn resolve_specific_rejects_missing_tools() {
    let executable = tool_names(&["add"]);
    let choice = ToolChoice::Specific {
        function_names: vec![rig_core::message::ToolName::new("missing").expect("tool name")],
    };

    let err = TurnPolicy::resolve(executable, Some(choice), None, None)
        .expect_err("missing specific tool should fail before provider request");

    assert!(matches!(
        err,
        PrepareError::Request(err)
            if err.to_string().contains("missing")
                && err.to_string().contains("add")
    ));
}

#[test]
fn resolve_specific_rejects_empty_names() {
    let executable = tool_names(&["add"]);
    let choice = ToolChoice::Specific {
        function_names: vec![],
    };

    let err = TurnPolicy::resolve(executable, Some(choice), None, None)
        .expect_err("empty specific tool choice should fail before provider request");

    assert!(matches!(
        err,
        PrepareError::Request(err)
            if err.to_string().contains("requires at least one function name")
    ));
}

#[test]
fn required_with_no_advertised_tool_is_local_error() {
    let empty = tool_names(&[]);
    let err = TurnPolicy::resolve(empty, Some(ToolChoice::Required), None, None)
        .expect_err("Required with no advertised tool must fail locally");
    assert!(matches!(
        err,
        PrepareError::Request(err) if err.to_string().contains("Required")
    ));
}

#[test]
fn required_with_active_tools_filter_names_the_filter_in_the_error() {
    let empty = tool_names(&[]);
    let err = TurnPolicy::resolve(
        empty,
        Some(ToolChoice::Required),
        None,
        Some(&tool_names(&["add"])),
    )
    .expect_err("Required after active_tools filtered everything must fail locally");
    let msg = err.to_string();
    assert!(
        msg.contains("active_tools"),
        "error should name active_tools: {msg}"
    );
    assert!(
        msg.contains("RequestPatch"),
        "error should suggest RequestPatch: {msg}"
    );
}

#[test]
fn specific_typo_is_not_blamed_on_active_tools() {
    // Specific names a tool that never existed (a typo), even though an
    // active_tools filter was applied. The error must NOT blame active_tools,
    // because the filter never had that tool to drop.
    let executable = tool_names(&["add"]);
    let choice = ToolChoice::Specific {
        function_names: vec![rig_core::message::ToolName::new("nonexistent").expect("tool name")],
    };
    let err = TurnPolicy::resolve(executable, Some(choice), None, Some(&tool_names(&["add"])))
        .expect_err("Specific naming a non-existent tool must fail locally");
    let msg = err.to_string();
    assert!(msg.contains("nonexistent"), "error names the typo: {msg}");
    assert!(
        !msg.contains("active_tools"),
        "a plain typo must not be blamed on active_tools: {msg}"
    );
}

#[test]
fn the_allowed_set_follows_the_choice() {
    let none = TurnPolicy::new(tool_names(&["add"]), Some(ToolChoice::None), None).expect("none");
    assert!(none.forbids_calls());
    assert!(!none.allows("add"));

    let auto = TurnPolicy::new(tool_names(&["add"]), None, Some("final".into())).expect("auto");
    assert!(!auto.forbids_calls());
    assert!(auto.allows("add") && auto.allows("final"));
    assert!(!auto.executable().contains("final"));
}

/// A persisted policy stores no allowed set: it is re-derived from the
/// choice, and a choice that cannot be honored is refused on load.
#[test]
fn a_persisted_policy_re_derives_its_allowed_set() {
    let choice = ToolChoice::Specific {
        function_names: vec![rig_core::message::ToolName::new("add").expect("tool name")],
    };
    let policy = TurnPolicy::new(tool_names(&["add", "sub"]), Some(choice), None).expect("policy");
    let json = serde_json::to_value(&policy).expect("serialize");
    assert!(json.get("allowed").is_none(), "{json}");
    let restored: TurnPolicy = serde_json::from_value(json.clone()).expect("deserialize");
    assert_eq!(restored, policy);
    assert_eq!(restored.allowed(), &tool_names(&["add"]));

    let mut stale = json;
    stale["executable"] = serde_json::json!(["sub"]);
    let error = serde_json::from_value::<TurnPolicy>(stale).expect_err("unadvertised choice");
    assert!(error.to_string().contains("add"), "{error}");
}

#[test]
fn the_context_reports_the_policy_it_was_built_from() {
    let policy = TurnPolicy::new(tool_names(&["add"]), Some(ToolChoice::None), None).expect("none");
    let call = rig_core::message::ToolCall::from_wire(
        "c1",
        rig_core::message::ToolFunction::new(
            rig_core::message::ToolName::new("add").expect("tool name"),
            serde_json::json!({}),
        ),
    );
    let context =
        policy.invalid_call_context(&call, None, Vec::new(), false, policy.name_reason(&call));
    assert_eq!(context.tool_choice, Some(ToolChoice::None));
    assert_eq!(context.available_tools, vec!["add".to_string()]);
    assert!(context.allowed_tools.is_empty());
    assert_eq!(
        context.reason,
        InvalidToolCallReason::DisallowedByToolChoice
    );
}

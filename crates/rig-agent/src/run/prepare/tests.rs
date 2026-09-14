use super::*;

/// A prepared request produced by `prepare_request` survives a JSON round
/// trip — a host caching one in serializable state (a saved world) can
/// restore it losslessly.
#[test]
fn prepared_request_round_trips_through_serde() {
    let spec = RunSpec {
        preamble: Some("be brief".to_string()),
        temperature: Some(0.2),
        ..RunSpec::default()
    };
    let prepared = prepare_request(
        &spec,
        &ProviderCapabilities::default(),
        &[Message::user("hi")],
        vec![ToolDefinition {
            name: "add".to_string(),
            description: "adds".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }],
        None,
        None,
    )
    .expect("prepare");
    let json = serde_json::to_string(&prepared).expect("serialize");
    let restored: PreparedRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored, prepared);
}

fn tool_names(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

#[test]
fn allowed_tool_names_defaults_to_all_executable_tools() {
    let executable = tool_names(&["add", "subtract"]);

    assert_eq!(
        allowed_tool_names_for_choice(&executable, None, None, None).unwrap(),
        executable
    );
}

#[test]
fn allowed_tool_names_auto_and_required_allow_all_executable_tools() {
    let executable = tool_names(&["add", "subtract"]);

    assert_eq!(
        allowed_tool_names_for_choice(&executable, Some(&ToolChoice::Auto), None, None).unwrap(),
        executable
    );
    assert_eq!(
        allowed_tool_names_for_choice(&executable, Some(&ToolChoice::Required), None, None)
            .unwrap(),
        executable
    );
}

#[test]
fn allowed_tool_names_none_allows_no_tools() {
    let executable = tool_names(&["add", "subtract"]);

    assert!(
        allowed_tool_names_for_choice(&executable, Some(&ToolChoice::None), None, None)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn allowed_tool_names_specific_allows_requested_executable_tools() {
    let executable = tool_names(&["add", "subtract"]);
    let choice = ToolChoice::Specific {
        function_names: vec!["add".to_string()],
    };

    assert_eq!(
        allowed_tool_names_for_choice(&executable, Some(&choice), None, None).unwrap(),
        tool_names(&["add"])
    );
}

#[test]
fn allowed_tool_names_specific_rejects_missing_tools() {
    let executable = tool_names(&["add"]);
    let choice = ToolChoice::Specific {
        function_names: vec!["missing".to_string()],
    };

    let err = allowed_tool_names_for_choice(&executable, Some(&choice), None, None)
        .expect_err("missing specific tool should fail before provider request");

    assert!(matches!(
        err,
        PrepareError::Request(err)
            if err.to_string().contains("missing")
                && err.to_string().contains("add")
    ));
}

#[test]
fn allowed_tool_names_specific_rejects_empty_names() {
    let executable = tool_names(&["add"]);
    let choice = ToolChoice::Specific {
        function_names: vec![],
    };

    let err = allowed_tool_names_for_choice(&executable, Some(&choice), None, None)
        .expect_err("empty specific tool choice should fail before provider request");

    assert!(matches!(
        err,
        PrepareError::Request(err)
            if err.to_string().contains("requires at least one function name")
    ));
}

#[test]
fn output_tool_callable_honors_specific_naming_the_output_tool() {
    // Auto / Required / no explicit choice all permit the output-tool call.
    assert!(output_tool_callable(None, "final_result"));
    assert!(output_tool_callable(
        Some(&ToolChoice::Auto),
        "final_result"
    ));
    assert!(output_tool_callable(
        Some(&ToolChoice::Required),
        "final_result"
    ));
    // A `Specific` set that NAMES the output tool can call it — the case the
    // pinned Tool-mode stall warning must not flag (it is accepted by
    // `allowed_tool_names_for_choice`, which advertises the output tool).
    assert!(output_tool_callable(
        Some(&ToolChoice::Specific {
            function_names: vec!["final_result".to_string()],
        }),
        "final_result",
    ));
    // A `Specific` set that omits it — or `ToolChoice::None` — genuinely cannot
    // finalize a pinned Tool-mode turn, so the warning should still fire there.
    assert!(!output_tool_callable(
        Some(&ToolChoice::Specific {
            function_names: vec!["search".to_string()],
        }),
        "final_result",
    ));
    assert!(!output_tool_callable(
        Some(&ToolChoice::None),
        "final_result"
    ));
}

#[test]
fn required_with_no_advertised_tool_is_local_error() {
    let empty = tool_names(&[]);
    let err = allowed_tool_names_for_choice(&empty, Some(&ToolChoice::Required), None, None)
        .expect_err("Required with no advertised tool must fail locally");
    assert!(matches!(
        err,
        PrepareError::Request(err) if err.to_string().contains("Required")
    ));
}

#[test]
fn required_with_only_the_output_tool_is_allowed() {
    // Structured-output Tool mode with no real tools: the model can still be
    // forced to call the synthetic output tool, so Required is valid.
    let empty = tool_names(&[]);
    let allowed = allowed_tool_names_for_choice(
        &empty,
        Some(&ToolChoice::Required),
        Some("final_result"),
        None,
    )
    .expect("Required is satisfiable by the output tool");
    // The output tool is added to the allowed set by the caller, so the
    // executable-derived allowed set is empty here.
    assert!(allowed.is_empty());
}

#[test]
fn required_with_active_tools_filter_names_the_filter_in_the_error() {
    let empty = tool_names(&[]);
    let err = allowed_tool_names_for_choice(
        &empty,
        Some(&ToolChoice::Required),
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
fn specific_naming_a_filtered_out_tool_is_a_local_error_with_hint() {
    // active_tools narrowed the advertised set to {add}; Specific still names
    // the now-filtered-out `subtract`.
    let executable = tool_names(&["add"]);
    let choice = ToolChoice::Specific {
        function_names: vec!["subtract".to_string()],
    };
    let err = allowed_tool_names_for_choice(
        &executable,
        Some(&choice),
        None,
        Some(&tool_names(&["add", "subtract"])),
    )
    .expect_err("Specific naming a filtered-out tool must fail locally");
    let msg = err.to_string();
    assert!(
        msg.contains("subtract"),
        "error should name the missing tool: {msg}"
    );
    assert!(
        msg.contains("active_tools"),
        "error should name active_tools: {msg}"
    );
}

#[test]
fn specific_may_name_the_output_tool() {
    // The effective advertised set includes the synthetic output tool.
    let empty = tool_names(&[]);
    let choice = ToolChoice::Specific {
        function_names: vec!["final_result".to_string()],
    };
    let allowed = allowed_tool_names_for_choice(&empty, Some(&choice), Some("final_result"), None)
        .expect("Specific naming the output tool is valid");
    assert_eq!(allowed, tool_names(&["final_result"]));
}

#[test]
fn specific_typo_is_not_blamed_on_active_tools() {
    // Specific names a tool that never existed (a typo), even though an
    // active_tools filter was applied. The error must NOT blame active_tools,
    // because the filter never had that tool to drop.
    let executable = tool_names(&["add"]);
    let choice = ToolChoice::Specific {
        function_names: vec!["nonexistent".to_string()],
    };
    let err = allowed_tool_names_for_choice(
        &executable,
        Some(&choice),
        None,
        Some(&tool_names(&["add"])),
    )
    .expect_err("Specific naming a non-existent tool must fail locally");
    let msg = err.to_string();
    assert!(msg.contains("nonexistent"), "error names the typo: {msg}");
    assert!(
        !msg.contains("active_tools"),
        "a plain typo must not be blamed on active_tools: {msg}"
    );
}

#[test]
fn resolve_output_mode_without_schema_is_always_native() {
    // No schema => nothing to enforce, regardless of the requested mode or tools.
    for requested in [
        OutputMode::Auto,
        OutputMode::Tool,
        OutputMode::Native,
        OutputMode::Prompted,
    ] {
        assert_eq!(
            resolve_output_mode(false, true, true, false, &requested),
            OutputMode::Native,
            "no schema should force Native for {requested:?}"
        );
        assert_eq!(
            resolve_output_mode(false, false, true, false, &requested),
            OutputMode::Native,
        );
    }
}

#[test]
fn resolve_output_mode_auto_picks_tool_only_when_tools_present() {
    // This is the #1928 fix: with tools on a provider that does NOT compose
    // native output with tools, the schema must not be a native `format`
    // constraint on every turn, so Auto routes to Tool.
    assert_eq!(
        resolve_output_mode(true, true, true, false, &OutputMode::Auto),
        OutputMode::Tool,
    );
    // No tools => native structured output is safe and preferred.
    assert_eq!(
        resolve_output_mode(true, false, true, false, &OutputMode::Auto),
        OutputMode::Native,
    );
}

#[test]
fn resolve_output_mode_auto_keeps_native_when_provider_composes() {
    // On providers that compose native structured output with tools (OpenAI,
    // Anthropic), Auto keeps guaranteed native output even with tools present.
    assert_eq!(
        resolve_output_mode(true, true, true, true, &OutputMode::Auto),
        OutputMode::Native,
    );
}

#[test]
fn resolve_output_mode_honors_explicit_choice_with_schema() {
    for (requested, expected) in [
        (OutputMode::Tool, OutputMode::Tool),
        (OutputMode::Native, OutputMode::Native),
        (OutputMode::Prompted, OutputMode::Prompted),
    ] {
        // Explicit modes are honored regardless of tools or provider support.
        assert_eq!(
            resolve_output_mode(true, true, true, false, &requested),
            expected
        );
        assert_eq!(
            resolve_output_mode(true, false, true, true, &requested),
            expected
        );
    }
}

#[test]
fn resolve_output_mode_degrades_to_native_when_output_tool_not_callable() {
    // Tool mode finalizes via the output-tool call; when the tool choice
    // forbids it (None / Specific), structured output must still be enforced
    // via Native rather than silently dropped (#1928 regression guard).
    assert_eq!(
        resolve_output_mode(true, true, false, false, &OutputMode::Auto),
        OutputMode::Native,
    );
    assert_eq!(
        resolve_output_mode(true, true, false, false, &OutputMode::Tool),
        OutputMode::Native,
    );
    // Prompted does not rely on tools, so it is unaffected.
    assert_eq!(
        resolve_output_mode(true, true, false, false, &OutputMode::Prompted),
        OutputMode::Prompted,
    );
}

#[test]
fn output_tool_callable_for_auto_required_unset_or_a_specific_set_naming_it() {
    assert!(output_tool_callable(None, "final_result"));
    assert!(output_tool_callable(
        Some(&ToolChoice::Auto),
        "final_result"
    ));
    assert!(output_tool_callable(
        Some(&ToolChoice::Required),
        "final_result"
    ));
    assert!(!output_tool_callable(
        Some(&ToolChoice::None),
        "final_result"
    ));
    assert!(!output_tool_callable(
        Some(&ToolChoice::Specific {
            function_names: vec!["add".to_string()],
        }),
        "final_result"
    ));
    assert!(output_tool_callable(
        Some(&ToolChoice::Specific {
            function_names: vec!["add".to_string(), "final_result".to_string()],
        }),
        "final_result"
    ));
}

#[test]
fn pick_output_tool_name_defaults_when_unused() {
    let executable = tool_names(&["add", "subtract"]);
    assert_eq!(pick_output_tool_name(&executable), DEFAULT_OUTPUT_TOOL_NAME);
}

#[test]
fn pick_output_tool_name_avoids_collision_with_real_tools() {
    // A user tool literally named `final_result` must not be shadowed, or
    // the model's output call would be dispatched to the tool server.
    let executable = tool_names(&["final_result"]);
    assert_eq!(pick_output_tool_name(&executable), "final_result_1");

    let executable = tool_names(&["final_result", "final_result_1"]);
    assert_eq!(pick_output_tool_name(&executable), "final_result_2");
}

use super::gemini_api_types::GenerateContentRequest;
use crate::completion::{CompletionError, CompletionRequest};
use crate::message::{Message, UserContent};

fn request_with(preamble: Option<&str>, tools: bool) -> GenerateContentRequest {
    let mut tool_defs = Vec::new();
    if tools {
        tool_defs.push(crate::completion::ToolDefinition {
            name: "probe".to_owned(),
            description: "probe".to_owned(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        });
    }
    super::create_request_body(CompletionRequest {
        chat_history: preamble
            .map(Message::system)
            .into_iter()
            .chain([Message::User {
                content: vec![UserContent::text("hi")],
            }])
            .collect(),
        documents: vec![],
        tools: tool_defs,
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        model: None,
        output_schema: None,
        record_telemetry_content: false,
    })
    .expect("request should build")
}

#[test]
fn a_bare_id_is_rejected_before_the_request_goes_out() {
    let mut request = request_with(None, false);
    let error = request
        .with_cached_content("abc123")
        .expect_err("a bare id is not a handle");
    assert!(error.to_string().contains("cachedContents/<id>"), "{error}");
}

/// Gemini answers this with a 400 after a round trip, and does not say
/// *which* of the three conflicted. Rig should not need the round trip.
#[test]
fn a_preamble_alongside_a_cache_handle_is_rejected_locally() {
    let mut request = request_with(Some("you are a helpful assistant"), false);
    let error = request
        .with_cached_content("cachedContents/abc123")
        .expect_err("a system instruction conflicts with cached content");
    let message = error.to_string();
    assert!(message.contains("system instruction"), "{message}");
    assert!(message.contains("cachedContents/abc123"), "{message}");
}

#[test]
fn tools_alongside_a_cache_handle_are_rejected_locally() {
    let mut request = request_with(None, true);
    let error = request
        .with_cached_content("cachedContents/abc123")
        .expect_err("tools conflict with cached content");
    assert!(error.to_string().contains("tools"), "{error}");
}

/// The remedy differs per conflict, and only the tools arm may say so.
///
/// "Move them into the cache" is exactly right for a system instruction —
/// that is what explicit caching is for. It is wrong for tools: a cache's
/// tool set is declarations, and rig's `Agent` derives the declarations it
/// sends from the same registry it dispatches through, so it can neither
/// advertise nothing while executing something nor execute a tool that lives
/// only in the cache. A reader who follows the bare remedy ends up with a
/// cache no agent can use, so the caveat must appear for tools and must not
/// appear when only the system instruction conflicted.
#[test]
fn the_tools_conflict_says_a_cached_tool_set_is_declarations_only() {
    let mut with_tools = request_with(None, true);
    let tools_error = with_tools
        .with_cached_content("cachedContents/abc123")
        .expect_err("tools conflict with cached content");
    let tools_message = tools_error.to_string();
    assert!(
        tools_message.contains("declarations only"),
        "the tools conflict must correct the `move them into the cache` remedy: \
             {tools_message}"
    );
    assert!(
        tools_message.contains("CompletionModel"),
        "the tools conflict must name the surface that can actually use a cached tool set: \
             {tools_message}"
    );

    let mut with_preamble = request_with(Some("you are terse"), false);
    let preamble_message = with_preamble
        .with_cached_content("cachedContents/abc123")
        .expect_err("a system instruction conflicts with cached content")
        .to_string();
    assert!(
        !preamble_message.contains("declarations only"),
        "moving a system instruction into the cache IS the remedy; the tools caveat must not \
             leak onto it: {preamble_message}"
    );
}

/// The branch the `declares_functions` gate exists for, and the one neither
/// cell above reaches: `tools` is present but carries no function
/// declarations.
///
/// A provider-hosted tool runs on Gemini's side and needs no loop, so
/// "you must run the tool loop yourself" is nonsense advice for it — the
/// code comment says as much. Without this cell the gate is free to
/// collapse to `self.tools.is_some()`: that mutation leaves every other
/// test in the workspace green, and hands a `codeExecution` caller a
/// paragraph about dispatching declarations they never wrote.
#[test]
fn a_provider_hosted_tool_conflicts_without_the_function_declaration_caveat() {
    for (label, tools) in [
        ("codeExecution", serde_json::json!([{"codeExecution": {}}])),
        ("googleSearch", serde_json::json!([{"googleSearch": {}}])),
        (
            "an empty functionDeclarations list",
            serde_json::json!([{"functionDeclarations": []}]),
        ),
    ] {
        let mut request = build_with(None, Some(serde_json::json!({ "tools": tools })))
            .expect("request should build");
        let message = request
            .with_cached_content("cachedContents/abc123")
            .expect_err("tools conflict with a cache handle however they are declared")
            .to_string();

        assert!(
            message.contains("also set tools"),
            "{label}: the conflict must still name the tool set: {message}"
        );
        assert!(
            !message.contains("declarations only"),
            "{label}: a cache carrying a provider-hosted tool is usable from an agent, so \
                 the function-declaration caveat must not appear: {message}"
        );
    }
}

/// The other side of the gate, through the same route: function
/// declarations arriving in `additional_params.tools` do earn the caveat.
#[test]
fn a_smuggled_function_declaration_still_earns_the_caveat() {
    let mut request = build_with(
        None,
        Some(serde_json::json!({
            "tools": [{"functionDeclarations": [{"name": "probe", "description": "probe"}]}]
        })),
    )
    .expect("request should build");
    let message = request
        .with_cached_content("cachedContents/abc123")
        .expect_err("tools conflict with a cache handle")
        .to_string();
    assert!(message.contains("declarations only"), "{message}");
}

#[test]
fn a_clean_request_accepts_the_handle_and_puts_it_on_the_wire() {
    let mut request = request_with(None, false);
    request
        .with_cached_content("cachedContents/abc123")
        .expect("a request with no system instruction or tools should accept a handle");

    let body = serde_json::to_value(&request).expect("serialize");
    assert_eq!(
        body.get("cachedContent").and_then(|v| v.as_str()),
        Some("cachedContents/abc123")
    );
}

/// `additional_params` is `#[serde(flatten)]`, so a caller can smuggle
/// `cachedContent` onto the wire that way. The typed field must win rather
/// than the two colliding into a duplicate key.
/// `additional_params` is flattened *after* the named fields, so a
/// `cachedContent` smuggled through it silently overwrote the typed one and
/// bypassed the conflict validation entirely. An earlier version of this
/// test asserted the opposite invariant while passing an unrelated key
/// (`topK`), so it never exercised the collision and stayed green over a
/// real wire bug.
fn build_with(
    preamble: Option<&str>,
    additional: Option<serde_json::Value>,
) -> Result<GenerateContentRequest, CompletionError> {
    super::create_request_body(CompletionRequest {
        chat_history: preamble
            .map(Message::system)
            .into_iter()
            .chain([Message::User {
                content: vec![UserContent::text("hi")],
            }])
            .collect(),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: additional,
        model: None,
        output_schema: None,
        record_telemetry_content: false,
    })
}

/// The route a caller reaches without ever touching the typed API.
///
/// `additional_params` is `#[serde(flatten)]`, and flattened fields
/// serialize *after* the named ones — so a handle set this way used to
/// overwrite the typed field and, worse, skip the conflict validation
/// entirely, because that only inspects the typed fields. Validation now
/// happens where the request is built, so every route reaches it.
#[test]
fn a_handle_set_only_through_additional_params_is_still_validated() {
    let error = build_with(
        Some("you are terse"),
        Some(serde_json::json!({"cachedContent": "cachedContents/smuggled"})),
    )
    .expect_err("a preamble alongside a cache handle must be refused however it was set");
    let message = error.to_string();
    assert!(message.contains("system instruction"), "{message}");
    assert!(message.contains("cachedContents/smuggled"), "{message}");
}

/// The same route with nothing to conflict with lands in the *typed* field,
/// so it can no longer be overwritten by the flattened copy.
#[test]
fn a_clean_handle_from_additional_params_is_lifted_into_the_typed_field() {
    let request = build_with(
        None,
        Some(serde_json::json!({"cachedContent": "cachedContents/lifted", "topK": 5})),
    )
    .expect("a clean request should build");

    assert_eq!(
        request.cached_content.as_deref(),
        Some("cachedContents/lifted")
    );
    let body = serde_json::to_value(&request).expect("serialize");
    assert_eq!(
        body.get("cachedContent").and_then(|value| value.as_str()),
        Some("cachedContents/lifted")
    );
    // And the unrelated key still flattens through.
    assert_eq!(body.get("topK").and_then(|value| value.as_u64()), Some(5));
}

/// Two different handles is an ambiguity, not a precedence puzzle.
#[test]
fn setting_the_handle_twice_with_different_values_is_refused() {
    let mut request = build_with(
        None,
        Some(serde_json::json!({"cachedContent": "cachedContents/from_params"})),
    )
    .expect("a clean request should build");

    let error = request
        .with_cached_content("cachedContents/from_builder")
        .expect_err("two different handles cannot both win");
    let message = error.to_string();
    assert!(message.contains("from_params"), "{message}");
    assert!(message.contains("from_builder"), "{message}");
}

/// Setting the same handle twice is harmless and must not error.
#[test]
fn setting_the_same_handle_twice_is_accepted() {
    let mut request = build_with(
        None,
        Some(serde_json::json!({"cachedContent": "cachedContents/same"})),
    )
    .expect("a clean request should build");
    request
        .with_cached_content("cachedContents/same")
        .expect("the same handle twice is not ambiguous");
}

/// A non-string handle is a caller error, caught before the wire.
#[test]
fn a_non_string_handle_in_additional_params_is_refused() {
    let error = build_with(None, Some(serde_json::json!({"cachedContent": 42})))
        .expect_err("a numeric handle should be refused");
    assert!(error.to_string().contains("should be a string"), "{error}");
}

/// Unrelated `additional_params` keys must still flatten alongside the
/// typed field.
#[test]
fn unrelated_additional_params_coexist_with_the_typed_field() {
    let mut request = super::create_request_body(CompletionRequest {
        chat_history: vec![Message::User {
            content: vec![UserContent::text("hi")],
        }],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: Some(serde_json::json!({"topK": 5})),
        model: None,
        output_schema: None,
        record_telemetry_content: false,
    })
    .expect("request should build");
    request
        .with_cached_content("cachedContents/typed")
        .expect("handle should be accepted");

    let body = serde_json::to_value(&request).expect("serialize");
    assert_eq!(
        body.get("cachedContent").and_then(|v| v.as_str()),
        Some("cachedContents/typed")
    );
    assert_eq!(body.get("topK").and_then(|v| v.as_u64()), Some(5));
}

/// The other two fields a cached content owns, smuggled the same way the
/// handle was.
///
/// Lifting `cachedContent` alone left this open: the conflict check reads
/// the typed fields, and a `systemInstruction` or `toolConfig` sitting in
/// the flattened blob is not one. The handle was accepted, the request went
/// out, and Gemini answered the 400 the check exists to pre-empt — while
/// the docs promised it would not.
#[test]
fn a_system_instruction_or_tool_choice_from_additional_params_still_conflicts() {
    for (label, smuggled) in [
        (
            "systemInstruction",
            serde_json::json!({
                "systemInstruction": {"parts": [{"text": "you are terse"}], "role": "model"}
            }),
        ),
        (
            "toolConfig",
            serde_json::json!({
                "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}
            }),
        ),
    ] {
        let mut request = build_with(None, Some(smuggled)).expect("request should build");
        let message = request
            .with_cached_content("cachedContents/abc123")
            .expect_err(&format!("a smuggled {label} conflicts with a cache handle"))
            .to_string();
        let expected = if label == "systemInstruction" {
            "a system instruction"
        } else {
            "a tool choice"
        };
        assert!(
            message.contains(expected),
            "the {label} route must reach the same conflict as the typed field: {message}"
        );
    }
}

/// Whether or not a cache is involved, one field reached two ways is
/// ambiguous — and used to be resolved silently, in favour of whichever
/// serde emitted last.
#[test]
fn setting_a_field_twice_is_refused_rather_than_resolved_by_serialization_order() {
    let message = build_with(
        Some("you are terse"),
        Some(serde_json::json!({
            "systemInstruction": {"parts": [{"text": "you are verbose"}], "role": "model"}
        })),
    )
    .expect_err("a preamble and a smuggled system instruction are two answers")
    .to_string();
    assert!(
        message.contains("set the system instruction twice"),
        "{message}"
    );

    let message = super::create_request_body(CompletionRequest {
        chat_history: vec![Message::User {
            content: vec![UserContent::text("hi")],
        }],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: Some(crate::message::ToolChoice::Auto),
        additional_params: Some(serde_json::json!({
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}
        })),
        model: None,
        output_schema: None,
        record_telemetry_content: false,
    })
    .expect_err("a tool_choice and a smuggled toolConfig are two answers")
    .to_string();
    assert!(message.contains("set the tool choice twice"), "{message}");
}

/// Whatever the caller put in `additional_params` reaches the wire byte for
/// byte — including everything rig has no type for.
///
/// This is the cell that forbids the obvious implementation. Detecting
/// these fields by deserializing them into rig's own `ToolConfig` was tried,
/// and it silently dropped `allowedFunctionNames`: a request restricted to
/// one function became a request free to call any, with no error anywhere.
/// `additional_params` exists precisely for shapes rig does not model, so
/// the one thing this path must never do is narrow it.
#[test]
fn a_smuggled_field_reaches_the_wire_exactly_as_the_caller_wrote_it() {
    for smuggled in [
        // The lossy case: `allowedFunctionNames` is not a field of rig's
        // `FunctionCallingMode` (which spells it `allowed_function_names`).
        serde_json::json!({
            "toolConfig": {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": ["get_weather"]
                }
            }
        }),
        // Modes rig does not enumerate. `MODE_UNSPECIFIED` is the proto
        // default and `VALIDATED` was added after rig's enum was written;
        // the next one Google adds must not need a rig release either.
        serde_json::json!({"toolConfig": {"functionCallingConfig": {"mode": "MODE_UNSPECIFIED"}}}),
        serde_json::json!({"toolConfig": {"functionCallingConfig": {"mode": "VALIDATED"}}}),
        // The proto-original spelling, which proto3 JSON accepts alongside
        // the lowerCamelCase alias.
        serde_json::json!({"tool_config": {"function_calling_config": {"mode": "ANY"}}}),
        // A system instruction carrying a part kind rig genuinely has no
        // variant for. `videoMetadata` is a real Gemini `Part` field and
        // `PartKind` does not model it, so a round-trip through `Content`
        // would drop it. (An earlier version of this cell used `inlineData`,
        // which rig *does* model — it failed under the reverted
        // implementation only because `Content::role` re-serialized as an
        // added `null`, i.e. for the wrong reason.)
        serde_json::json!({
            "systemInstruction": {
                "parts": [{"videoMetadata": {"startOffset": "0s", "endOffset": "5s"}}]
            }
        }),
    ] {
        let request = build_with(None, Some(smuggled.clone()))
            .unwrap_or_else(|error| panic!("{smuggled} should build, got {error}"));
        // `to_string`, not `to_value`: a `Value` is a map, so round-tripping
        // through one silently deduplicates repeated members and shows only
        // the last. The claim here is about the bytes.
        let body = serde_json::to_string(&request).expect("serialize");

        for (key, expected) in smuggled.as_object().expect("object") {
            let needle = format!(
                "\"{key}\":{}",
                serde_json::to_string(expected).expect("serialize")
            );
            assert!(
                body.contains(&needle),
                "additional_params must reach the wire byte for byte; expected {needle} in \
                     {body}"
            );
        }
    }
}

/// The conflict check has to know both spellings, or it documents its own
/// bypass.
#[test]
fn the_proto_original_spelling_conflicts_too() {
    for (spelling, expected) in [
        ("system_instruction", "a system instruction"),
        ("tool_config", "a tool choice"),
    ] {
        let mut request = build_with(
            None,
            Some(serde_json::json!({ spelling: {"parts": [{"text": "x"}]} })),
        )
        .expect("request should build");
        let message = request
            .with_cached_content("cachedContents/abc123")
            .expect_err("the proto spelling is the same field")
            .to_string();
        assert!(
            message.contains(expected),
            "`{spelling}` must reach the same conflict as its camelCase alias: {message}"
        );
    }
}

/// A field reached through the blob is emitted *twice* — once as the typed
/// `null`, once from the blob — and Gemini accepts that.
///
/// Pinned rather than fixed, because "fixed" means `skip_serializing_if` on
/// `system_instruction` and `tool_config`, which would drop
/// `"systemInstruction":null` from every recorded Gemini request body and
/// invalidate the whole cassette corpus. It is worth pinning because it
/// looks like a bug and is not: measured against the live API,
/// `{"toolConfig":null,"toolConfig":{...allowedFunctionNames:["get_weather"]}}`
/// returns 200 *and honours the allow-list*, and the `systemInstruction`
/// equivalent returns 200 with the instruction applied. A `null` does not
/// set the proto field, so no `oneof` is claimed and nothing is overwritten.
///
/// Two *non-null* copies are a different matter, and that is exactly what
/// the set-twice refusals above prevent: Gemini answers a doubled
/// `systemInstruction` with `oneof field '_system_instruction' is already
/// set`, and merges a doubled `toolConfig` into the union of both
/// allowed-function lists.
///
/// So: if this cell ever fails, the fix is not to make it pass — it is to
/// re-record the corpus deliberately.
#[test]
fn a_blob_field_is_emitted_beside_its_typed_null_and_that_is_accepted() {
    let request = build_with(
        None,
        Some(serde_json::json!({"toolConfig": {"functionCallingConfig": {"mode": "ANY"}}})),
    )
    .expect("request should build");
    let body = serde_json::to_string(&request).expect("serialize");

    assert_eq!(
        body.matches("\"toolConfig\"").count(),
        2,
        "the typed null and the blob copy are both emitted, and the provider accepts it: \
             {body}"
    );
    assert!(
        body.find("\"toolConfig\":null") < body.find("\"toolConfig\":{"),
        "the null must come first — serde flattens the blob after the named fields, which is \
             what makes the caller's value the one that survives: {body}"
    );
}

/// The handle's own proto-original spelling, which used to skip everything.
///
/// `cached_content` is a working wire spelling — measured against the live
/// API, it reaches the cache lookup and answers `CachedContent not found`
/// for a bogus handle. Matching only `cachedContent` therefore meant a
/// handle written the other way was never lifted, `with_cached_content` was
/// never called, and every check it owns was skipped while the handle sailed
/// through the flattened blob onto the wire.
#[test]
fn the_protos_own_spelling_of_the_handle_is_validated_too() {
    // The conflict check.
    let message = build_with(
        Some("you are terse"),
        Some(serde_json::json!({"cached_content": "cachedContents/abc123"})),
    )
    .expect_err("a preamble conflicts with a handle however the handle is spelled")
    .to_string();
    assert!(message.contains("a system instruction"), "{message}");

    // The handle-shape check.
    let message = build_with(None, Some(serde_json::json!({"cached_content": "abc123"})))
        .expect_err("a bare id is not a handle however it is spelled")
        .to_string();
    assert!(message.contains("cachedContents/<id>"), "{message}");

    // The type check.
    let message = build_with(None, Some(serde_json::json!({"cached_content": 7})))
        .expect_err("a number is not a handle")
        .to_string();
    assert!(
        message.contains("additional_params.cached_content"),
        "{message}"
    );

    // The set-twice check, across the two spellings.
    let message = build_with(
        None,
        Some(serde_json::json!({
            "cachedContent": "cachedContents/one",
            "cached_content": "cachedContents/two"
        })),
    )
    .expect_err("two spellings naming different caches is still two caches")
    .to_string();
    assert!(message.contains("twice"), "{message}");

    // And the clean case still works, under either spelling.
    let request = build_with(
        None,
        Some(serde_json::json!({"cached_content": "cachedContents/ok"})),
    )
    .expect("a well-formed handle is accepted under the proto spelling");
    assert_eq!(request.cached_content.as_deref(), Some("cachedContents/ok"));
}

/// `with_cached_content` is `pub`, and on a hand-built request nothing has
/// run `extract_tools_from_additional_params` — so the blob is the only
/// place its tools live, and the check has to look there.
#[test]
fn a_hand_built_request_conflicts_on_tools_left_in_additional_params() {
    for (label, tools, wants_caveat) in [
        (
            "function declarations",
            serde_json::json!([{"functionDeclarations": [{"name": "f", "description": "d"}]}]),
            true,
        ),
        (
            "the proto spelling of function declarations",
            serde_json::json!([{"function_declarations": [{"name": "f", "description": "d"}]}]),
            true,
        ),
        (
            "a provider-hosted tool",
            serde_json::json!([{"codeExecution": {}}]),
            false,
        ),
    ] {
        let mut request = GenerateContentRequest {
            contents: vec![],
            generation_config: None,
            safety_settings: None,
            tools: None,
            tool_config: None,
            system_instruction: None,
            cached_content: None,
            additional_params: Some(serde_json::json!({ "tools": tools })),
        };
        let message = request
            .with_cached_content("cachedContents/abc123")
            .expect_err("tools in the blob conflict with a handle just as typed ones do")
            .to_string();
        assert!(message.contains("also set tools"), "{label}: {message}");
        assert_eq!(
            message.contains("declarations only"),
            wants_caveat,
            "{label}: the caveat must track whether functions were declared: {message}"
        );
    }
}

/// An explicit `null` is how serde spells "unset", so it must not be
/// mistaken for a value the caller set.
#[test]
fn an_explicit_null_is_not_a_conflict() {
    let mut request = build_with(
        None,
        Some(serde_json::json!({"systemInstruction": null, "toolConfig": null})),
    )
    .expect("request should build");
    request
        .with_cached_content("cachedContents/abc123")
        .expect("a null is not a value the caller set");
}

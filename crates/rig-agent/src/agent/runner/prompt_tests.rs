use crate::agent::ResponseIdentity;
use crate::agent::typed::{TypedPromptResponse, deserialize_structured_output};
use crate::run::response::{CompletionCall, MemoryAppend, PromptResponse};
use crate::run::transcript::is_empty_assistant_turn;
use crate::{
    agent::{
        AgentBuilder,
        hook::{
            AgentHook, DispatchAction, DispatchEvent, HookContext, InvalidToolCallAction,
            InvalidToolCallContext, ModelTurnAction, ModelTurnFinished,
        },
    },
    completion::{
        AssistantContent, CompletionRequest, FinishReason, Message, PromptError,
        StructuredOutputError, Usage,
    },
    test_utils::{
        AppendFailingMemory, CountingMemory, FailingMemory, MockAddTool, MockCompletionModel,
        MockContextProbeTool, MockSubtractTool, MockTurn, SessionId,
    },
    tool::ToolContext,
};
use rig_core::error::ProviderError;
use rig_core::message::{ToolChoice, UserContent};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// rig#2322 — the **blocking** surface enforces the same truncation
/// contract as the streamed one.
///
/// The premise of the whole fix is that the two surfaces disagreeing is
/// what let truncation surface as a blank answer, yet every other guard
/// test drives the streamed surface. Until `MockTurn::with_finish_reason`
/// existed the blocking mock could not report a reason at all, so
/// `runner.rs`'s `.with_finish_reason(resp.finish_reason())` — and its
/// propagation through `model_response` → `record_completion_call` → this
/// guard — was never exercised. Deleting that one line failed nothing.
#[tokio::test]
async fn blocking_prompt_rejects_an_empty_truncated_turn() {
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::from_contents(
        [],
    )
    .with_finish_reason(FinishReason::Length)]))
    .build();

    let err = agent
        .prompt("write a long essay")
        .await
        .expect_err("a content-less truncated turn must not return an empty answer");

    let rendered = format!("{err:?}");
    assert!(
        rendered.contains("Length") && rendered.contains("max_tokens"),
        "the blocking error must name the reason and the remedy: {rendered}"
    );
}

/// rig#2322 — blocking counterpart of the reasoning-only case: the shape
/// that motivated the predicate fix must be caught on both surfaces.
#[tokio::test]
async fn blocking_prompt_rejects_a_reasoning_only_truncated_turn() {
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::from_content(
        AssistantContent::Reasoning(rig_core::message::Reasoning::new(
            "thinking, never answering",
        )),
    )
    .with_finish_reason(FinishReason::Length)]))
    .build();

    let err = agent
        .prompt("solve this")
        .await
        .expect_err("a reasoning-only truncated turn must not return an empty answer");

    assert!(format!("{err:?}").contains("Length"));
}

#[derive(Serialize)]
struct SerializeOnly {
    value: &'static str,
}

#[derive(Deserialize)]
struct DeserializeOnly {
    value: String,
}

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct TypedAnswer {
    value: String,
}

#[test]
fn deserialize_structured_output_tolerates_fences_and_prose() {
    // Clean JSON (native / output-tool path).
    assert_eq!(
        deserialize_structured_output::<TypedAnswer>(r#"{"value":"x"}"#).unwrap(),
        TypedAnswer { value: "x".into() }
    );
    // Markdown-fenced JSON (weak Prompted-mode models).
    assert_eq!(
        deserialize_structured_output::<TypedAnswer>("```json\n{\"value\":\"y\"}\n```").unwrap(),
        TypedAnswer { value: "y".into() }
    );
    // Prose around the JSON object.
    assert_eq!(
        deserialize_structured_output::<TypedAnswer>(
            "Here you go: {\"value\":\"z\"} — hope that helps!"
        )
        .unwrap(),
        TypedAnswer { value: "z".into() }
    );
    // No JSON at all still errors.
    assert!(deserialize_structured_output::<TypedAnswer>("no json here").is_err());
}

#[derive(Clone)]
struct PanicOnUnknownToolHook;

impl AgentHook for PanicOnUnknownToolHook {
    // The rejected attempt's completion outcome still fires, observe-only;
    // the accepted-turn hook must not.
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        panic!("unknown tool response should fail before response hooks run")
    }
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event.tool_name().is_some() {
            panic!("unknown tool call should fail before tool hooks run")
        }
        DispatchAction::proceed()
    }
}

#[derive(Clone)]
struct RepairDefaultApiHook;

impl AgentHook for RepairDefaultApiHook {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        assert_eq!(event.tool_name, "default_api");
        Some(InvalidToolCallAction::repair("add"))
    }
}

#[derive(Clone)]
struct RetryDefaultApiHook;

impl AgentHook for RetryDefaultApiHook {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(InvalidToolCallAction::retry(format!(
            "Use one of these tools instead: {:?}",
            event.allowed_tools
        )))
    }
}

fn usage(input_tokens: u64, output_tokens: u64) -> Usage {
    Usage::new()
        .input_tokens(input_tokens)
        .output_tokens(output_tokens)
        .total_tokens(input_tokens + output_tokens)
}

#[test]
fn typed_prompt_response_serializes_with_serialize_only_output() {
    let response = TypedPromptResponse::new(
        SerializeOnly { value: "ok" },
        Usage::new()
            .input_tokens(1)
            .output_tokens(2)
            .total_tokens(3),
    );

    let json = serde_json::to_string(&response).expect("serialize typed prompt response");
    assert!(json.contains("\"value\":\"ok\""));
}

#[test]
fn typed_prompt_response_deserializes_with_deserialize_only_output() {
    let response: TypedPromptResponse<DeserializeOnly> = serde_json::from_str(
        r#"{"output":{"value":"ok"},"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3,"cached_input_tokens":0,"cache_creation_input_tokens":0,"reasoning_tokens":0}}"#,
    )
    .expect("deserialize typed prompt response");

    assert_eq!(response.requests(), 0);
    assert_eq!(response.output.value, "ok");
    assert_eq!(response.usage.input_tokens, Some(1));
    assert_eq!(response.usage.output_tokens, Some(2));
    assert_eq!(response.usage.total_tokens, Some(3));
}

#[test]
fn prompt_response_serializes_completion_calls_with_missing_usage() {
    let reported_usage = usage(3, 4);
    let response = PromptResponse::new("ok", reported_usage).with_completion_calls(vec![
        CompletionCall::new(0, Usage::default(), json!({"id": "resp_0"})),
        CompletionCall::new(1, reported_usage, json!({"id": "resp_1"})),
    ]);

    let value = serde_json::to_value(&response).expect("serialize prompt response");

    // Unreported usage serializes as an empty object: every counter is
    // `None`, and `None` counters are omitted rather than encoded as zero.
    assert_eq!(
        value.get("completion_calls"),
        Some(&json!([
            {
                "call_index": 0,
                "usage": {},
                "raw": {"id": "resp_0"}
            },
            {
                "call_index": 1,
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 4,
                    "total_tokens": 7,
                },
                "raw": {"id": "resp_1"}
            }
        ]))
    );

    let response: PromptResponse =
        serde_json::from_value(value).expect("deserialize prompt response");
    assert_eq!(
        response.completion_calls(),
        &[
            CompletionCall::new(0, Usage::default(), json!({"id": "resp_0"})),
            CompletionCall::new(1, reported_usage, json!({"id": "resp_1"}))
        ]
    );
    assert_eq!(response.requests(), 2);
}

#[test]
fn prompt_response_output_tool_marker_is_never_serialized() {
    let response = PromptResponse::new("ok", usage(1, 2)).with_output_tool_calls(3);

    let value = serde_json::to_value(&response).expect("serialize prompt response");
    assert!(value.get("output_tool_calls").is_none());

    let decoded: PromptResponse =
        serde_json::from_value(value).expect("deserialize prompt response");
    assert_eq!(decoded.output_tool_calls(), 0);
}

#[test]
fn empty_turn_classification_survives_a_serde_round_trip() {
    // A suspended run restored from JSON must classify its empty-text
    // turn exactly like the live run did, whatever spelling of "no
    // extras" the JSON carries, and an *annotated* empty block must
    // still read as content either way. The serde canonicalization
    // mechanics behind this (`{}`/`null` decode to `None`, empty params
    // never serialize) are pinned where they live, by rig-core's
    // `empty_params_canonicalize_to_none_in_both_serde_directions` —
    // this test asserts classification only.
    let live = vec![AssistantContent::text("")];
    assert!(is_empty_assistant_turn(&live));

    let round: Vec<AssistantContent> =
        serde_json::from_str(&serde_json::to_string(&live).expect("serialize"))
            .expect("deserialize");
    assert!(
        is_empty_assistant_turn(&round),
        "restored turn must classify like the live one: {round:?}"
    );

    // An explicit `{}` or `null` in the JSON — the shape a mechanical
    // migration script writes — classifies exactly like an absent field.
    for empty_spelling in [serde_json::json!({}), serde_json::Value::Null] {
        let migrated: Vec<AssistantContent> = serde_json::from_value(serde_json::json!([
            {"type": "text", "text": "", "additional_params": empty_spelling}
        ]))
        .expect("deserialize migrated");
        assert!(is_empty_assistant_turn(&migrated));
    }

    // Text with no provider item is empty, live and restored alike.
    let canonical_absent = vec![AssistantContent::Text(rig_core::message::Text::new(""))];
    assert!(is_empty_assistant_turn(&canonical_absent));
    let restored: Vec<AssistantContent> =
        serde_json::from_value(serde_json::to_value(&canonical_absent).expect("serialize"))
            .expect("deserialize");
    assert!(is_empty_assistant_turn(&restored));

    // An empty block holding a current provider item carries data; one whose
    // item no longer matches it (an edit) replays as nothing, so it is empty.
    let annotated =
        vec![AssistantContent::text("").with_native(serde_json::json!({"signature": "sig"}))];
    let restored: Vec<AssistantContent> =
        serde_json::from_value(serde_json::to_value(&annotated).expect("serialize"))
            .expect("deserialize annotated");
    assert!(
        !is_empty_assistant_turn(&restored),
        "an annotated empty block carries data: {restored:?}"
    );
    let stale: Vec<AssistantContent> = serde_json::from_value(serde_json::json!([
        {"type": "text", "text": "", "native": {"item": {"signature": "sig"}, "fingerprint": "0000000000000000"}}
    ]))
    .expect("deserialize stale");
    assert!(is_empty_assistant_turn(&stale), "{stale:?}");
}

#[test]
fn the_type_key_is_the_tag_and_the_untagged_shape_does_not_load() {
    // Assistant content is tagged like user content: `"type"` is consumed
    // as the discriminant, never captured into `additional_params`. And
    // there is deliberately no untagged fallback — the bare shape 0.41
    // serialized fails to deserialize (MIGRATING carries the recipe),
    // pinned here so removing the tag requirement is a visible decision,
    // not an accident.
    let tagged: Vec<AssistantContent> =
        serde_json::from_value(serde_json::json!([{"type": "text", "text": "ok"}]))
            .expect("deserialize");
    let [AssistantContent::Text(text)] = tagged.as_slice() else {
        panic!("expected one text block, got {tagged:?}");
    };
    assert_eq!(text.text, "ok");
    assert_eq!(text.native, None, "the tag is not data");

    serde_json::from_value::<Vec<AssistantContent>>(serde_json::json!([{"text": "ok"}]))
        .expect_err("the untagged shape must not deserialize");
}

#[test]
fn prompt_response_roundtrip_preserves_structured_content() {
    let response = PromptResponse::from_content(
        vec![
            AssistantContent::Reasoning(rig_core::message::Reasoning::new("thinking")),
            AssistantContent::text("visible "),
            AssistantContent::text("text"),
        ],
        Usage::default(),
    );
    assert_eq!(response.output(), "visible text");

    let value = serde_json::to_value(&response).expect("serialize prompt response");
    assert!(
        value.get("content").is_some(),
        "content is part of the serialized shape"
    );
    assert!(
        value.get("output").is_none(),
        "output is derived from content, not stored"
    );

    let round: PromptResponse = serde_json::from_value(value).expect("deserialize prompt response");
    assert_eq!(round.output(), "visible text");
    // Compare parts directly to sidestep the `Text::additional_params` serde
    // round-trip asymmetry.
    assert!(matches!(
        round.content(),
        [AssistantContent::Reasoning(_), AssistantContent::Text(a), AssistantContent::Text(b)]
            if a.text == "visible " && b.text == "text"
    ));
}

#[test]
fn prompt_response_serialize_and_deserialize_agree_on_wire_shape() {
    // `content` is a required, bare list in both serde directions — the
    // pre-`content` reconstruction (and the shadow repr that carried it)
    // is gone, so serialize and deserialize agree by construction. Pin
    // the shape: `content` present, `completion_calls` omitted only when
    // empty, and the value round-trips.
    let response = PromptResponse::new("hi", usage(1, 2))
        .with_completion_calls(vec![CompletionCall::new(0, usage(1, 2), json!({}))]);

    let from_response = serde_json::to_value(&response).expect("serialize response");
    assert!(from_response.get("content").is_some());
    assert!(from_response.get("completion_calls").is_some());

    let round: PromptResponse =
        serde_json::from_value(from_response).expect("deserialize response");
    assert_eq!(round.output(), "hi");
    assert_eq!(round.usage(), usage(1, 2));
    assert_eq!(
        round.completion_calls(),
        &[CompletionCall::new(0, usage(1, 2), json!({}))]
    );

    // The omission direction of `completion_calls`' skip-when-empty:
    // an empty list serializes without the key (the shadow-era wire
    // shape), and the keyless JSON still deserializes.
    let bare = serde_json::to_value(PromptResponse::new("hi", usage(1, 2)))
        .expect("serialize bare response");
    assert!(bare.get("completion_calls").is_none());
    let round: PromptResponse = serde_json::from_value(bare).expect("deserialize keyless response");
    assert!(round.completion_calls().is_empty());
}

#[tokio::test]
async fn typed_prompt_response_preserves_completion_calls() {
    let call_usage = Usage::new()
        .input_tokens(4)
        .output_tokens(6)
        .total_tokens(10);
    let turn = MockTurn::text(r#"{"value":"ok"}"#).with_usage(call_usage);
    let raw = turn.raw().expect("a scripted turn has a document");
    let model = MockCompletionModel::from_turns([turn]);
    let agent = AgentBuilder::new(model).build();

    let response = agent
        .prompt_typed::<TypedAnswer>("return typed json")
        .await
        .expect("typed prompt should succeed");

    assert_eq!(
        response.output,
        TypedAnswer {
            value: "ok".to_string()
        }
    );
    assert_eq!(response.usage, call_usage);
    assert_eq!(
        response.completion_calls(),
        &[CompletionCall::new(0, call_usage, raw)]
    );
    assert_eq!(
        response.messages.len(),
        2,
        "the accepted attempt's prompt and answer: {:?}",
        response.messages
    );
}

#[tokio::test]
async fn typed_prompt_deserialization_error_keeps_the_model_output() {
    let model = MockCompletionModel::from_turns([MockTurn::text("the answer is ok")]);
    let agent = AgentBuilder::new(model).build();

    let error = agent
        .prompt_typed::<TypedAnswer>("return typed json")
        .await
        .expect_err("prose is not a TypedAnswer");

    match error {
        StructuredOutputError::Deserialization { output, .. } => {
            assert_eq!(output, "the answer is ok");
        }
        other => panic!("expected a deserialization error, got {other:?}"),
    }
}

fn validate_follow_up_tool_history(request: &CompletionRequest) {
    let history = request.chat_history.clone();
    assert_eq!(
        history.len(),
        3,
        "follow-up request should contain the prompt, assistant tool call, and user tool result: {history:?}"
    );

    assert!(matches!(
        history.first(),
        Some(Message::User { content })
            if matches!(
                content.first(),
                Some(UserContent::Text(text)) if text.text == "do tool work"
            )
    ));

    // The provider's correlator was overridden to "call_1"; the call carries
    // that one id and the result answers it.
    assert!(matches!(
        history.get(1),
        Some(Message::Assistant(rig_core::message::AssistantMessage { content, .. }))
            if matches!(
                content.first(),
                Some(AssistantContent::ToolCall(tool_call))
                    if tool_call.id.provider().map(|provider| provider.as_str()) == Some("call_1")
            )
    ));

    assert!(matches!(
        history.get(2),
        Some(Message::User { content })
            if matches!(
                content.first(),
                Some(UserContent::ToolResult(tool_result))
                    if tool_result.call.provider().map(|provider| provider.as_str()) == Some("call_1")
            )
    ));
}

fn history_contains_tool_call(history: &[Message], tool_name: &str) -> bool {
    history.iter().any(|message| {
        matches!(
            message,
            Message::Assistant(rig_core::message::AssistantMessage { content, .. })
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.function.name == tool_name
                ))
        )
    })
}

#[tokio::test]
async fn unknown_tool_call_fails_before_non_streaming_second_request() {
    let model = MockCompletionModel::from_turns([
        MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 1, "y": 2})),
        MockTurn::text("should not be requested"),
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let err = agent
        .prompt("use the tool")
        .add_hook(PanicOnUnknownToolHook)
        .max_turns(3)
        .await
        .expect_err("unknown model-emitted tool should fail");

    match err {
        PromptError::UnknownToolCall {
            tool_name,
            available_tools,
            allowed_tools,
            chat_history,
        } => {
            assert_eq!(tool_name, "default_api");
            assert_eq!(available_tools, vec!["add".to_string()]);
            assert_eq!(allowed_tools, vec!["add".to_string()]);
            assert!(history_contains_tool_call(&chat_history, "default_api"));
        }
        other => panic!("expected UnknownToolCall, got {other:?}"),
    }
    assert_eq!(recorded.request_count(), 1);
}

/// Context values persist for the whole run, across *multiple* tool-call rounds
/// (the headline value prop). The model calls the probe in two consecutive
/// rounds; both must observe the same injected value, not just the first.
#[tokio::test]
async fn tool_context_persists_across_multiple_rounds() {
    let model = MockCompletionModel::from_turns([
        MockTurn::tool_call("c1", "context_probe", json!({})),
        MockTurn::tool_call("c2", "context_probe", json!({})),
        MockTurn::text("done"),
    ]);
    let probe = MockContextProbeTool::default();
    let agent = AgentBuilder::new(model).tool(probe.clone()).build();

    let mut context = ToolContext::new();
    context.insert(SessionId("abc-123".to_string())).unwrap();

    let out = agent
        .prompt("use the tool twice")
        .tool_context(context)
        .max_turns(5)
        .await
        .expect("run succeeds");

    assert_eq!(out.output(), "done");
    assert_eq!(
        probe.observations(),
        vec!["session:abc-123".to_string(), "session:abc-123".to_string()],
    );
}

#[tokio::test]
async fn disallowed_specific_tool_call_fails_before_non_streaming_second_request() {
    let model = MockCompletionModel::from_turns([
        MockTurn::tool_call("tool_call_1", "subtract", json!({"x": 3, "y": 1})),
        MockTurn::text("should not be requested"),
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model)
        .tool(MockAddTool)
        .tool(MockSubtractTool)
        .tool_choice(ToolChoice::Specific {
            function_names: vec![rig_core::message::ToolName::new("add").expect("tool name")],
        })
        .build();

    let err = agent
        .prompt("use the allowed tool")
        .add_hook(PanicOnUnknownToolHook)
        .max_turns(3)
        .await
        .expect_err("disallowed model-emitted tool should fail");

    match err {
        PromptError::UnknownToolCall {
            tool_name,
            available_tools,
            allowed_tools,
            chat_history,
        } => {
            assert_eq!(tool_name, "subtract");
            assert_eq!(
                available_tools,
                vec!["add".to_string(), "subtract".to_string()]
            );
            assert_eq!(allowed_tools, vec!["add".to_string()]);
            assert!(history_contains_tool_call(&chat_history, "subtract"));
        }
        other => panic!("expected UnknownToolCall, got {other:?}"),
    }
    assert_eq!(recorded.request_count(), 1);
}

#[tokio::test]
async fn tool_choice_none_rejects_non_streaming_tool_call() {
    let model = MockCompletionModel::from_turns([
        MockTurn::tool_call("tool_call_1", "add", json!({"x": 1, "y": 2})),
        MockTurn::text("should not be requested"),
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model)
        .tool(MockAddTool)
        .tool_choice(ToolChoice::None)
        .build();

    let err = agent
        .prompt("do not use tools")
        .add_hook(PanicOnUnknownToolHook)
        .max_turns(3)
        .await
        .expect_err("ToolChoice::None should reject returned tool calls");

    match err {
        PromptError::UnknownToolCall {
            tool_name,
            available_tools,
            allowed_tools,
            chat_history,
        } => {
            assert_eq!(tool_name, "add");
            assert_eq!(available_tools, vec!["add".to_string()]);
            assert!(allowed_tools.is_empty());
            assert!(history_contains_tool_call(&chat_history, "add"));
        }
        other => panic!("expected UnknownToolCall, got {other:?}"),
    }
    assert_eq!(recorded.request_count(), 1);
}

#[tokio::test]
async fn typed_prompt_invalid_tool_call_hook_can_repair_tool_name() {
    let model = MockCompletionModel::from_turns([
        MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
        MockTurn::text(r#"{"value":"repaired"}"#),
    ]);
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let response = agent
        .prompt_typed::<TypedAnswer>("return typed json")
        .add_hook(RepairDefaultApiHook)
        .max_turns(3)
        .await
        .expect("typed prompt should repair invalid tool call");

    assert_eq!(
        response.output,
        TypedAnswer {
            value: "repaired".to_string()
        }
    );
}

#[tokio::test]
async fn typed_prompt_invalid_tool_call_hook_can_retry_and_parse_response() {
    let model = MockCompletionModel::from_turns([
        MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
        MockTurn::text(r#"{"value":"retried"}"#),
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let response = agent
        .prompt_typed::<TypedAnswer>("return typed json")
        .add_hook(RetryDefaultApiHook)
        .max_invalid_tool_call_retries(1)
        .max_turns(3)
        .await
        .expect("typed prompt should retry invalid tool call");

    assert_eq!(
        response.output,
        TypedAnswer {
            value: "retried".to_string()
        }
    );
    assert_eq!(recorded.request_count(), 2);
}

#[tokio::test]
async fn invalid_specific_tool_choice_fails_before_non_streaming_provider_request() {
    let model = MockCompletionModel::text("should not be requested");
    let recorded = model.clone();
    let agent = AgentBuilder::new(model)
        .tool(MockAddTool)
        .tool_choice(ToolChoice::Specific {
            function_names: vec![rig_core::message::ToolName::new("missing").expect("tool name")],
        })
        .build();

    let err = agent
        .prompt("use the missing tool")
        .await
        .expect_err("invalid ToolChoice::Specific should fail before provider request");

    match err {
        PromptError::Provider(ProviderError::Request(err)) => {
            let msg = err.to_string();
            assert!(msg.contains("missing"), "got: {msg}");
            assert!(msg.contains("add"), "got: {msg}");
        }
        other => panic!("expected ProviderError::Request, got {other:?}"),
    }
    assert_eq!(recorded.request_count(), 0);
}

#[tokio::test]
async fn prompt_request_stops_cleanly_on_empty_terminal_turn() {
    let first_call_usage = Usage::new()
        .input_tokens(1)
        .output_tokens(1)
        .total_tokens(2);
    let second_call_usage = Usage::new()
        .input_tokens(1)
        .output_tokens(1)
        .total_tokens(2);
    let first_turn = MockTurn::tool_call("tool_call_1", "add", json!({"x": 1, "y": 2}))
        .with_call_id("call_1")
        .with_usage(first_call_usage);
    let second_turn = MockTurn::text("").with_usage(second_call_usage);
    let first_raw = first_turn.raw().expect("a scripted turn has a document");
    let second_raw = second_turn.raw().expect("a scripted turn has a document");
    let model = MockCompletionModel::from_turns([first_turn, second_turn]);
    let agent = AgentBuilder::new(model.clone()).tool(MockAddTool).build();

    let response = agent
        .prompt("do tool work")
        .max_turns(3)
        .await
        .expect("empty terminal turn should not error");

    assert!(response.output().is_empty());
    assert_eq!(
        response.usage,
        Usage::new()
            .input_tokens(2)
            .output_tokens(2)
            .total_tokens(4)
    );
    assert_eq!(
        response.completion_calls(),
        &[
            CompletionCall::new(0, first_call_usage, first_raw),
            CompletionCall::new(1, second_call_usage, second_raw)
        ]
    );

    let history = response.messages;
    assert_eq!(history.len(), 3);
    assert!(matches!(
        history.first(),
        Some(Message::User { content })
            if matches!(
                content.first(),
                Some(UserContent::Text(text)) if text.text == "do tool work"
            )
    ));
    assert!(history.iter().any(|message| matches!(
        message,
        Message::Assistant(rig_core::message::AssistantMessage { content, .. })
            if matches!(
                content.first(),
                Some(AssistantContent::ToolCall(tool_call))
                    if tool_call.id.provider().map(|provider| provider.as_str()) == Some("call_1")
                        && tool_call.id.provider().as_ref().is_some_and(
                            |provider| provider.as_str() == "call_1"
                        )
            )
    )));
    assert!(history.iter().any(|message| matches!(
        message,
        Message::User { content }
            if matches!(
                content.first(),
                Some(UserContent::ToolResult(tool_result))
                    if tool_result.call.provider().map(|provider| provider.as_str()) == Some("call_1")
                        && tool_result.call.provider().as_ref().is_some_and(
                            |provider| provider.as_str() == "call_1"
                        )
            )
    )));
    assert!(!history.iter().any(|message| matches!(
        message,
        Message::Assistant(rig_core::message::AssistantMessage { content, .. })
            if content.iter().any(|item| matches!(
                item,
                AssistantContent::Text(text) if text.text.is_empty()
            ))
    )));
    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    validate_follow_up_tool_history(&requests[1]);
}

// ----- Conversation memory integration tests -----

use rig_core::memory::ConversationMemory;

#[tokio::test]
async fn append_persists_only_newly_committed_messages() {
    // With pre-loaded history, a run must append only the new turn's
    // messages, never re-append the loaded history (which would duplicate
    // it). Pre-load directly through `inner()` so it does not count as an
    // append by the run.
    let memory = CountingMemory::default();
    memory
        .inner()
        .append(
            &"t1".into(),
            vec![Message::user("old-q"), Message::assistant("old-a")],
        )
        .await
        .unwrap();

    let model = MockCompletionModel::text("new-a");
    let agent = AgentBuilder::new(model).memory(memory.clone()).build();

    let _ = agent
        .prompt("new-q")
        .conversation("t1")
        .await
        .expect("prompt should succeed");

    assert_eq!(memory.append_count(), 1, "one append for the run");

    let stored = memory.load(&"t1".into()).await.unwrap();
    // preloaded [old-q, old-a] + new [new-q, new-a]; re-appending the loaded
    // history would instead make this 6.
    assert_eq!(
        stored.len(),
        4,
        "only the new turn is appended, loaded history is not duplicated: {stored:?}"
    );
    assert!(
        matches!(
            stored.first(),
            Some(Message::User { content })
                if matches!(content.first(), Some(UserContent::Text(t)) if t.text == "old-q")
        ),
        "loaded history is preserved once at the front: {stored:?}"
    );
}

#[tokio::test]
async fn without_memory_disables_for_request() {
    let memory = CountingMemory::default();
    let model = MockCompletionModel::text("ack");
    let agent = AgentBuilder::new(model)
        .memory(memory.clone())
        .conversation("t1")
        .build();

    let response = agent
        .prompt("hello")
        .without_memory()
        .await
        .expect("prompt should succeed");

    assert_eq!(memory.load_count(), 0);
    assert_eq!(memory.append_count(), 0);
    assert_eq!(response.memory_append, None);
}

#[tokio::test]
async fn memory_load_error_surfaces_as_prompt_error() {
    let model = MockCompletionModel::text("ack");
    let agent = AgentBuilder::new(model)
        .memory(FailingMemory::default())
        .build();
    let result = agent.prompt("hello").conversation("t1").await;

    match result {
        Err(PromptError::Memory(err)) => {
            let msg = err.to_string();
            assert!(msg.contains("load boom"), "got: {msg}");
        }
        other => panic!("expected PromptError::Memory, got {other:?}"),
    }
}

/// A refused append does not fail the run: the answer stands, and the
/// response says the transcript was not persisted — a caller can tell the
/// two endings apart without a hook or the effect log.
#[tokio::test]
async fn memory_append_error_does_not_drop_response() {
    let model = MockCompletionModel::text("ack");
    let agent = AgentBuilder::new(model)
        .memory(AppendFailingMemory::default())
        .build();
    let response = agent
        .prompt("hello")
        .conversation("t1")
        .await
        .expect("append failure must not block successful completion");

    assert_eq!(response.output(), "ack");
    assert_eq!(
        response.messages.len(),
        2,
        "the transcript the run tried to persist"
    );
    let report = response
        .memory_append
        .as_ref()
        .and_then(MemoryAppend::failure)
        .expect("the refused append is reported on the response");
    assert_eq!(report.kind, rig_core::error::ErrorKind::MemoryBackend);
    assert!(report.message.contains("append boom"), "{report:?}");
    assert!(!response.memory_append.as_ref().unwrap().is_acknowledged());
}

/// The identity fields are skipped when `None` (rig#2265), so a record
/// written without them loads with every identity field `None`.
#[test]
fn completion_call_without_identity_fields_deserializes() {
    let call: CompletionCall = serde_json::from_str(
        r#"{"call_index": 3, "usage": {"input_tokens": 1, "output_tokens": 2,
                "total_tokens": 3, "cached_input_tokens": 0,
                "cache_creation_input_tokens": 0, "reasoning_tokens": 0},
                "raw": {}}"#,
    )
    .expect("a CompletionCall without identity fields should load");
    assert_eq!(call.call_index, 3);
    assert_eq!(call.identity(), ResponseIdentity::default());
}

/// And a populated record round-trips the identity losslessly.
#[test]
fn completion_call_identity_round_trips() {
    let call = CompletionCall::new(0, crate::completion::Usage::default(), json!({}))
        .with_identity(ResponseIdentity {
            response_id: Some("resp_1".into()),
            provider_request_id: Some("req_1".into()),
        });
    let json = serde_json::to_string(&call).expect("serialize");
    let restored: CompletionCall = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored, call);
}

/// A turn the provider failed with nothing to show never ends a run as a
/// success (an empty answer would pass for one).
#[tokio::test]
async fn a_failed_turn_without_an_answer_fails_the_run() {
    let model = MockCompletionModel::from_turns([MockTurn::from_contents(Vec::new())
        .with_finish_reason(rig_core::completion::FinishReason::Other(
            "MALFORMED_FUNCTION_CALL".to_owned(),
        ))]);
    let agent = AgentBuilder::new(model).build();
    let error = agent
        .prompt("hello")
        .await
        .expect_err("a failed, answerless turn fails the run");
    assert!(
        error.to_string().contains("MALFORMED_FUNCTION_CALL"),
        "the run names why the provider failed the turn: {error}"
    );
}

fn unknown_finish(turn: MockTurn) -> MockTurn {
    turn.with_finish_reason(FinishReason::Other("weird".to_owned()))
}

fn replays_assistant_text(request: &CompletionRequest, text: &str) -> bool {
    request.chat_history.iter().any(|message| {
        matches!(
            message,
            Message::Assistant(turn) if turn.content.iter().any(|block| matches!(
                block,
                AssistantContent::Text(t) if t.text == text
            ))
        )
    })
}

/// By default a text answer that ends in an unknown finish reason fails the
/// run, naming the reason, as replay would leave the turn out.
#[tokio::test]
async fn an_answer_with_an_unknown_finish_reason_fails_the_run_by_default() {
    let model = MockCompletionModel::from_turns([unknown_finish(MockTurn::text("answer"))]);
    let agent = AgentBuilder::new(model).build();
    let error = agent
        .prompt("hello")
        .await
        .expect_err("a failed turn is no answer");
    assert!(error.to_string().contains("weird"), "{error}");
}

/// Accepted, the answer succeeds and the next run replays it.
#[tokio::test]
async fn an_accepted_unknown_finish_reason_answers_and_replays() {
    let model = MockCompletionModel::from_turns([
        unknown_finish(MockTurn::text("answer")),
        MockTurn::text("again"),
    ]);
    let agent = AgentBuilder::new(model.clone())
        .accept_unknown_finish_reasons(true)
        .build();
    let first = agent
        .prompt("hello")
        .await
        .expect("the accepted turn answers");
    assert_eq!(first.output(), "answer");
    agent
        .prompt("and now")
        .history(first.messages)
        .await
        .expect("the follow-up answers");
    let requests = model.requests();
    assert!(
        requests
            .iter()
            .all(|request| request.accept_unknown_finish_reasons)
    );
    assert!(replays_assistant_text(&requests[1], "answer"));
}

/// A run can accept unknown reasons where its agent does not.
#[tokio::test]
async fn a_run_accepts_unknown_finish_reasons_on_its_own() {
    let model = MockCompletionModel::from_turns([unknown_finish(MockTurn::text("answer"))]);
    let agent = AgentBuilder::new(model).build();
    let answer = agent
        .prompt("hello")
        .accept_unknown_finish_reasons(true)
        .await
        .expect("the run accepts the reason");
    assert_eq!(answer.output(), "answer");
}

/// A host driving the run itself from the agent's spec keeps the setting:
/// the prepared request accepts the reason and the run answers.
#[tokio::test]
async fn a_sans_io_run_from_the_agent_spec_accepts_unknown_finish_reasons() {
    use crate::run::{AgentRun, AgentRunStep, ModelTurn, prepare_request};

    let model = MockCompletionModel::from_turns([unknown_finish(MockTurn::text("answer"))]);
    let agent = AgentBuilder::new(model.clone())
        .accept_unknown_finish_reasons(true)
        .build();
    let spec = agent.run_spec();
    assert!(spec.accept_unknown_finish_reasons);
    let mut run = AgentRun::from_spec(&spec, "hello", None);
    let output = loop {
        match run.next_step().expect("a step") {
            AgentRunStep::CallModel {
                prompt, history, ..
            } => {
                let prepared =
                    prepare_request(&spec, &Default::default(), &history, Vec::new(), None, None)
                        .expect("prepared");
                let executable = prepared.executable_tool_names.clone();
                let allowed = prepared.allowed_tool_names.clone();
                let request = prepared.apply(CompletionRequest::new(prompt));
                assert!(request.accept_unknown_finish_reasons);
                let response = model.call(request).await.expect("the reply folds");
                run.model_response(ModelTurn::new(
                    response.head(),
                    response.choice,
                    response.usage,
                    executable,
                    allowed,
                    response.raw,
                ))
                .expect("the accepted turn is no failure");
            }
            AgentRunStep::CallTools { .. } => panic!("no tools were called"),
            AgentRunStep::Done(response) => break response.output().to_owned(),
        }
    };
    assert_eq!(output, "answer");
}

/// The tool calls of a turn with an unknown finish reason run only when the
/// agent accepts the reason.
#[tokio::test]
async fn calls_with_an_unknown_finish_reason_run_only_when_accepted() {
    for accept in [false, true] {
        let model = MockCompletionModel::from_turns([
            unknown_finish(MockTurn::tool_call(
                "call_1",
                "add",
                json!({"x": 1, "y": 2}),
            )),
            MockTurn::text("3"),
        ]);
        let agent = AgentBuilder::new(model.clone())
            .tool(MockAddTool)
            .accept_unknown_finish_reasons(accept)
            .build();
        let outcome = agent.prompt("add").max_turns(3).await;
        if accept {
            assert_eq!(outcome.expect("the call runs").output(), "3");
            assert_eq!(model.request_count(), 2);
        } else {
            let error = outcome.expect_err("the failed turn runs no call");
            assert!(
                error.to_string().contains("none of its tool calls ran"),
                "{error}"
            );
            assert_eq!(model.request_count(), 1);
        }
    }
}

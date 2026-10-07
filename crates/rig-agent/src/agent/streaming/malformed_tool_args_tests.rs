//! A tool call whose arguments are not a JSON object never reaches the tool.
//! The tool step offers it to the invalid-call hook, then answers it with a
//! tool result so the model can call again. A model that keeps sending
//! malformed arguments is answered until `max_turns`, or stopped after
//! `max_consecutive_malformed_tool_calls` consecutive turns when that is set.
//! Every case runs on the blocking and the streamed
//! surface, which share the tool step, and asserts what the model sees on
//! its next request.

use std::sync::{Arc, Mutex};

use super::MultiTurnStreamItem;
use crate::AgentRun;
use crate::agent::hook::{AgentHook, HookContext};
use crate::agent::{
    AgentBuilder, AgentRunner, InvalidToolCallAction, InvalidToolCallContext, InvalidToolCallReason,
};
use crate::run::{
    AgentRunStep, ModelTurn, ModelTurnOutcome, OutputMode, TurnPolicy, UnhandledInvalidToolCall,
};
use crate::test_utils::{MockAddTool, MockCompletionModel, MockStreamEvent, MockTurn};
use futures::StreamExt;
use rig_core::completion::{CompletionRequest, Usage};
use rig_core::message::{
    AssistantContent, AssistantMessage, Message, ToolCall, ToolFunction, ToolName, ToolResult,
    ToolResultContent, UserContent,
};
use serde_json::json;

const RAW: &str = "{\"x\": 2, \"y\": \x01";

/// One scripted model turn.
#[derive(Clone, Copy)]
enum Turn {
    /// A call to `add` whose arguments are [`RAW`].
    Malformed,
    /// A call to `add` with valid arguments.
    WellFormed,
    /// The final answer, `recovered`.
    Answer,
}

#[derive(Clone, Copy, Debug)]
enum Surface {
    Blocking,
    Streamed,
}

const SURFACES: [Surface; 2] = [Surface::Blocking, Surface::Streamed];

fn call_id(turn: usize) -> String {
    format!("tool_call_{turn}")
}

fn add() -> ToolName {
    ToolName::new("add").unwrap_or_else(|error| panic!("{error}"))
}

fn model(surface: Surface, script: &[Turn]) -> MockCompletionModel {
    match surface {
        Surface::Blocking => MockCompletionModel::from_turns(script.iter().enumerate().map(
            |(index, turn)| match turn {
                Turn::Malformed => MockTurn::from_content(AssistantContent::ToolCall(
                    ToolCall::from_wire(call_id(index + 1), ToolFunction::parse(add(), RAW)),
                )),
                Turn::WellFormed => {
                    MockTurn::tool_call(call_id(index + 1), "add", json!({"x": 1, "y": 2}))
                }
                Turn::Answer => MockTurn::text("recovered"),
            },
        )),
        Surface::Streamed => MockCompletionModel::from_stream_turns(script.iter().enumerate().map(
            |(index, turn)| {
                let id = call_id(index + 1);
                let mut events = match turn {
                    Turn::Malformed => vec![
                        MockStreamEvent::tool_call_name_delta(&id, "add"),
                        MockStreamEvent::tool_call_arguments_delta(&id, RAW),
                        MockStreamEvent::tool_call_end(&id),
                    ],
                    Turn::WellFormed => {
                        vec![MockStreamEvent::tool_call(
                            &id,
                            "add",
                            json!({"x": 1, "y": 2}),
                        )]
                    }
                    Turn::Answer => vec![MockStreamEvent::text("recovered")],
                };
                events.push(MockStreamEvent::final_response_with_total_tokens(4));
                events
            },
        )),
    }
}

/// Answers every invalid call with `action` and records what it saw.
#[derive(Clone, Default)]
struct Decide {
    action: Option<InvalidToolCallAction>,
    seen: Arc<Mutex<Vec<InvalidToolCallContext>>>,
}

impl Decide {
    fn new(action: Option<InvalidToolCallAction>) -> Self {
        Self {
            action,
            seen: Arc::default(),
        }
    }

    fn seen(&self) -> Vec<InvalidToolCallContext> {
        self.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl AgentHook for Decide {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        context: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        self.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(context.clone());
        self.action.clone()
    }
}

/// The run's output, or its error as text.
async fn drive(surface: Surface, runner: AgentRunner) -> Result<String, String> {
    match surface {
        Surface::Blocking => runner
            .run()
            .await
            .map(|response| response.output())
            .map_err(|error| error.to_string()),
        Surface::Streamed => {
            let mut stream = runner.stream();
            let mut output = None;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                        output = Some(response.output());
                    }
                    Ok(_) => {}
                    Err(error) => return Err(error.to_string()),
                }
            }
            output.ok_or_else(|| "no final response".to_owned())
        }
    }
}

/// Run `script` on `surface` with `configure` applied to the runner.
async fn run(
    surface: Surface,
    script: &[Turn],
    configure: impl FnOnce(AgentRunner) -> AgentRunner,
) -> (Result<String, String>, Vec<CompletionRequest>) {
    let model = model(surface, script);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();
    let runner = configure(agent.prompt("add 2 and something").max_turns(10));
    let outcome = drive(surface, runner).await;
    (outcome, recorded.requests())
}

/// The result answering the call of model turn `turn` in `history`.
fn answer(history: &[Message], turn: usize) -> Option<&ToolResult> {
    let id = call_id(turn);
    history.iter().find_map(|message| match message {
        Message::User { content } => content.iter().find_map(|item| match item {
            UserContent::ToolResult(result)
                if result.call.provider().map(|id| id.as_str()) == Some(id.as_str()) =>
            {
                Some(result)
            }
            _ => None,
        }),
        Message::System { .. } | Message::Assistant(_) => None,
    })
}

fn answer_text(requests: &[CompletionRequest], turn: usize) -> String {
    let request = requests
        .get(turn)
        .unwrap_or_else(|| panic!("request {} was made", turn + 1));
    let result = answer(&request.chat_history, turn)
        .unwrap_or_else(|| panic!("call {turn} is answered: {:?}", request.chat_history));
    assert!(result.is_error, "the answer is an error result: {result:?}");
    result
        .content
        .iter()
        .map(|content| match content {
            ToolResultContent::Text(text) => text.text.clone(),
            other => panic!("a text answer, not {other:?}"),
        })
        .collect()
}

#[tokio::test]
async fn by_default_a_malformed_call_is_answered_with_feedback_and_the_run_continues() {
    for surface in SURFACES {
        let (outcome, requests) = run(surface, &[Turn::Malformed, Turn::Answer], |r| r).await;
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(
            requests.len(),
            2,
            "{surface:?}: the model gets another turn"
        );
        let text = answer_text(&requests, 1);
        assert!(
            text.starts_with("The arguments for tool `add` are not a JSON object: ")
                && text.contains(RAW),
            "{surface:?}: the answer names the problem and the arguments: {text}"
        );
    }
}

#[tokio::test]
async fn the_hook_sees_the_reason_and_the_raw_arguments() {
    for surface in SURFACES {
        let hook = Decide::new(None);
        let (outcome, requests) = run(surface, &[Turn::Malformed, Turn::Answer], |r| {
            r.add_hook(hook.clone())
        })
        .await;
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        let seen = hook.seen();
        let [context] = seen.as_slice() else {
            panic!("{surface:?}: the hook is consulted once, saw {seen:?}");
        };
        assert_eq!(context.tool_name, "add");
        assert_eq!(context.args.as_deref(), Some(RAW));
        assert!(
            matches!(
                &context.reason,
                InvalidToolCallReason::MalformedArguments { error } if !error.is_empty()
            ),
            "{surface:?}: {:?}",
            context.reason
        );
        assert_eq!(
            context
                .tool_call_id
                .as_ref()
                .and_then(|id| id.provider())
                .map(|id| id.as_str()),
            Some("tool_call_1")
        );
        assert_eq!(context.available_tools, ["add"]);
        assert_eq!(context.allowed_tools, ["add"]);
        assert_eq!(
            context.is_streaming,
            matches!(surface, Surface::Streamed),
            "{surface:?}"
        );
        assert!(
            context.chat_history.iter().any(|message| matches!(
                message,
                Message::Assistant(AssistantMessage { content, .. })
                    if content.iter().any(|item| matches!(item, AssistantContent::ToolCall(_)))
            )),
            "{surface:?}: the diagnostic history holds the committed turn"
        );
        // No action leaves the default feedback.
        assert!(answer_text(&requests, 1).contains("not a JSON object"));
    }
}

#[tokio::test]
async fn retry_answers_with_the_hooks_feedback() {
    for surface in SURFACES {
        let hook = Decide::new(Some(InvalidToolCallAction::retry(
            "arguments were not JSON; try again",
        )));
        let (outcome, requests) = run(surface, &[Turn::Malformed, Turn::Answer], |r| {
            r.add_hook(hook.clone())
        })
        .await;
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(
            answer_text(&requests, 1),
            "arguments were not JSON; try again"
        );
    }
}

#[tokio::test]
async fn skip_answers_with_a_skipped_result() {
    for surface in SURFACES {
        let hook = Decide::new(Some(InvalidToolCallAction::skip("add: skipped")));
        let (outcome, requests) = run(surface, &[Turn::Malformed, Turn::Answer], |r| {
            r.add_hook(hook.clone())
        })
        .await;
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(answer_text(&requests, 1), "add: skipped");
    }
}

#[tokio::test]
async fn stop_ends_the_run_with_the_hooks_reason() {
    for surface in SURFACES {
        let hook = Decide::new(Some(InvalidToolCallAction::stop("operator halted")));
        let (outcome, requests) = run(surface, &[Turn::Malformed, Turn::Answer], |r| {
            r.add_hook(hook.clone())
        })
        .await;
        let error = outcome.expect_err("stop ends the run");
        assert!(error.contains("operator halted"), "{surface:?}: {error}");
        assert_eq!(requests.len(), 1, "{surface:?}");
    }
}

/// `Fail` ends the run naming the tool and the parse error, and `Repair` is
/// refused the same way: renaming the tool cannot fix its arguments.
#[tokio::test]
async fn fail_and_repair_end_the_run_naming_the_tool_and_the_parse_error() {
    for action in [
        InvalidToolCallAction::fail(),
        InvalidToolCallAction::repair("add"),
    ] {
        for surface in SURFACES {
            let hook = Decide::new(Some(action.clone()));
            let (outcome, requests) = run(surface, &[Turn::Malformed, Turn::Answer], |r| {
                r.add_hook(hook.clone())
            })
            .await;
            let error = outcome.expect_err("the run fails");
            assert!(
                error.contains("tool `add` was called with arguments that are not a JSON object: ")
                    && error.contains("at line 1 column"),
                "{surface:?} {action:?}: {error}"
            );
            assert_eq!(requests.len(), 1, "{surface:?} {action:?}");
        }
    }
}

#[tokio::test]
async fn without_a_limit_malformed_arguments_are_answered_until_max_turns() {
    let script = [Turn::Malformed; 10];
    for surface in SURFACES {
        let (outcome, requests) = run(surface, &script, |r| r).await;
        let error = outcome.expect_err("the turn budget ends the run");
        assert!(
            error.contains("reached the max turns limit of 10"),
            "{surface:?}: {error}"
        );
        assert_eq!(requests.len(), 10, "{surface:?}");
        for turn in 1..10 {
            assert!(
                answer_text(&requests, turn).contains("not a JSON object"),
                "{surface:?}: call {turn} is answered with feedback"
            );
        }
    }
}

#[tokio::test]
async fn a_model_that_keeps_sending_malformed_arguments_stops_after_the_limit() {
    let script = [Turn::Malformed; 8];
    for surface in SURFACES {
        let hook = Decide::new(Some(InvalidToolCallAction::retry("again")));
        let (outcome, requests) = run(surface, &script, |r| {
            r.max_consecutive_malformed_tool_calls(3)
                .add_hook(hook.clone())
        })
        .await;
        let error = outcome.expect_err("the limit ends the run");
        assert!(
            error.contains("tool `add` was called with arguments that are not a JSON object on 4 consecutive turns")
                && !error.contains("max turns"),
            "{surface:?}: {error}"
        );
        assert_eq!(
            requests.len(),
            4,
            "{surface:?}: three retries, then the limit"
        );
        assert_eq!(hook.seen().len(), 3, "{surface:?}: hook retries count");
    }
}

#[tokio::test]
async fn the_limit_is_configurable() {
    for surface in SURFACES {
        let (outcome, requests) = run(surface, &[Turn::Malformed; 3], |r| {
            r.max_consecutive_malformed_tool_calls(1)
        })
        .await;
        let error = outcome.expect_err("the limit ends the run");
        assert!(
            error.contains("on 2 consecutive turns"),
            "{surface:?}: {error}"
        );
        assert_eq!(requests.len(), 2, "{surface:?}");
    }
}

/// `None` clears a limit set earlier, on the runner and on the run.
#[tokio::test]
async fn none_clears_the_limit() {
    let script = [
        Turn::Malformed,
        Turn::Malformed,
        Turn::Malformed,
        Turn::Answer,
    ];
    for surface in SURFACES {
        let (outcome, requests) = run(surface, &script, |r| {
            r.max_consecutive_malformed_tool_calls(1)
                .max_consecutive_malformed_tool_calls(None)
        })
        .await;
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(requests.len(), 4, "{surface:?}");
    }
    let run = AgentRun::new("add")
        .max_consecutive_malformed_tool_calls(1)
        .max_consecutive_malformed_tool_calls(None);
    let json = serde_json::to_value(&run).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(json["max_consecutive_malformed_tool_calls"], json!(null));
}

#[tokio::test]
async fn a_well_formed_call_between_malformed_ones_resets_the_count() {
    use Turn::{Answer, Malformed, WellFormed};
    let script = [
        Malformed, Malformed, Malformed, WellFormed, Malformed, Malformed, Malformed, Answer,
    ];
    for surface in SURFACES {
        let (outcome, requests) = run(surface, &script, |r| {
            r.max_consecutive_malformed_tool_calls(3)
        })
        .await;
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(requests.len(), 8, "{surface:?}");
        assert!(!answer(&requests[4].chat_history, 4).is_some_and(|result| result.is_error));
    }
}

/// `Ignore` governs unknown and disallowed names only; it does not lift a
/// malformed-arguments limit.
#[tokio::test]
async fn ignore_does_not_lift_the_limit() {
    let mut script = vec![Turn::Malformed; 6];
    script.push(Turn::Answer);
    for surface in SURFACES {
        let (outcome, requests) = run(surface, &script, |r| {
            r.max_consecutive_malformed_tool_calls(3)
                .unhandled_invalid_tool_call(UnhandledInvalidToolCall::Ignore)
        })
        .await;
        let error = outcome.expect_err("the limit ends the run");
        assert!(
            error.contains("on 4 consecutive turns"),
            "{surface:?}: {error}"
        );
        assert_eq!(requests.len(), 4, "{surface:?}");
    }
}

/// A run persisted after two malformed turns resumes with its count and
/// limit: two more malformed turns pass its limit of three.
#[tokio::test]
async fn the_count_survives_a_serialize_and_resume() {
    let mut run = AgentRun::new("add 2 and something")
        .max_turns(10)
        .max_consecutive_malformed_tool_calls(3);
    for turn in 1..=2 {
        assert!(matches!(
            run.next_step(),
            Ok(AgentRunStep::CallModel { .. })
        ));
        let call = ToolCall::from_wire(call_id(turn), ToolFunction::parse(add(), RAW));
        let outcome = run.model_response(ModelTurn::new(
            AssistantMessage::default(),
            vec![AssistantContent::ToolCall(call.clone())],
            Usage::default(),
            TurnPolicy::new(["add".to_owned()].into(), None, None).expect("policy"),
            json!({}),
        ));
        assert!(matches!(outcome, Ok(ModelTurnOutcome::Continue { .. })));
        assert!(matches!(
            run.next_step(),
            Ok(AgentRunStep::CallTools { .. })
        ));
        run.tool_results(vec![UserContent::ToolResult(
            call.error_result(vec![ToolResultContent::text("not a JSON object")]),
        )])
        .unwrap_or_else(|error| panic!("{error}"));
    }
    let saved = serde_json::to_string(&run).unwrap_or_else(|error| panic!("{error}"));

    for surface in SURFACES {
        let restored: AgentRun =
            serde_json::from_str(&saved).unwrap_or_else(|error| panic!("{error}"));
        let model = model(surface, &[Turn::Malformed, Turn::Malformed, Turn::Answer]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();
        let error = drive(surface, agent.resume(restored))
            .await
            .expect_err("the persisted count reaches the limit");
        assert!(
            error.contains("on 4 consecutive turns"),
            "{surface:?}: {error}"
        );
        assert_eq!(recorded.request_count(), 2, "{surface:?}");
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
#[allow(dead_code)]
struct Answer {
    answer: String,
}

/// Malformed arguments to the structured-output tool keep their own
/// reprompt budget: the invalid-call hook is not consulted.
#[tokio::test]
async fn the_structured_output_path_is_unchanged() {
    for surface in SURFACES {
        let output_raw = "{\"answer\":";
        let model = match surface {
            Surface::Blocking => MockCompletionModel::from_turns([
                MockTurn::from_content(AssistantContent::ToolCall(ToolCall::from_wire(
                    "out1",
                    ToolFunction::parse(
                        ToolName::new("final_result").unwrap_or_else(|error| panic!("{error}")),
                        output_raw,
                    ),
                ))),
                MockTurn::tool_call("out2", "final_result", json!({"answer": "done"})),
            ]),
            Surface::Streamed => MockCompletionModel::from_stream_turns([
                vec![
                    MockStreamEvent::tool_call_name_delta("out1", "final_result"),
                    MockStreamEvent::tool_call_arguments_delta("out1", output_raw),
                    MockStreamEvent::tool_call_end("out1"),
                    MockStreamEvent::final_response_with_total_tokens(4),
                ],
                vec![
                    MockStreamEvent::tool_call("out2", "final_result", json!({"answer": "done"})),
                    MockStreamEvent::final_response_with_total_tokens(4),
                ],
            ]),
        };
        let recorded = model.clone();
        let hook = Decide::new(Some(InvalidToolCallAction::fail()));
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .output_schema::<Answer>()
            .output_mode(OutputMode::Tool)
            .build();
        let runner = agent
            .prompt("answer")
            .max_turns(3)
            .max_consecutive_malformed_tool_calls(0)
            .add_hook(hook.clone());
        let output = drive(surface, runner)
            .await
            .unwrap_or_else(|error| panic!("{surface:?}: {error}"));
        assert!(output.contains("done"), "{surface:?}: {output}");
        assert!(
            hook.seen().is_empty(),
            "{surface:?}: the hook is not consulted"
        );
        let requests = recorded.requests();
        assert_eq!(requests.len(), 2, "{surface:?}");
        let reprompt = serde_json::to_string(&requests[1].chat_history)
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(
            reprompt.contains("not a JSON object"),
            "{surface:?}: {reprompt}"
        );
    }
}

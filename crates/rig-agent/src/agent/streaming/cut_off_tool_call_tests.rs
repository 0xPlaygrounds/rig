//! A streamed reply that reached the output limit inside a tool call leaves
//! that call cut off. The call never runs, whatever its arguments look like,
//! and the model reads that its reply reached the limit, so it sends less
//! rather than the same call again. A cut-off call is not a malformed one.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use futures::StreamExt;
use rig_core::completion::{CompletionRequest, FinishReason, Usage};
use rig_core::message::{Message, ToolResultContent, UserContent};
use rig_core::operation::Finish;
use serde::Deserialize;
use serde_json::json;

use super::MultiTurnStreamItem;
use crate::agent::hook::{AgentHook, HookContext};
use crate::agent::{AgentBuilder, AgentRunner, InvalidToolCallAction, InvalidToolCallContext};
use crate::run::OutputMode;
use crate::test_utils::{MockCompletionModel, MockStreamEvent, mock_final};
use crate::tool::{Tool, ToolContext, ToolExecutionError};
use rig_core::structured_output::OUTPUT_TOOL_NAME;

const ID: &str = "call_1";

/// A file-writing tool whose arguments are all optional, so `{}` is a call
/// it accepts. It counts the times it ran.
#[derive(Clone, Default)]
struct WriteFile(Arc<AtomicU32>);

#[derive(Deserialize)]
struct WriteArgs {
    #[serde(default)]
    _path: Option<String>,
    #[serde(default)]
    _content: Option<String>,
}

impl Tool for WriteFile {
    const NAME: &'static str = "write_file";
    type Error = ToolExecutionError;
    type Args = WriteArgs;
    type Output = String;

    fn description(&self) -> String {
        "Write a file".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            }
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok("written".into())
    }
}

/// How far the call got before the reply reached the output limit.
#[derive(Clone, Copy, Debug)]
enum Cut {
    /// Inside the arguments.
    Midway,
    /// Before any argument text.
    Empty,
    /// After a whole arguments object, the call never closed.
    Unclosed,
}

const CUTS: [Cut; 3] = [Cut::Midway, Cut::Empty, Cut::Unclosed];

/// A reply that ends at the output limit inside a call to `tool`, `cut`
/// far into its arguments.
fn cut_turn(tool: &str, cut: Cut) -> Vec<MockStreamEvent> {
    let mut events = vec![MockStreamEvent::tool_call_name_delta(ID, tool)];
    match cut {
        Cut::Midway => events.push(MockStreamEvent::tool_call_arguments_delta(
            ID,
            r#"{"path": "a.rs", "content": "fn ma"#,
        )),
        Cut::Empty => {}
        Cut::Unclosed => events.push(MockStreamEvent::tool_call_arguments_delta(
            ID,
            r#"{"path": "a.rs", "content": "fn main() {}"}"#,
        )),
    }
    events.push(MockStreamEvent::FinalResponse(Finish {
        reason: Some(FinishReason::Length),
        ..mock_final(Usage::default())
    }));
    events
}

/// A reply that ends at the output limit inside a `write_file` call, then
/// a reply that answers.
fn model(cut: Cut) -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([
        cut_turn("write_file", cut),
        vec![
            MockStreamEvent::text("split it"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ])
}

/// Stream `runner` to its end: its output or its error as text.
async fn drive(runner: AgentRunner) -> Result<String, String> {
    let mut stream = runner.stream();
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(response)) => return Ok(response.output()),
            Ok(_) => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    Err("no final response".to_owned())
}

/// Stream the run to its end: its output or its error as text, the
/// requests the model saw, and how many times the tool ran.
async fn run(
    cut: Cut,
    configure: impl FnOnce(AgentRunner) -> AgentRunner,
) -> (Result<String, String>, Vec<CompletionRequest>, u32) {
    let model = model(cut);
    let recorded = model.clone();
    let tool = WriteFile::default();
    let ran = tool.0.clone();
    let agent = AgentBuilder::new(model).tool(tool).build();
    let outcome = drive(configure(agent.prompt("write a long file").max_turns(5))).await;
    (outcome, recorded.requests(), ran.load(Ordering::SeqCst))
}

/// The text of the result answering the cut-off call in the second request.
fn answer(requests: &[CompletionRequest]) -> Option<String> {
    requests
        .get(1)?
        .chat_history
        .iter()
        .find_map(|message| match &**message {
            Message::User { content } => content.iter().find_map(|item| match item {
                UserContent::ToolResult(result) => Some(
                    result
                        .content
                        .iter()
                        .filter_map(|content| match content {
                            ToolResultContent::Text(text) => Some(text.text.as_str()),
                            _ => None,
                        })
                        .collect(),
                ),
                _ => None,
            }),
            Message::System { .. } | Message::Assistant(_) => None,
        })
}

#[tokio::test]
async fn a_call_cut_at_the_output_limit_never_runs_and_is_answered_as_cut_off() {
    for cut in CUTS {
        let (outcome, requests, ran) = run(cut, |runner| runner).await;
        assert_eq!(outcome.as_deref(), Ok("split it"), "{cut:?}");
        assert_eq!(ran, 0, "{cut:?}: the cut-off call never runs");
        let answer = answer(&requests);
        assert!(
            answer
                .as_deref()
                .is_some_and(|text| text.contains("reached the output limit")),
            "{cut:?}: the model reads that its reply reached the limit: {answer:?}"
        );
    }
}

#[tokio::test]
async fn a_cut_off_call_does_not_use_up_the_malformed_call_allowance() {
    for cut in CUTS {
        let (outcome, _, ran) =
            run(cut, |runner| runner.max_consecutive_malformed_tool_calls(0)).await;
        assert_eq!(outcome.as_deref(), Ok("split it"), "{cut:?}");
        assert_eq!(ran, 0, "{cut:?}: the cut-off call never runs");
    }
}

/// The same cut on a real wire, DeepSeek's chat stream over a scripted
/// transport: inside the arguments, and before any came.
#[tokio::test]
async fn a_call_a_real_wire_cut_at_the_output_limit_never_runs() {
    use rig_core::providers::openai::{OpenAIConfig, wire::DEEPSEEK};
    use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};

    let sse = |delta: serde_json::Value, finish: &str| {
        let chunk = json!({"id": "c", "object": "chat.completion.chunk", "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        MockHttpResponse::success_typed(
            format!("data: {chunk}\n\ndata: [DONE]\n\n"),
            "text/event-stream",
        )
    };
    for arguments in [r#"{"path": "a.rs", "content": "fn ma"#, ""] {
        let cut = sse(
            json!({"role": "assistant", "tool_calls": [{"index": 0, "id": ID, "type": "function",
                "function": {"name": "write_file", "arguments": arguments}}]}),
            "length",
        );
        let done = sse(json!({"role": "assistant", "content": "split it"}), "stop");
        let http = SequencedHttpClient::new([cut, done]);
        let model = OpenAIConfig::with_key(&DEEPSEEK, "key")
            .connect(http.clone())
            .completion("deepseek-v4-flash");
        let tool = WriteFile::default();
        let ran = tool.0.clone();
        let agent = AgentBuilder::new(model).tool(tool).build();
        let mut stream = agent.prompt("write a long file").max_turns(5).stream();
        let mut outcome = Err("no final response".to_owned());
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    outcome = Ok(response.output());
                }
                Ok(_) => {}
                Err(error) => {
                    outcome = Err(error.to_string());
                    break;
                }
            }
        }
        assert_eq!(outcome.as_deref(), Ok("split it"), "{arguments:?}");
        assert_eq!(ran.load(Ordering::SeqCst), 0, "{arguments:?}: never runs");
        let bodies: Vec<String> = http
            .requests()
            .iter()
            .map(|request| String::from_utf8_lossy(&request.body).into_owned())
            .collect();
        assert!(
            matches!(bodies.as_slice(), [_, next]
                if next.contains("Not run: your reply reached the output limit")),
            "{arguments:?}: {bodies:?}"
        );
    }
}

/// A structured answer whose fields are all optional, so `{}` satisfies it.
#[derive(Deserialize, schemars::JsonSchema)]
struct Note {
    #[serde(default)]
    _path: Option<String>,
    #[serde(default)]
    _content: Option<String>,
}

/// A cut call to the output tool is never the run's answer, whatever its
/// arguments: the model reads that its reply reached the limit, and its
/// next, whole answer is the run's.
#[tokio::test]
async fn a_cut_output_tool_call_is_never_the_answer() {
    for cut in CUTS {
        let whole = json!({"_path": "a.rs", "_content": "done"});
        let model = MockCompletionModel::from_stream_turns([
            cut_turn(OUTPUT_TOOL_NAME, cut),
            vec![
                MockStreamEvent::tool_call("call_2", OUTPUT_TOOL_NAME, whole),
                MockStreamEvent::final_response_with_default_usage(),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(WriteFile::default())
            .output_schema::<Note>()
            .output_mode(OutputMode::Tool)
            .build();
        let outcome = drive(agent.prompt("write a long file").max_turns(5)).await;
        assert!(
            outcome
                .as_deref()
                .is_ok_and(|output| output.contains("done")),
            "{cut:?}: the whole answer is the run's: {outcome:?}"
        );
        let requests = recorded.requests();
        let answer = answer(&requests);
        assert!(
            answer
                .as_deref()
                .is_some_and(|text| text.contains("reached the output limit")),
            "{cut:?}: the model reads that its reply reached the limit: {answer:?}"
        );
    }
}

/// Records every invalid call it is offered and leaves the decision to the
/// run.
#[derive(Clone, Default)]
struct Offered(Arc<std::sync::Mutex<Vec<String>>>);

impl AgentHook for Offered {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        context: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(context.tool_name.clone());
        None
    }
}

/// A cut call naming a tool the turn does not offer is answered as cut
/// off like any other: it is not offered to the invalid-call hook, and the
/// run goes on.
#[tokio::test]
async fn a_cut_call_to_an_unknown_tool_is_answered_as_cut_off() {
    for cut in CUTS {
        let model = MockCompletionModel::from_stream_turns([
            cut_turn("delete_everything", cut),
            vec![
                MockStreamEvent::text("split it"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
        ]);
        let recorded = model.clone();
        let offered = Offered::default();
        let agent = AgentBuilder::new(model).tool(WriteFile::default()).build();
        let runner = agent
            .prompt("write a long file")
            .max_turns(5)
            .add_hook(offered.clone());
        let outcome = drive(runner).await;
        assert_eq!(outcome.as_deref(), Ok("split it"), "{cut:?}");
        let seen = offered
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert!(
            seen.is_empty(),
            "{cut:?}: not offered to the hook: {seen:?}"
        );
        let answer = answer(&recorded.requests());
        assert!(
            answer
                .as_deref()
                .is_some_and(|text| text.contains("reached the output limit")),
            "{cut:?}: {answer:?}"
        );
    }
}

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use futures::StreamExt;
use rig_harness::prelude::Processes;
use rig_harness::rig_core::completion::{
    AssistantContent, CompletionRequest, CompletionResponse, FinishReason, Message, ToolDefinition,
    Usage,
};
use rig_harness::rig_core::effect::{Answered, EffectId, EffectKind};
use rig_harness::rig_core::message::{
    AssistantMessage, ToolCall, ToolFunction, ToolName, ToolResultContent,
};
use rig_harness::rig_core::serve::{Dispatch, Reply, Serve};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use super::{Bridge, Model};
use crate::harness::{Event, Harness, Launch, Part, Running};
use crate::mcp::tests::{post, request};

/// What the scripted harness saw: each start's system prompt and prompts.
#[derive(Default)]
struct Seen {
    started: Vec<(String, Vec<String>)>,
    /// The tool names of each listing a harness made once told the tools
    /// changed.
    listed: Vec<Vec<String>>,
}

/// A harness whose model does what its prompt says. A prompt containing
/// `call read <path>` calls Rig's `read` tool once for each such phrase, at
/// once, and then says what came back; any other prompt is echoed.
#[derive(Default)]
struct Scripted {
    seen: Arc<Mutex<Seen>>,
    /// How many started harnesses were dropped.
    ended: Arc<AtomicUsize>,
    /// Whether the harness holds a stream open for the server's
    /// notifications and lists the tools again when told they changed.
    listens: bool,
    /// What the harness says of taking tools while it runs.
    fixed_tools: bool,
}

/// Holds a notification stream open on the server at `url` and lists the
/// tools again, into `seen`, each time it says they changed.
async fn listen(url: String, token: String, seen: Arc<Mutex<Seen>>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let address = url
        .trim_start_matches("http://")
        .trim_end_matches("/mcp")
        .to_owned();
    let Ok(mut stream) = tokio::net::TcpStream::connect(&address).await else {
        return;
    };
    let get = format!(
        "GET /mcp HTTP/1.1\r\nHost: {address}\r\nAuthorization: Bearer {token}\r\n\
         Accept: text/event-stream\r\n\r\n"
    );
    if stream.write_all(get.as_bytes()).await.is_err() {
        return;
    }
    let mut chunk = [0u8; 4096];
    while let Ok(read) = stream.read(&mut chunk).await {
        let told = chunk
            .get(..read)
            .map(String::from_utf8_lossy)
            .unwrap_or_default();
        if read == 0 {
            return;
        }
        if told.contains("tools/list_changed") {
            let (_, listed) = post(&url, &token, &request(99, "tools/list", json!({})))
                .await
                .unwrap_or_default();
            let tools = listed.pointer("/result/tools").and_then(Value::as_array);
            let names = tools
                .into_iter()
                .flatten()
                .filter_map(|tool| tool.get("name")?.as_str());
            let names = names.map(str::to_owned).collect();
            seen.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .listed
                .push(names);
        }
    }
}

struct Ended(Arc<AtomicUsize>);

impl Drop for Ended {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl Harness for Scripted {
    fn note(&self, _: Option<&std::path::Path>) -> String {
        "Tools are `x__read`.".to_owned()
    }

    fn relists(&self) -> bool {
        !self.fixed_tools
    }

    fn start(&self, launch: Launch<'_>) -> Result<Running, String> {
        let (prompts, mut prompted) = mpsc::unbounded_channel::<Vec<Part>>();
        let (events, reported) = mpsc::unbounded_channel();
        let seen = self.seen.clone();
        let index = {
            let mut seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
            seen.started.push((launch.system.to_owned(), Vec::new()));
            seen.started.len() - 1
        };
        let (url, token) = (launch.mcp.url().to_owned(), launch.mcp.token().to_owned());
        if self.listens {
            tokio::spawn(listen(url.clone(), token.clone(), seen.clone()));
        }
        tokio::spawn(async move {
            while let Some(prompt) = prompted.recv().await {
                let text: String = prompt
                    .iter()
                    .map(|part| match part {
                        Part::Text(text) => text.as_str(),
                        Part::Image { .. } => "[image]",
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if let Some((_, prompts)) = seen
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .started
                    .get_mut(index)
                {
                    prompts.push(text.clone());
                }
                let asked = text.rsplit("</conversation>").next().unwrap_or(&text);
                let paths: Vec<&str> = asked
                    .split("call read ")
                    .skip(1)
                    .filter_map(|rest| rest.split_whitespace().next())
                    .collect();
                let _ = events.send(Event::Began);
                let _ = events.send(Event::Reasoning("thinking".to_owned()));
                if paths.is_empty() {
                    let _ = events.send(Event::Text(format!("echo: {asked}")));
                } else {
                    let _ = events.send(Event::Text("Reading.".to_owned()));
                    let calls = paths.iter().enumerate().map(|(id, path)| {
                        let call = request(
                            id as u64,
                            "tools/call",
                            json!({ "name": "read", "arguments": { "path": path } }),
                        );
                        let (url, token) = (url.clone(), token.clone());
                        async move { post(&url, &token, &call).await }
                    });
                    // As Claude Code does, the tools are called before the
                    // inference that asked for them is counted.
                    let counted = events.clone();
                    tokio::spawn(async move {
                        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                        let mut usage = Usage::new();
                        usage.input_tokens = Some(7);
                        let _ = counted.send(Event::Usage(usage));
                    });
                    let results = futures::future::join_all(calls).await;
                    let _ = events.send(Event::Began);
                    let said: Vec<String> = results
                        .into_iter()
                        .map(|result| {
                            let (_, response) = result.unwrap_or_default();
                            let text = response.pointer("/result/content/0/text");
                            let error = response.pointer("/result/isError") == Some(&json!(true));
                            let text = text.and_then(Value::as_str).unwrap_or("nothing");
                            format!("{}{text}", if error { "error: " } else { "" })
                        })
                        .collect();
                    let _ = events.send(Event::Text(format!("read said: {}", said.join(" | "))));
                }
                let mut usage = Usage::new();
                usage.input_tokens = Some(10);
                usage.output_tokens = Some(2);
                let _ = events.send(Event::Usage(usage));
                let _ = events.send(Event::Done(Ok(())));
            }
        });
        Ok(Running {
            prompts,
            events: reported,
            _alive: Box::new(Ended(self.ended.clone())),
        })
    }
}

/// A model on a scripted harness, with what the harness saw and how many
/// of its processes ended.
fn model(limit: usize) -> (Model, Arc<Mutex<Seen>>, Arc<AtomicUsize>) {
    on(Scripted::default(), limit)
}

/// A model on `harness`.
fn on(harness: Scripted, limit: usize) -> (Model, Arc<Mutex<Seen>>, Arc<AtomicUsize>) {
    let (seen, ended) = (harness.seen.clone(), harness.ended.clone());
    let bridge = Bridge::new(
        tokio::runtime::Handle::current(),
        Processes::default(),
        limit,
        None,
    );
    let model = Model {
        bridge: Arc::new(bridge),
        harness: Arc::new(harness),
        program: "scripted".into(),
        vendor: "scripted",
        model: "model-1".to_owned(),
    };
    (model, seen, ended)
}

fn read_tool() -> Option<ToolDefinition> {
    let name = ToolName::new("read").ok()?;
    Some(ToolDefinition::new(
        name,
        "Reads a file.",
        json!({ "type": "object" }),
    ))
}

/// A request for `history` under the system prompt, offering `read`.
fn asks(system: &str, history: &[Message]) -> CompletionRequest {
    CompletionRequest::new(Message::system(system))
        .messages(history.iter().cloned())
        .tools(read_tool())
}

async fn reply(model: &Model, request: CompletionRequest) -> Reply {
    let kind = EffectKind::Completion {
        request: Arc::new(request),
        stream: true,
    };
    model
        .serve(kind, Dispatch::new(EffectId::from_raw(1), true))
        .await
}

/// The response to `request`, if the model answered.
async fn answer(model: &Model, request: CompletionRequest) -> Option<Arc<CompletionResponse>> {
    match reply(model, request).await.into_outcome().await {
        Ok(Answered::Completion(response)) => Some(response),
        _ => None,
    }
}

/// A response's text, the tools it called and why it stopped.
fn said(response: Option<&Arc<CompletionResponse>>) -> (String, Vec<String>, Option<FinishReason>) {
    let Some(response) = response else {
        return Default::default();
    };
    let calls = response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(format!(
                "{}({})",
                call.function.name.as_str(),
                call.function.arguments_json(),
            )),
            _ => None,
        })
        .collect();
    (
        AssistantContent::concat_text(&response.choice),
        calls,
        response.finish_reason(),
    )
}

/// The assistant message a response becomes in the history.
fn turn(response: Option<&Arc<CompletionResponse>>) -> Message {
    let content = response
        .map(|response| response.choice.clone())
        .unwrap_or_default();
    Message::Assistant(AssistantMessage::new(content))
}

/// The user message that answers each tool call of `response` with what
/// `result` says for its arguments.
fn results(response: Option<&Arc<CompletionResponse>>, result: impl Fn(&str) -> String) -> Message {
    let calls = response.into_iter().flat_map(|response| &response.choice);
    let results = calls
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.result(vec![ToolResultContent::Text(
                result(&call.function.arguments_json()).into(),
            )])),
            _ => None,
        })
        .collect();
    Message::tool_results(results)
}

fn prompts(seen: &Arc<Mutex<Seen>>) -> Vec<Vec<String>> {
    let seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
    seen.started
        .iter()
        .map(|(_, prompts)| prompts.clone())
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tool_call_is_rig_s_to_run_and_its_result_continues_the_same_harness_turn() {
    let (model, seen, ended) = model(4);
    let mut history = vec![Message::user("call read a.txt")];

    let called = answer(&model, asks("Be brief.", &history)).await;
    history.push(turn(called.as_ref()));
    history.push(results(called.as_ref(), |_| "alpha".to_owned()));
    let finished = answer(&model, asks("Be brief.", &history)).await;

    assert_eq!(
        said(called.as_ref()),
        (
            "Reading.".to_owned(),
            vec![r#"read({"path":"a.txt"})"#.to_owned()],
            Some(FinishReason::ToolCalls),
        ),
    );
    assert_eq!(
        said(finished.as_ref()),
        (
            "read said: alpha".to_owned(),
            Vec::new(),
            Some(FinishReason::Stop)
        ),
    );
    // Each reply carries the counters of its own inference.
    assert_eq!(
        called.map(|response| response.usage.input_tokens),
        Some(Some(7))
    );
    assert_eq!(
        finished.map(|response| response.usage.input_tokens),
        Some(Some(10))
    );
    assert_eq!(prompts(&seen), vec![vec!["call read a.txt".to_owned()]]);
    assert_eq!(ended.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn calls_made_together_are_one_reply_and_each_gets_its_own_result() {
    let (model, _, _) = model(4);
    let mut history = vec![Message::user("call read a.txt and call read b.txt")];

    let called = answer(&model, asks("", &history)).await;
    history.push(turn(called.as_ref()));
    // Answered in the other order: a result goes to the call it names.
    let Message::User { content } = results(called.as_ref(), |arguments| format!("<{arguments}>"))
    else {
        return;
    };
    history.push(Message::User {
        content: content.into_iter().rev().collect(),
    });
    let finished = answer(&model, asks("", &history)).await;

    assert_eq!(said(called.as_ref()).1.len(), 2);
    assert_eq!(
        said(finished.as_ref()).0,
        r#"read said: <{"path":"a.txt"}> | <{"path":"b.txt"}>"#,
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_next_user_message_goes_to_the_same_harness() {
    let (model, seen, ended) = model(4);
    let mut history = vec![Message::user("one")];

    let first = answer(&model, asks("Be brief.", &history)).await;
    history.push(turn(first.as_ref()));
    history.push(Message::user("two"));
    let second = answer(&model, asks("Be brief.", &history)).await;

    assert_eq!(said(first.as_ref()).0, "echo: one");
    assert_eq!(said(second.as_ref()).0, "echo: two");
    assert_eq!(
        prompts(&seen),
        vec![vec!["one".to_owned(), "two".to_owned()]]
    );
    assert_eq!(ended.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_history_the_harness_did_not_answer_starts_a_new_one_with_it_written_out() {
    let (model, seen, _) = model(4);
    let Ok(read) = ToolName::new("read") else {
        return;
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(read, json!({ "path": "a.txt" })));
    let result = call.result(vec![ToolResultContent::Text("alpha".into())]);
    let earlier = [
        Message::user("call read a.txt"),
        Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::text("Reading."),
            AssistantContent::ToolCall(call),
        ])),
        Message::tool_results(vec![result]),
        Message::assistant("It says alpha."),
        Message::user("and now?"),
    ];

    let answered = answer(&model, asks("Be brief.", &earlier)).await;

    let opening = prompts(&seen)
        .into_iter()
        .flatten()
        .next()
        .unwrap_or_default();
    assert!(
        said(answered.as_ref()).0.ends_with("and now?"),
        "{:?}",
        said(answered.as_ref())
    );
    for expected in [
        "<conversation>\n## User\ncall read a.txt\n",
        "## Assistant\nReading.\n[called `read` with {\"path\":\"a.txt\"}]\n",
        "[result of `read`]\nalpha\n",
        "## Assistant\nIt says alpha.\n\n</conversation>\n\nand now?",
    ] {
        assert!(
            opening.contains(expected),
            "{expected:?} is not in {opening:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn another_system_prompt_or_tool_set_is_another_harness() {
    let (model, seen, _) = model(4);
    let mut history = vec![Message::user("one")];
    let first = answer(&model, asks("Be brief.", &history)).await;
    history.push(turn(first.as_ref()));
    history.push(Message::user("two"));

    let _ = answer(&model, asks("Be thorough.", &history)).await;
    let untooled =
        CompletionRequest::new(Message::system("Be brief.")).messages(history.iter().cloned());
    let _ = answer(&model, untooled).await;

    let systems: Vec<String> = {
        let seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
        seen.started
            .iter()
            .map(|(system, _)| system.clone())
            .collect()
    };
    // Each harness is told its own note after the system prompt.
    assert_eq!(
        systems,
        [
            "Be brief.\n\nTools are `x__read`.",
            "Be thorough.\n\nTools are `x__read`.",
            "Be brief.\n\nTools are `x__read`.",
        ],
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dropped_reply_ends_its_harness() {
    let (model, _, ended) = model(4);

    let Reply::Stream(mut stream) =
        reply(&model, asks("", &[Message::user("call read a.txt")])).await
    else {
        return;
    };
    // Up to the tool call, then the reply is dropped while the harness waits.
    while let Some(Ok(item)) = stream.next().await {
        if format!("{item:?}").contains("ToolCall") {
            break;
        }
    }
    drop(stream);
    for _ in 0..200 {
        if ended.load(Ordering::SeqCst) == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    assert_eq!(ended.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn waiting_harnesses_past_the_limit_end_oldest_first() {
    let (model, seen, ended) = model(2);

    for prompt in ["one", "two", "three"] {
        let _ = answer(&model, asks("", &[Message::user(prompt)])).await;
    }
    let after_three = ended.load(Ordering::SeqCst);
    // The oldest conversation goes on in a new harness, the newest in its own.
    let again = |prompt: &str| {
        let history = [
            Message::user(prompt),
            Message::assistant(format!("echo: {prompt}")),
            Message::user("more"),
        ];
        asks("", &history)
    };
    let _ = answer(&model, again("three")).await;
    let _ = answer(&model, again("one")).await;

    assert_eq!(after_three, 1);
    assert_eq!(prompts(&seen).len(), 4);
    assert_eq!(
        prompts(&seen).get(2),
        Some(&vec!["three".to_owned(), "more".to_owned()])
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_options_a_harness_cannot_take_are_named() {
    let (model, _, _) = model(1);
    let plain = asks("", &[Message::user("one")]);
    let tuned = plain.clone().temperature(0.2).max_tokens(100);
    let left_out = |request: CompletionRequest| -> Vec<String> {
        let kind = EffectKind::Completion {
            request: Arc::new(request),
            stream: true,
        };
        let refused = model.check(&kind);
        refused
            .into_iter()
            .map(|refused| refused.option.into_owned())
            .collect()
    };

    assert_eq!(left_out(plain), Vec::<String>::new());
    assert_eq!(left_out(tuned), ["temperature", "max_tokens"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_history_with_another_reply_is_another_conversation() {
    let (model, seen, _) = model(4);
    let first = answer(&model, asks("", &[Message::user("one")])).await;
    let elsewhere = [
        Message::user("one"),
        Message::assistant("something this harness never said"),
        Message::user("two"),
    ];

    let second = answer(&model, asks("", &elsewhere)).await;

    assert_eq!(said(first.as_ref()).0, "echo: one");
    assert!(said(second.as_ref()).0.ends_with("two"));
    assert_eq!(prompts(&seen).len(), 2);
}

/// One turn offering `read`, then the next offering `read` and `write`:
/// how many harnesses ran, and the listings made on being told.
async fn retooled(harness: Scripted) -> (usize, Vec<Vec<String>>) {
    let (model, seen, _) = on(harness, 4);
    let mut history = vec![Message::user("one")];
    let first = answer(&model, asks("", &history)).await;
    history.push(turn(first.as_ref()));
    history.push(Message::user("two"));
    let write = ToolName::new("write").ok();
    let write =
        write.map(|name| ToolDefinition::new(name, "Writes a file.", json!({ "type": "object" })));
    // Time for the harness to open its stream.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let _ = answer(&model, asks("", &history).tools(write)).await;

    let seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
    (seen.started.len(), seen.listed.clone())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_harness_told_of_other_tools_lists_them_and_runs_on() {
    let listening = Scripted {
        listens: true,
        ..Scripted::default()
    };

    let (harnesses, listed) = retooled(listening).await;

    assert_eq!(harnesses, 1);
    assert_eq!(listed, [["read".to_owned(), "write".to_owned()]]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_harness_that_cannot_be_told_or_fixes_its_tools_is_replaced() {
    let deaf = Scripted::default();
    let fixed = Scripted {
        listens: true,
        fixed_tools: true,
        ..Scripted::default()
    };

    let (unheard, _) = retooled(deaf).await;
    let (unchanged, listed) = retooled(fixed).await;

    assert_eq!(unheard, 2);
    assert_eq!((unchanged, listed), (2, Vec::new()));
}

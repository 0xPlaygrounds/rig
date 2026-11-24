use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use futures::StreamExt;
use rig_harness::prelude::Processes;
use rig_harness::rig_core::completion::{
    AssistantContent, CompletionRequest, CompletionResponse, Effort, FinishReason, Message,
    Reasoning, ToolDefinition, Usage,
};
use rig_harness::rig_core::effect::{Answered, EffectId, EffectKind};
use rig_harness::rig_core::message::{
    AssistantMessage, ToolCall, ToolFunction, ToolName, ToolResultContent, UserContent,
};
use rig_harness::rig_core::message::{DocumentSourceKind, Image};
use rig_harness::rig_core::serve::{Dispatch, Reply, Serve};
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc};

use super::{Bridge, Model, STOPPED, apply_context, budgeted_opening, digested};
use crate::harness::{Event, Harness, Launch, Part, Prompt, Running, Steer, Vendor};
use crate::mcp::tests::{post, request};

/// What the scripted harness saw: each start's system prompt and prompts.
#[derive(Default)]
struct Seen {
    started: Vec<(String, Vec<String>)>,
    /// The tool names of each listing a harness made once told the tools
    /// changed.
    listed: Vec<Vec<String>>,
    /// The messages a harness took for its running turn.
    steered: Vec<String>,
    /// The saved sessions harnesses were started on.
    resumed: Vec<String>,
}

/// A harness whose model does what its prompt says. A prompt containing
/// `call read <path>` calls Rig's `read` tool once for each such phrase, at
/// once, and then says what came back; one containing `wait` says so and
/// goes on until it is stopped, and one containing `crash` fails then; any
/// other prompt is echoed.
#[derive(Default)]
struct Scripted {
    seen: Arc<Mutex<Seen>>,
    /// How many started harnesses were dropped.
    ended: Arc<AtomicUsize>,
    /// Whether the harness holds a stream open for the server's
    /// notifications and lists the tools again when told they changed.
    listens: bool,
    /// Signaled once the SSE notification stream has received its 200 OK header.
    listening: Arc<Notify>,
    /// Signaled after `listen` records a refreshed tool list into `seen.listed`.
    listed_done: Arc<Notify>,
    /// What the harness says of taking tools while it runs.
    fixed_tools: bool,
    /// Whether the harness takes a message for its running turn, when it
    /// has a way to be given one.
    steers: Option<bool>,
    /// Whether the harness saves its session, under a name it says.
    saves: bool,
    /// Whether a harness started on a saved session finds none.
    forgets: bool,
}

/// Holds a notification stream open on the server at `url` and lists the
/// tools again, into `seen`, each time it says they changed.
async fn listen(
    url: String,
    token: String,
    seen: Arc<Mutex<Seen>>,
    ready: Arc<Notify>,
    listed_done: Arc<Notify>,
) {
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
    let mut signaled = false;
    while let Ok(read) = stream.read(&mut chunk).await {
        let told = chunk
            .get(..read)
            .map(String::from_utf8_lossy)
            .unwrap_or_default();
        if read == 0 {
            return;
        }
        if !signaled && told.contains("200") {
            signaled = true;
            ready.notify_one();
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
            listed_done.notify_one();
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

    fn retools_saved(&self) -> bool {
        !self.fixed_tools
    }

    fn start(&self, launch: Launch<'_>) -> Result<Running, String> {
        let (prompts, mut prompted) = mpsc::unbounded_channel::<Prompt>();
        let (stops, mut stopped) = mpsc::unbounded_channel::<u64>();
        let forgot = self.forgets && launch.resume.is_some();
        let (events, reported) = mpsc::unbounded_channel();
        let seen = self.seen.clone();
        let index = {
            let mut seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
            seen.started.push((launch.system.to_owned(), Vec::new()));
            seen.resumed.extend(launch.resume.map(str::to_owned));
            seen.started.len() - 1
        };
        if self.saves {
            let name = launch
                .resume
                .map_or_else(|| format!("saved-{index}"), str::to_owned);
            let _ = events.send(Event::Saved(name));
        }
        let (url, token) = (launch.mcp.url().to_owned(), launch.mcp.token().to_owned());
        if self.listens {
            tokio::spawn(listen(
                url.clone(),
                token.clone(),
                seen.clone(),
                self.listening.clone(),
                self.listed_done.clone(),
            ));
        }
        tokio::spawn(async move {
            loop {
                // A stop with no turn to stop does nothing.
                let prompt = tokio::select! {
                    prompt = prompted.recv() => prompt,
                    Some(_) = stopped.recv() => continue,
                };
                let Some(prompt) = prompt else {
                    return;
                };
                let text: String = prompt
                    .parts
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
                if forgot {
                    let _ = events.send(Event::Done(Err("no such session".to_owned())));
                    return;
                }
                let _ = events.send(Event::Began);
                let _ = events.send(Event::Reasoning("thinking".to_owned()));
                if asked.contains("wait") || asked.contains("crash") {
                    let _ = events.send(Event::Text("Waiting.".to_owned()));
                    let _ = stopped.recv().await;
                    let _ = events.send(match asked.contains("crash") {
                        true => Event::Done(Err("it crashed".to_owned())),
                        false => Event::Stopped,
                    });
                    continue;
                }
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
                            let content = response.pointer("/result/content");
                            let error = response.pointer("/result/isError") == Some(&json!(true));
                            let text: String = content
                                .and_then(Value::as_array)
                                .into_iter()
                                .flatten()
                                .filter_map(|block| block.get("text")?.as_str())
                                .collect();
                            format!("{}{text}", if error { "error: " } else { "" })
                        })
                        .collect();
                    let _ = events.send(Event::Text(format!("read said: {}", said.join(" | "))));
                }
                let mut usage = Usage::new();
                usage.input_tokens = Some(10);
                usage.output_tokens = Some(2);
                let _ = events.send(Event::Usage(usage));
                let _ = events.send(Event::Done(Ok(FinishReason::Stop)));
            }
        });
        let steers = self.steers.map(|takes| {
            let (steers, mut steered) = mpsc::unbounded_channel::<Steer>();
            let seen = self.seen.clone();
            tokio::spawn(async move {
                while let Some(steer) = steered.recv().await {
                    let taken = match (takes, steer.parts.as_slice()) {
                        (true, [Part::Text(text)]) => {
                            let mut seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
                            seen.steered.push(text.clone());
                            Ok(())
                        }
                        _ => Err("not now".to_owned()),
                    };
                    let _ = steer.taken.send(taken);
                }
            });
            steers
        });
        Ok(Running {
            prompts,
            steers,
            stops,
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
    let scratch =
        std::env::temp_dir().join(format!("rig-bridge-test-{}", uuid::Uuid::new_v4().simple()));
    at(&scratch, harness, limit)
}

/// A model on `harness` whose bridge keeps its directories under `root`.
fn at(
    root: &std::path::Path,
    harness: Scripted,
    limit: usize,
) -> (Model, Arc<Mutex<Seen>>, Arc<AtomicUsize>) {
    let (seen, ended) = (harness.seen.clone(), harness.ended.clone());
    let bridge = Bridge::new(
        tokio::runtime::Handle::current(),
        Processes::default(),
        limit,
        None,
        root.join("scratch"),
        root.join("kept"),
    );
    let model = Model {
        bridge: Arc::new(bridge),
        harness: Arc::new(harness),
        program: "scripted".into(),
        vendor: Vendor::ClaudeCode,
        model: "model-1".to_owned(),
        context_limit: Some(200_000),
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
    // A tool's output is under a heading of its own, not the user's.
    for expected in [
        "<conversation>\n## User\ncall read a.txt\n",
        "## Assistant\nReading.\n[called `read` with {\"path\":\"a.txt\"}]\n",
        "## Tool result of `read`\nalpha\n",
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

/// The user message that answers each tool call of `response` with `result`
/// and says `said` beside the results, as Rig sends what the user typed
/// while the tools ran.
fn results_and(response: Option<&Arc<CompletionResponse>>, result: &str, said: &str) -> Message {
    let Message::User { content } = results(response, |_| result.to_owned()) else {
        return Message::user(said);
    };
    Message::User {
        content: content
            .into_iter()
            .chain([UserContent::text(said)])
            .collect(),
    }
}

/// What a model on `harness` said once its user had said `and b.txt too`
/// while its tool ran, the prompts the harness got and the messages it took
/// for its running turn. Either way the harness goes on.
async fn steered(harness: Scripted) -> (String, Vec<String>, Vec<String>) {
    let (model, seen, ended) = on(harness, 4);
    let mut history = vec![Message::user("call read a.txt")];
    let called = answer(&model, asks("Be brief.", &history)).await;
    history.push(turn(called.as_ref()));
    history.push(results_and(called.as_ref(), "alpha", "and b.txt too"));
    let finished = answer(&model, asks("Be brief.", &history)).await;

    assert_eq!(ended.load(Ordering::SeqCst), 0);
    let prompted = prompts(&seen).into_iter().flatten().collect();
    let taken = std::mem::take(&mut seen.lock().unwrap_or_else(PoisonError::into_inner).steered);
    (said(finished.as_ref()).0, prompted, taken)
}

#[tokio::test(flavor = "multi_thread")]
async fn what_the_user_says_while_a_tool_runs_is_a_message_for_the_running_turn() {
    let takes = Scripted {
        steers: Some(true),
        ..Scripted::default()
    };

    let (said, prompted, taken) = steered(takes).await;

    // The tool's result is the tool's alone.
    assert_eq!(said, "read said: alpha");
    assert_eq!(taken, ["and b.txt too"]);
    assert_eq!(prompted, ["call read a.txt"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_harness_that_takes_no_message_is_prompted_with_it_when_its_turn_ends() {
    let refuses = Scripted {
        steers: Some(false),
        ..Scripted::default()
    };
    for harness in [Scripted::default(), refuses] {
        let (said, prompted, taken) = steered(harness).await;

        // One reply holds both turns: the tool's result, the tool's alone,
        // and then the answer to the message.
        assert_eq!(said, "read said: alpha\n\necho: and b.txt too");
        assert_eq!(taken, Vec::<String>::new());
        assert_eq!(prompted, ["call read a.txt", "and b.txt too"]);
    }
}

/// Reads `stream` until `text` was said, then drops it, as Rig does with a
/// reply the user stopped, and gives the session time to stop its turn.
async fn stopped_at(model: &Model, request: CompletionRequest, text: &str) {
    let Reply::Stream(mut stream) = reply(model, request).await else {
        return;
    };
    while let Some(Ok(item)) = stream.next().await {
        if format!("{item:?}").contains(text) {
            break;
        }
    }
    drop(stream);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dropped_reply_stops_the_turn_and_the_harness_goes_on_with_what_is_said_next() {
    let (model, seen, ended) = model(4);
    let stopped = Message::user(STOPPED);

    // Rig may keep what it had of the reply, or nothing of it.
    stopped_at(&model, asks("", &[Message::user("wait a")]), "Waiting.").await;
    let kept = [
        Message::user("wait a"),
        Message::assistant("Wai"),
        stopped.clone(),
        Message::user("one"),
    ];
    let first = answer(&model, asks("", &kept)).await;
    let mut history = kept.to_vec();
    history.extend([turn(first.as_ref()), Message::user("wait b")]);
    stopped_at(&model, asks("", &history), "Waiting.").await;
    history.extend([stopped, Message::user("two")]);
    let second = answer(&model, asks("", &history)).await;

    assert_eq!(said(first.as_ref()).0, format!("echo: {STOPPED}\none"));
    assert_eq!(said(second.as_ref()).0, format!("echo: {STOPPED}\ntwo"));
    // One harness had every prompt, and only what was said after each stop.
    assert_eq!(
        prompts(&seen),
        vec![vec![
            "wait a".to_owned(),
            format!("{STOPPED}\none"),
            "wait b".to_owned(),
            format!("{STOPPED}\ntwo"),
        ]]
    );
    assert_eq!(ended.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stopped_harness_goes_on_only_the_conversation_it_was_stopped_in() {
    let (model, seen, _) = model(4);

    stopped_at(&model, asks("", &[Message::user("wait a")]), "Waiting.").await;
    // The same request answered in full elsewhere, and one stopped after
    // other words: neither is what this harness has.
    let whole = [
        Message::user("wait a"),
        Message::assistant("Waiting. Done."),
        Message::user("one"),
    ];
    let other = [
        Message::user("wait a"),
        Message::assistant("Wondering"),
        Message::user(STOPPED),
        Message::user("one"),
    ];
    let _ = answer(&model, asks("", &whole)).await;
    let _ = answer(&model, asks("", &other)).await;

    let opened = prompts(&seen);
    assert_eq!(opened.len(), 3, "{opened:?}");
    let written = opened.concat().concat();
    assert!(written.contains("Waiting. Done.") && written.contains("Wondering"));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_turn_that_fails_instead_of_stopping_ends_its_harness() {
    let (model, _, ended) = model(4);

    stopped_at(&model, asks("", &[Message::user("crash")]), "Waiting.").await;
    for _ in 0..300 {
        if ended.load(Ordering::SeqCst) == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    // Nothing says the harness has the prompt: nothing goes on from it.
    assert_eq!(ended.load(Ordering::SeqCst), 1);
}

#[test]
fn the_digest_of_a_message_tells_images_apart() {
    let with = |bytes: &str| {
        let image = Image {
            data: DocumentSourceKind::Base64(bytes.to_owned()),
            ..Image::default()
        };
        let content = vec![UserContent::text("look"), UserContent::Image(image)];
        digested(&Message::User { content })
    };

    assert_eq!(with("AAAA"), with("AAAA"));
    assert_ne!(with("AAAA"), with("AAAB"));
    // And where an image is among the words.
    let image = || UserContent::Image(Image::default());
    let word = || UserContent::text("[an image]");
    assert_ne!(
        digested(&Message::User {
            content: vec![image(), word()]
        }),
        digested(&Message::User {
            content: vec![word(), image()]
        }),
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_reply_dropped_with_a_call_open_gives_the_call_up_and_keeps_the_harness() {
    let (model, seen, ended) = model(4);
    let history = [Message::user("call read a.txt")];

    stopped_at(&model, asks("", &history), "ToolCall").await;
    let next = [
        history[0].clone(),
        Message::user(STOPPED),
        Message::user("more"),
    ];
    let went_on = answer(&model, asks("", &next)).await;

    assert_eq!(said(went_on.as_ref()).0, format!("echo: {STOPPED}\nmore"));
    assert_eq!(prompts(&seen).len(), 1);
    assert_eq!(ended.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_new_bridge_goes_on_from_the_session_a_harness_saved() {
    let root =
        std::env::temp_dir().join(format!("rig-bridge-test-{}", uuid::Uuid::new_v4().simple()));
    let saving = || Scripted {
        saves: true,
        ..Scripted::default()
    };
    let mut history = vec![Message::user("one")];
    {
        let (model, ..) = at(&root, saving(), 4);
        let first = answer(&model, asks("", &history)).await;
        history.extend([turn(first.as_ref()), Message::user("two")]);
    }

    // Another bridge, as after Rig started again: the harness is started on
    // what the first saved, and sent only what the user said since.
    let (model, seen, _) = at(&root, saving(), 4);
    let second = answer(&model, asks("", &history)).await;
    assert_eq!(said(second.as_ref()).0, "echo: two");
    assert_eq!(prompts(&seen), vec![vec!["two".to_owned()]]);
    let resumed = |seen: &Arc<Mutex<Seen>>| {
        let seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
        seen.resumed.clone()
    };
    assert_eq!(resumed(&seen), ["saved-0"]);

    // While the harness answers, nothing can go on from what it saved: its
    // session is ahead of Rig's history.
    history.extend([turn(second.as_ref()), Message::user("call read a.txt")]);
    let called = answer(&model, asks("", &history)).await;
    history.extend([
        turn(called.as_ref()),
        results(called.as_ref(), |_| "alpha".to_owned()),
    ]);
    drop(model);
    let (model, seen, _) = at(&root, saving(), 4);
    let _ = answer(&model, asks("", &history)).await;
    assert_eq!(resumed(&seen), Vec::<String>::new());
    let opened = prompts(&seen).concat().concat();
    assert!(opened.contains("<conversation>"), "{opened}");

    // Nor from a conversation that is not the one it saved.
    let other = [
        Message::user("one"),
        Message::assistant("echo: uno"),
        Message::user("two"),
    ];
    let (model, seen, _) = at(&root, saving(), 4);
    let _ = answer(&model, asks("", &other)).await;
    assert_eq!(resumed(&seen), Vec::<String>::new());
    drop(model);
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_new_bridge_goes_on_from_a_turn_that_was_stopped_and_writes_out_what_it_cannot() {
    let root =
        std::env::temp_dir().join(format!("rig-bridge-test-{}", uuid::Uuid::new_v4().simple()));
    let saving = |forgets| Scripted {
        saves: true,
        forgets,
        ..Scripted::default()
    };
    let mut history = vec![Message::user("wait a")];
    {
        let (model, ..) = at(&root, saving(false), 4);
        stopped_at(&model, asks("", &history), "Waiting.").await;
        // The stopped session is waiting once its turn has ended.
        for _ in 0..300 {
            if model.bridge.snapshot().warm.len() == 1 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
    history.extend([
        Message::assistant("Wai"),
        Message::user(STOPPED),
        Message::user("one"),
    ]);

    let (model, seen, _) = at(&root, saving(false), 4);
    let first = answer(&model, asks("", &history)).await;
    assert_eq!(said(first.as_ref()).0, format!("echo: {STOPPED}\none"));
    assert_eq!(prompts(&seen), vec![vec![format!("{STOPPED}\none")]]);
    history.extend([turn(first.as_ref()), Message::user("two")]);
    drop(model);

    // A harness that no longer has what it saved: the conversation is
    // written out for a new one, and answered all the same.
    let (model, seen, _) = at(&root, saving(true), 4);
    let second = answer(&model, asks("", &history)).await;
    assert_eq!(said(second.as_ref()).0, "echo: \n\ntwo");
    let opened = prompts(&seen);
    assert_eq!(opened.len(), 2, "{opened:?}");
    assert_eq!(opened.first(), Some(&vec!["two".to_owned()]));
    assert!(opened.concat().concat().contains("<conversation>"));
    drop(model);
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_saved_session_that_keeps_its_tools_is_gone_on_from_only_under_them() {
    let root =
        std::env::temp_dir().join(format!("rig-bridge-test-{}", uuid::Uuid::new_v4().simple()));
    for (fixed_tools, goes_on) in [(false, 1), (true, 0)] {
        let root = root.join(if fixed_tools { "fixed" } else { "free" });
        let saving = || Scripted {
            saves: true,
            fixed_tools,
            ..Scripted::default()
        };
        let mut history = vec![Message::user("one")];
        {
            let (model, ..) = at(&root, saving(), 4);
            let first = answer(&model, asks("", &history)).await;
            history.extend([turn(first.as_ref()), Message::user("two")]);
        }

        // The conversation goes on without the tool it was offered before.
        let (model, seen, _) = at(&root, saving(), 4);
        let bare = CompletionRequest::new(Message::system("")).messages(history.iter().cloned());
        let _ = answer(&model, bare).await;
        let seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
        assert_eq!(seen.resumed.len(), goes_on, "fixed tools: {fixed_tools}");
    }
    let _ = std::fs::remove_dir_all(&root);
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
    let waits_relist = harness.listens && !harness.fixed_tools;
    let listens = harness.listens;
    let listening = harness.listening.clone();
    let listed_done = harness.listed_done.clone();
    let (model, seen, _) = on(harness, 4);
    let mut history = vec![Message::user("one")];
    let first = answer(&model, asks("", &history)).await;
    history.push(turn(first.as_ref()));
    history.push(Message::user("two"));
    let write = ToolName::new("write").ok();
    let write =
        write.map(|name| ToolDefinition::new(name, "Writes a file.", json!({ "type": "object" })));
    if listens {
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), listening.notified()).await;
    }

    let _ = answer(&model, asks("", &history).tools(write)).await;
    if waits_relist {
        let _ =
            tokio::time::timeout(std::time::Duration::from_secs(2), listed_done.notified()).await;
    }

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

#[test]
fn budgeted_opening_always_preserves_latest_user_turn_under_huge_tool_history() {
    let huge_output = "🚀".repeat(40_000); // 160 KB of 4-byte UTF-8
    let Ok(shell_name) = ToolName::new("shell") else {
        return;
    };
    let call = ToolCall::new(
        rig_harness::rig_core::message::CallId::from_wire("call-1"),
        ToolFunction::new(shell_name, json!({})),
    );
    let tool_result = Message::tool_results(vec![
        call.result(vec![ToolResultContent::Text(huge_output.into())]),
    ]);
    let assistant_call =
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            call,
        )]));
    let latest_canary = "LATEST_USER_CANARY_998877";
    let history = vec![
        Arc::new(Message::user("first turn")),
        Arc::new(assistant_call),
        Arc::new(tool_result),
        Arc::new(Message::user(latest_canary)),
    ];

    let (parts, trimmed) = budgeted_opening(&history, 16 * 1024);
    assert!(trimmed.is_some());
    let joined: String = parts
        .iter()
        .filter_map(|part| match part {
            Part::Text(text) => Some(text.as_str()),
            Part::Image { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(joined.contains(latest_canary));
    assert!(joined.contains("tool output elided"));
    assert!(joined.len() < 20 * 1024);
}

#[tokio::test(flavor = "multi_thread")]
async fn bridge_reset_stops_waiting_sessions() {
    let (model, _, ended) = model(4);
    let _ = answer(&model, asks("", &[Message::user("hello")])).await;
    assert_eq!(model.bridge.snapshot().warm.len(), 1);
    assert_eq!(model.bridge.reset(), 1);
    assert_eq!(model.bridge.snapshot().warm.len(), 0);
    assert_eq!(ended.load(Ordering::SeqCst), 1);
}

#[test]
fn apply_context_prefers_per_inference_context_over_summed_totals() {
    // Multi-round ACP turn: turn totals have 3.5M input, but latest
    // usage_update says 105k context.
    let turn_totals = Usage::new()
        .input_tokens(3_500_000)
        .output_tokens(40_000)
        .total_tokens(3_540_000);
    let fixed = apply_context(turn_totals, Some(105_000), Some(3_540_000), 1);
    assert_eq!(fixed.context_tokens(), Some(105_000));
    assert_eq!(fixed.input_tokens, Some(3_500_000));
    assert_eq!(fixed.output_tokens, Some(40_000));

    // Delayed multi-report harness (e.g. Codex): two per-inference usages
    // landed in one answer() call, summing input_tokens to 5,736; context
    // must reflect only the last inference (2,921), not the sum (5,761).
    let mut summed = Usage::new()
        .input_tokens(2_830)
        .output_tokens(10)
        .total_tokens(2_840);
    summed += Usage::new()
        .input_tokens(2_906)
        .output_tokens(15)
        .total_tokens(2_921);
    let fixed = apply_context(summed, None, Some(2_921), 2);
    assert_eq!(fixed.context_tokens(), Some(2_921));
    assert_eq!(fixed.input_tokens, Some(5_736));
}

#[test]
fn budgeted_opening_pins_leading_compaction_summary_before_dropping_middle_turns() {
    let summary = "The conversation history before this point was compacted into the following summary:\n<summary>\nPINNED_SUMMARY_CANARY\n</summary>";
    let filler = "x".repeat(4_000);
    let ack = || {
        Arc::new(Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::text("ok"),
        ])))
    };
    let history = vec![
        Arc::new(Message::user(summary)),
        ack(),
        Arc::new(Message::user(format!("middle-1 {filler}"))),
        ack(),
        Arc::new(Message::user(format!("middle-2 {filler}"))),
        ack(),
        Arc::new(Message::user("LATEST_USER_TURN")),
    ];

    let (parts, trimmed) = budgeted_opening(&history, 5_000);
    assert!(trimmed.is_some());
    let joined: String = parts
        .iter()
        .filter_map(|part| match part {
            Part::Text(text) => Some(text.as_str()),
            Part::Image { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(joined.contains("PINNED_SUMMARY_CANARY"));
    assert!(joined.contains("LATEST_USER_TURN"));
    assert!(!joined.contains("middle-1"));
}

#[tokio::test(flavor = "multi_thread")]
async fn duplicate_tool_results_do_not_reuse_waiting_session() {
    let (model, seen, _) = model(4);
    let mut history = vec![Message::user("call read a.txt and call read b.txt")];
    let called = answer(&model, asks("", &history)).await;
    let first_call = called
        .as_ref()
        .into_iter()
        .flat_map(|response| &response.choice)
        .find_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.clone()),
            _ => None,
        });
    assert_eq!(said(called.as_ref()).1.len(), 2);
    let Some(first_call) = first_call else {
        return;
    };
    history.push(turn(called.as_ref()));
    // Send two results for the SAME CallId instead of one for each call.
    history.push(Message::tool_results(vec![
        first_call.result(vec![ToolResultContent::Text("one".into())]),
        first_call.result(vec![ToolResultContent::Text("two".into())]),
    ]));

    let _ = answer(&model, asks("", &history)).await;
    // Duplicate results must reject session reuse and start a fresh harness.
    assert_eq!(prompts(&seen).len(), 2);
}

/// Closure-backed test harness that records how many sessions started and
/// runs `run(start_index, mcp_url, mcp_token, prompts, events)` per session.
struct FnHarness<F> {
    starts: Arc<AtomicUsize>,
    dynamic_effort: bool,
    run: Arc<F>,
}

impl<F, Fut> Harness for FnHarness<F>
where
    F: Fn(
            usize,
            String,
            String,
            mpsc::UnboundedReceiver<Prompt>,
            mpsc::UnboundedSender<Event>,
        ) -> Fut
        + Send
        + Sync
        + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    fn note(&self, _: Option<&std::path::Path>) -> String {
        String::new()
    }
    fn changes_effort(&self) -> bool {
        self.dynamic_effort
    }
    fn start(&self, launch: Launch<'_>) -> Result<Running, String> {
        let run = self.starts.fetch_add(1, Ordering::SeqCst);
        let (prompts, prompted) = mpsc::unbounded_channel::<Prompt>();
        let (events, reported) = mpsc::unbounded_channel();
        tokio::spawn((self.run)(
            run,
            launch.mcp.url().to_owned(),
            launch.mcp.token().to_owned(),
            prompted,
            events,
        ));
        Ok(Running {
            prompts,
            steers: None,
            stops: mpsc::unbounded_channel().0,
            events: reported,
            _alive: Box::new(()),
        })
    }
}

fn fn_model<F, Fut>(
    vendor: Vendor,
    dynamic_effort: bool,
    run: F,
) -> (Arc<Bridge>, Model, Arc<AtomicUsize>)
where
    F: Fn(
            usize,
            String,
            String,
            mpsc::UnboundedReceiver<Prompt>,
            mpsc::UnboundedSender<Event>,
        ) -> Fut
        + Send
        + Sync
        + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let starts = Arc::new(AtomicUsize::new(0));
    let scratch =
        std::env::temp_dir().join(format!("rig-bridge-test-{}", uuid::Uuid::new_v4().simple()));
    let bridge = Arc::new(Bridge::new(
        tokio::runtime::Handle::current(),
        Processes::default(),
        2,
        None,
        scratch.join("scratch"),
        scratch.join("kept"),
    ));
    let model = Model {
        bridge: bridge.clone(),
        harness: Arc::new(FnHarness {
            starts: starts.clone(),
            dynamic_effort,
            run: Arc::new(run),
        }),
        program: "test".into(),
        vendor,
        model: "model-1".to_owned(),
        context_limit: Some(200_000),
    };
    (bridge, model, starts)
}

#[tokio::test(flavor = "multi_thread")]
async fn counting_tool_call_without_usage_settles_on_watchdog() {
    let (_, model, _) = fn_model(
        Vendor::ClaudeCode,
        false,
        |_, url, token, mut prompted, events| async move {
            if prompted.recv().await.is_some() {
                // Emit Began and invoke a tool, but never send Event::Usage.
                let _ = events.send(Event::Began);
                let call = request(
                    1,
                    "tools/call",
                    json!({ "name": "read", "arguments": { "path": "a.txt" } }),
                );
                let _ = post(&url, &token, &call).await;
                let _ = events.send(Event::Done(Ok(FinishReason::Stop)));
            }
        },
    );
    let reply = tokio::time::timeout(
        std::time::Duration::from_secs(4),
        answer(&model, asks("", &[Message::user("go")])),
    )
    .await;
    assert!(
        reply.is_ok(),
        "counting watchdog must settle instead of hanging"
    );
    assert_eq!(said(reply.ok().flatten().as_ref()).1.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn idle_crashed_session_is_pruned_and_restarts_cleanly() {
    let (_, model, starts) = fn_model(
        Vendor::ClaudeCode,
        false,
        |run, _, _, mut prompted, events| async move {
            if prompted.recv().await.is_some() {
                let _ = events.send(Event::Text(format!("reply-{run}")));
                let _ = events.send(Event::Done(Ok(FinishReason::Stop)));
            }
            if run == 0 {
                return;
            }
            while prompted.recv().await.is_some() {
                let _ = events.send(Event::Text(format!("reply-{run}-next")));
                let _ = events.send(Event::Done(Ok(FinishReason::Stop)));
            }
        },
    );
    let h1 = vec![Message::user("first")];
    assert_eq!(
        said(answer(&model, asks("", &h1)).await.as_ref()).0,
        "reply-0"
    );
    tokio::task::yield_now().await;

    let h2 = vec![
        Message::user("first"),
        Message::assistant("reply-0"),
        Message::user("second"),
    ];
    assert_eq!(
        said(answer(&model, asks("", &h2)).await.as_ref()).0,
        "reply-1"
    );
    assert_eq!(starts.load(Ordering::SeqCst), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn warm_prompt_failing_before_output_recovers_on_fresh_session() {
    let prompts_seen = Arc::new(Mutex::new(Vec::new()));
    let captured_in = prompts_seen.clone();
    let (bridge, model, starts) = fn_model(
        Vendor::Antigravity,
        false,
        move |run, _, _, mut prompted, events| {
            let prompts_seen = captured_in.clone();
            async move {
                let mut turn = 0;
                while let Some(prompt) = prompted.recv().await {
                    let text: String = prompt
                        .parts
                        .into_iter()
                        .filter_map(|p| match p {
                            Part::Text(t) => Some(t),
                            Part::Image { .. } => None,
                        })
                        .collect();
                    prompts_seen
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push(text);
                    turn += 1;
                    if run == 0 && turn == 2 {
                        let _ = events
                            .send(Event::Done(Err("corrupted thinking signature".to_owned())));
                        return;
                    }
                    let _ = events.send(Event::Text(format!("run-{run}-turn-{turn}")));
                    let _ = events.send(Event::Done(Ok(FinishReason::Stop)));
                }
            }
        },
    );
    let h1 = vec![Message::user("first")];
    assert_eq!(
        said(answer(&model, asks("", &h1)).await.as_ref()).0,
        "run-0-turn-1"
    );

    let h2 = vec![
        Message::user("first"),
        Message::assistant("run-0-turn-1"),
        Message::user("second"),
    ];
    assert_eq!(
        said(answer(&model, asks("", &h2)).await.as_ref()).0,
        "run-1-turn-1"
    );
    assert_eq!(starts.load(Ordering::SeqCst), 2);
    let captured = prompts_seen
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    assert_eq!(captured.len(), 3);
    let replayed = captured.get(2).map(String::as_str).unwrap_or_default();
    assert!(replayed.contains("first") && replayed.contains("second"));
    let note = bridge
        .snapshot()
        .vendors
        .get(&Vendor::Antigravity)
        .and_then(|stats| stats.last_note.clone())
        .unwrap_or_default();
    assert!(
        note.contains("corrupted thinking signature"),
        "expected recovery note, got {note:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn warm_prompt_does_not_retry_after_partial_output_or_tool_call() {
    #[derive(Clone, Copy)]
    enum BeforeErr {
        Text,
        Reasoning,
        ToolCall,
    }
    for mode in [BeforeErr::Text, BeforeErr::Reasoning, BeforeErr::ToolCall] {
        let (_, model, starts) = fn_model(
            Vendor::Antigravity,
            false,
            move |run, url, token, mut prompted, events| async move {
                let mut turn = 0;
                while prompted.recv().await.is_some() {
                    turn += 1;
                    if run == 0 && turn == 2 {
                        match mode {
                            BeforeErr::Text => {
                                let _ = events.send(Event::Text("partial".to_owned()));
                            }
                            BeforeErr::Reasoning => {
                                let _ = events.send(Event::Reasoning("thinking".to_owned()));
                            }
                            BeforeErr::ToolCall => {
                                let _ = events.send(Event::Began);
                                let req = request(
                                    1,
                                    "tools/call",
                                    json!({ "name": "read", "arguments": { "path": "a" } }),
                                );
                                tokio::spawn(async move {
                                    let _ = post(&url, &token, &req).await;
                                });
                                tokio::time::sleep(std::time::Duration::from_millis(30)).await;
                            }
                        }
                        let _ = events.send(Event::Done(Err("mid-turn boom".to_owned())));
                        return;
                    }
                    let _ = events.send(Event::Text(format!("ok-{run}-{turn}")));
                    let _ = events.send(Event::Done(Ok(FinishReason::Stop)));
                }
            },
        );
        let h1 = vec![Message::user("first")];
        assert_eq!(
            said(answer(&model, asks("", &h1)).await.as_ref()).0,
            "ok-0-1"
        );

        let h2 = vec![
            Message::user("first"),
            Message::assistant("ok-0-1"),
            Message::user("second"),
        ];
        assert!(
            reply(&model, asks("", &h2))
                .await
                .into_outcome()
                .await
                .is_err(),
            "partial-output failure must surface error"
        );
        assert_eq!(
            starts.load(Ordering::SeqCst),
            1,
            "must not restart session after emitting output or tool call"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn warm_prompt_retries_at_most_once_and_skips_non_recoverable_errors() {
    for (error, expected_starts) in [
        ("corrupted thinking signature", 2),
        ("the model refused to answer", 1),
        // A turn the harness says was stopped, though Rig stopped none.
        ("the turn was stopped", 1),
    ] {
        let (_, model, starts) = fn_model(
            Vendor::Antigravity,
            false,
            move |run, _, _, mut prompted, events| async move {
                let mut turn = 0;
                while prompted.recv().await.is_some() {
                    turn += 1;
                    if run == 0 && turn == 1 {
                        let _ = events.send(Event::Text("ok-0".to_owned()));
                        let _ = events.send(Event::Done(Ok(FinishReason::Stop)));
                    } else {
                        let _ = events.send(match error {
                            "the turn was stopped" => Event::Stopped,
                            error => Event::Done(Err(error.to_owned())),
                        });
                        return;
                    }
                }
            },
        );
        let h1 = vec![Message::user("first")];
        assert_eq!(said(answer(&model, asks("", &h1)).await.as_ref()).0, "ok-0");

        let h2 = vec![
            Message::user("first"),
            Message::assistant("ok-0"),
            Message::user("second"),
        ];
        assert!(
            reply(&model, asks("", &h2))
                .await
                .into_outcome()
                .await
                .is_err()
        );
        assert_eq!(starts.load(Ordering::SeqCst), expected_starts);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn warm_session_reuses_harness_across_effort_changes_only_when_supported() {
    for (dynamic, expected_starts, expected_warm) in [(true, 1, 1), (false, 3, 2)] {
        let efforts = Arc::new(Mutex::new(Vec::new()));
        let recorded = efforts.clone();
        let (bridge, model, starts) = fn_model(
            Vendor::Codex,
            dynamic,
            move |_, _, _, mut prompted, events| {
                let efforts = recorded.clone();
                async move {
                    while let Some(prompt) = prompted.recv().await {
                        efforts
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .push(prompt.effort);
                        let label = prompt.effort.map_or("default", |e| e.as_str());
                        let _ = events.send(Event::Text(format!("ok-{label}")));
                        let _ = events.send(Event::Done(Ok(FinishReason::Stop)));
                    }
                }
            },
        );

        let ask_effort = |history: &[Message], effort: Option<Effort>| {
            let mut req = asks("", history);
            req.options.reasoning = effort.map(Reasoning::Effort);
            req
        };

        let mut h = vec![Message::user("t1")];
        assert_eq!(
            said(answer(&model, ask_effort(&h, None)).await.as_ref()).0,
            "ok-default"
        );
        h.extend([Message::assistant("ok-default"), Message::user("t2")]);
        assert_eq!(
            said(
                answer(&model, ask_effort(&h, Some(Effort::Low)))
                    .await
                    .as_ref()
            )
            .0,
            "ok-low"
        );
        h.extend([Message::assistant("ok-low"), Message::user("t3")]);
        assert_eq!(
            said(
                answer(&model, ask_effort(&h, Some(Effort::High)))
                    .await
                    .as_ref()
            )
            .0,
            "ok-high"
        );

        assert_eq!(starts.load(Ordering::SeqCst), expected_starts);
        assert_eq!(
            *efforts.lock().unwrap_or_else(PoisonError::into_inner),
            vec![None, Some(Effort::Low), Some(Effort::High)]
        );
        let snap = bridge.snapshot();
        assert_eq!(snap.warm.len(), expected_warm);
        assert_eq!(snap.warm.last().and_then(|w| w.effort), Some(Effort::High));
    }
}

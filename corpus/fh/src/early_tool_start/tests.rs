use std::collections::VecDeque;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use futures::channel::mpsc::{UnboundedReceiver, UnboundedSender, unbounded};
use rig_core::catalog::Catalog;
use rig_core::completion::{CompletionRequest, ProviderCapabilities};
use rig_core::effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, family, model_key};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::{Message as ChatMessage, Origin as Made, ToolResultContent};
use rig_core::operation::Finish;
use rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve, stream_truncated};
use rig_core::tool::{ContentHash, FileRead, FilesRead, ToolContext};
use rig_harness::rig_ecs::effects::Handler;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{Ended, app_on, first_agent, log_ended, run_until};
use schemars::JsonSchema;
use serde_json::json;

use super::*;

/// What happened, in order: each chunk as the model sends it, each tool
/// call as it runs, and the reply's end.
type Log = Arc<Mutex<Vec<String>>>;

fn note(log: &Log, line: String) {
    if let Ok(mut log) = log.lock() {
        log.push(line);
    }
}

fn logged(log: &Log) -> Vec<String> {
    log.lock().map(|log| log.clone()).unwrap_or_default()
}

/// Where `line` is in the log.
fn at(log: &[String], line: &str) -> Option<usize> {
    log.iter().position(|logged| logged == line)
}

/// A chunk of a scripted reply.
enum Step {
    Text(&'static str),
    Call(&'static str, serde_json::Value),
    /// The stream breaks off.
    Fail,
}

/// A scripted reply: each step waits for a permit from `gate`, if any.
struct Script {
    steps: Vec<Step>,
    gate: Option<UnboundedReceiver<()>>,
}

/// A model that streams its scripted replies a chunk at a time, as the
/// test lets them through.
#[derive(Clone, Default)]
struct Paced {
    scripts: Arc<Mutex<VecDeque<Script>>>,
    requests: Arc<Mutex<Vec<CompletionRequest>>>,
    log: Log,
}

impl Serve for Paced {
    type Family = family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: model_key("test/paced"),
            family: FamilyDescriptor::Completion {
                model: "test/paced".into(),
                capabilities: ProviderCapabilities::new(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _: Dispatch) -> Reply {
        let EffectKind::Completion { request, .. } = kind else {
            let why = ErrorReport::new(ErrorKind::Internal, "completions only");
            return Reply::Outcome(Err(why));
        };
        if let Ok(mut requests) = self.requests.lock() {
            requests.push(std::sync::Arc::unwrap_or_clone(request));
        }
        let script = self.scripts.lock().ok().and_then(|mut all| all.pop_front());
        let log = self.log.clone();
        Reply::written(
            Made::new("test", "test", "paced"),
            move |mut writer| async move {
                let Some(Script { steps, mut gate }) = script else {
                    return;
                };
                for (at, step) in steps.into_iter().enumerate() {
                    if let Some(gate) = gate.as_mut()
                        && gate.next().await.is_none()
                    {
                        return;
                    }
                    note(&log, format!("chunk {}", at + 1));
                    let sent = match step {
                        Step::Text(text) => writer.text(text).await,
                        Step::Call(tool, args) => writer.tool_call(tool, args).await,
                        Step::Fail => {
                            writer.error(stream_truncated()).await.ok();
                            return;
                        }
                    };
                    if sent.is_err() {
                        return;
                    }
                }
                note(&log, "end".to_owned());
                writer.finish(Finish::default()).await.ok();
            },
        )
    }
}

#[derive(Deserialize, JsonSchema)]
struct PathArgs {
    path: String,
}

/// `read`, logging each path it reads, and publishing it as read.
struct ReadStub(Log);

impl Tool for ReadStub {
    const NAME: &'static str = "read";
    type Args = PathArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Read a file.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<PathArgs>()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: PathArgs,
    ) -> Result<String, ToolExecutionError> {
        note(&self.0, format!("read {}", args.path));
        let read = FileRead::new(args.path.clone(), ContentHash("h".to_owned()));
        context.insert_result(FilesRead(vec![read]))?;
        Ok(format!("contents of {}", args.path))
    }
}

#[derive(Deserialize, JsonSchema)]
struct ShellArgs {
    command: String,
}

/// `shell`, logging each command it runs.
struct ShellStub(Log);

impl Tool for ShellStub {
    const NAME: &'static str = "shell";
    type Args = ShellArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Run a command.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<ShellArgs>()
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        args: ShellArgs,
    ) -> Result<String, ToolExecutionError> {
        note(&self.0, format!("shell {}", args.command));
        Ok("[exit code 0]".to_owned())
    }
}

/// Another plugin's rewrite: `read a.txt` reads `src/a.txt`.
fn rewrite_reads(mut calls: Query<&mut ToolRequest, (Added<Admitted>, ToolCalls)>) {
    for mut request in &mut calls {
        let function = &mut request.call.function;
        if function.name.as_str() == "read"
            && let Some(path) = function.arguments.get_mut("path")
            && path == "a.txt"
        {
            *path = json!("src/a.txt");
        }
    }
}

/// The reply under test: `read a.txt` complete at chunk 3, `second` at
/// chunk 6, then `tail`.
fn reply_steps(second: Step, tail: Vec<Step>) -> Vec<Step> {
    let mut steps = vec![
        Step::Text("I will check"),
        Step::Text(" both files."),
        Step::Call("read", json!({ "path": "a.txt" })),
        Step::Text(" Then"),
        Step::Text(" the other."),
        second,
    ];
    steps.extend(tail);
    steps
}

/// Twenty more chunks of text.
fn explanation() -> Vec<Step> {
    (0..20).map(|_| Step::Text(" More explanation.")).collect()
}

/// The kernel, this plugin, the stub tools and the rewrite, and one agent
/// on a paced model that answers `first` (gated by the returned sender)
/// and then `Done.`; the agent was asked to compare the files.
fn start(first: Vec<Step>) -> (App, Receiver<()>, Paced, UnboundedSender<()>) {
    let (gate, permits) = unbounded();
    let paced = Paced::default();
    if let Ok(mut scripts) = paced.scripts.lock() {
        scripts.push_back(Script {
            steps: first,
            gate: Some(permits),
        });
        scripts.push_back(Script {
            steps: vec![Step::Text("Done.")],
            gate: None,
        });
    }
    let reads = ToolOptions {
        footprint: Some(Touches::Reads(Scope::Anything)),
    };
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((AgentPlugin, SessionLogPlugin, EarlyToolStartPlugin))
        .add_tool_with(ReadStub(paced.log.clone()), reads)
        .add_tool_with(ShellStub(paced.log.clone()), ToolOptions::default())
        .add_systems(AgentPass, rewrite_reads.in_set(PrepareSystems::Input))
        .init_resource::<ReadFacts>()
        .add_observer(note_read_facts)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    let spec = Catalog::builtin()
        .resolve("deepseek/deepseek-flash")
        .map(|spec| spec.shared());
    if let Ok(spec) = spec {
        let handler = Handler(ErasedHandler::new(paced.clone()));
        world.entity_mut(agent).insert(Connection { spec, handler });
    }
    world.trigger(Deliver::new(
        agent,
        "Compare a.txt and b.txt.",
        DeliveryMode::Steer,
    ));
    (app, wakes, paced, gate)
}

/// Lets `chunks` more chunks of the reply through.
fn release(gate: &UnboundedSender<()>, chunks: usize) {
    for _ in 0..chunks {
        gate.unbounded_send(()).ok();
    }
}

/// The tool results each request sent, in order.
fn results(paced: &Paced) -> Vec<Vec<String>> {
    let requests = paced.requests.lock().map(|r| r.clone()).unwrap_or_default();
    requests
        .iter()
        .map(|request| {
            let mut texts = Vec::new();
            for message in &request.chat_history {
                if let ChatMessage::User { content } = &**message {
                    for item in content {
                        if let UserContent::ToolResult(result) = item {
                            texts.extend(result.content.iter().map(|part| match part {
                                ToolResultContent::Text(text) => text.text.clone(),
                                other => format!("{other:?}"),
                            }));
                        }
                    }
                }
            }
            texts
        })
        .collect()
}

/// The files each settled call of a reply (not an early one) published as
/// read, in the order they settled.
#[derive(Resource, Default)]
struct ReadFacts(Vec<Vec<String>>);

fn note_read_facts(
    settled: On<Add<Settled>>,
    calls: Query<Option<&Published>, (ToolCalls, Without<StartedEarly>)>,
    mut facts: ResMut<ReadFacts>,
) {
    let Ok(published) = calls.get(settled.entity) else {
        return;
    };
    let read = published.and_then(|published| published.0.get::<FilesRead>().ok().flatten());
    let paths = read.into_iter().flat_map(|read| read.0);
    let paths = paths.map(|file| file.path);
    facts.0.push(paths.collect());
}

fn ended(world: &World) -> bool {
    !world.resource::<Ended>().0.is_empty()
}

#[test]
fn safe_calls_start_as_they_complete_and_answer_in_order() {
    let (mut app, wakes, paced, gate) = start(reply_steps(
        Step::Call("read", json!({ "path": "b.txt" })),
        explanation(),
    ));
    let log = paced.log.clone();
    release(&gate, 3);
    run_until(&mut app, &wakes, |_| {
        at(&logged(&log), "read src/a.txt").is_some()
    });
    let seen = logged(&log);
    // The rewrite applied, before chunk 4 was even let through.
    assert!(at(&seen, "read src/a.txt").is_some(), "{seen:?}");
    assert!(at(&seen, "chunk 4").is_none(), "{seen:?}");
    release(&gate, 3);
    run_until(&mut app, &wakes, |_| {
        at(&logged(&log), "read b.txt").is_some()
    });
    let seen = logged(&log);
    assert!(at(&seen, "read b.txt").is_some(), "{seen:?}");
    assert!(at(&seen, "chunk 7").is_none(), "{seen:?}");
    release(&gate, 20);
    run_until(&mut app, &wakes, ended);
    let seen = logged(&log);
    let reads = seen.iter().filter(|line| line.starts_with("read ")).count();
    assert_eq!(reads, 2, "each read ran once: {seen:?}");
    let sent = results(&paced);
    let next = sent.get(1).cloned().unwrap_or_default();
    assert_eq!(next.len(), 2, "{sent:?}");
    let first = next.first().map(String::as_str).unwrap_or_default();
    let second = next.get(1).map(String::as_str).unwrap_or_default();
    assert!(first.contains("contents of src/a.txt"), "{next:?}");
    assert!(second.contains("contents of b.txt"), "{next:?}");
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let stats = app.world().get::<EarlyStats>(agent);
    assert_eq!(stats.map(|stats| stats.calls), Some(2), "{stats:?}");
    // The reply's calls carry what their early calls published.
    let facts = &app.world().resource::<ReadFacts>().0;
    assert_eq!(facts, &[vec!["src/a.txt"], vec!["b.txt"]]);
}

#[test]
fn a_shell_call_waits_for_the_end_of_the_reply() {
    let (mut app, wakes, paced, gate) = start(reply_steps(
        Step::Call("shell", json!({ "command": "ls" })),
        explanation(),
    ));
    let log = paced.log.clone();
    release(&gate, 3);
    run_until(&mut app, &wakes, |_| {
        at(&logged(&log), "read src/a.txt").is_some()
    });
    assert!(at(&logged(&log), "chunk 4").is_none(), "{:?}", logged(&log));
    release(&gate, 23);
    run_until(&mut app, &wakes, ended);
    let seen = logged(&log);
    let (end, shell) = (at(&seen, "end"), at(&seen, "shell ls"));
    assert!(end.is_some() && shell > end, "{seen:?}");
    let sent = results(&paced);
    assert_eq!(sent.get(1).map(Vec::len), Some(2), "{sent:?}");
}

#[test]
fn a_failed_stream_leaves_no_trace_of_its_early_calls() {
    let mut steps = reply_steps(Step::Text(" Still"), Vec::new());
    steps.extend((0..3).map(|_| Step::Text(" going.")));
    steps.push(Step::Fail);
    let (mut app, wakes, paced, gate) = start(steps);
    let log = paced.log.clone();
    release(&gate, 3);
    run_until(&mut app, &wakes, |_| {
        at(&logged(&log), "read src/a.txt").is_some()
    });
    assert!(at(&logged(&log), "chunk 4").is_none(), "{:?}", logged(&log));
    release(&gate, 7);
    run_until(&mut app, &wakes, ended);
    let seen = logged(&log);
    assert!(at(&seen, "chunk 10").is_some(), "{seen:?}");
    let sent = results(&paced);
    assert_eq!(sent.len(), 2, "the reply was asked for again: {sent:?}");
    assert!(
        sent.iter().all(Vec::is_empty),
        "no result was sent: {sent:?}"
    );
    let retried = paced.requests.lock().map(|r| r.clone()).unwrap_or_default();
    let calls = |request: &CompletionRequest| {
        request.chat_history.iter().any(|message| match &**message {
            ChatMessage::Assistant(answer) => answer.tool_calls().next().is_some(),
            _ => false,
        })
    };
    assert!(!retried.iter().any(calls), "no call was sent back");
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let conversation = app.world().get::<Conversation>(agent);
    let traced = conversation.is_some_and(|conversation| {
        conversation
            .messages()
            .iter()
            .any(|message| match &**message {
                ChatMessage::User { content } => content
                    .iter()
                    .any(|item| matches!(item, UserContent::ToolResult(_))),
                ChatMessage::Assistant(answer) => answer.tool_calls().next().is_some(),
                _ => false,
            })
    });
    assert!(!traced, "the conversation has no early call");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(EarlyToolStartPlugin);
}

//! The effect corpus's two interpreters and the program table they share.
//!
//! Every golden effect log under `fixtures/*.effects.json` is one row of
//! one matrix: a [`Program`] the producing root test built verbatim,
//! replayed here by the bus-driven engine and by a hand driver of
//! `AgentRun` with **no provider, no tool, no memory and no index behind
//! any key**. The oracle is the record as data — kind, outcome and, for a
//! stream recorded with its events, the event sequence — position by
//! position. See `golden_replay.rs` for the original corpus and the
//! `corpus_*.rs` modules for the matrices.
//!
//! # The dimensions of an effect trace
//!
//! What an agent program can ask the bus, and how it is served. Every
//! matrix module prunes this table and says why.
//!
//! | axis | values |
//! |---|---|
//! | completion transport | unary · streamed, events dropped · streamed, events kept |
//! | tool shape | none · one call then answer · two calls in one turn · two turns · zero-arg tool · a tool that errors |
//! | tool id wire | provider id (anthropic) · id-less, minted `tool-<n>` (gemini) · dual `call_id`/`item_id` (openai) |
//! | serving | `serial_per_handler` false · true; `tool_concurrency` 1 · 2; capacities default · 1 |
//! | memory | none · `Load` + `Append` · `Load` of an empty conversation |
//! | retrieval | none · `dynamic_context(n, index)` (`TopN`) · `retrieved_tools(n, index, toolset)` (`TopNIds`) · both |
//! | embedding, rerank | never dispatched by the agent: an index embeds its query inside the handler (`RetrieveAdapter`), and nothing in `rig-agent` reranks; a host dispatches those families over its own bus |
//! | hooks | none · observe-only · `on_dispatch` → `Patch` · `Deny` · `on_outcome` → `Replace` · `on_invalid_tool_call` → `Retry` · `on_completion_call` → request patch · a hook that dispatches through `HookContext` |
//! | model routing | one model · `model_route` with `on_model_select` choosing the other |
//! | output | text · `output_schema` |
//! | bus ownership | own bus (`bus` in the header) · a host's bus via `over_bus` (`bus: None`) |
//! | run continuation | one run · serialize mid-run, resume on a fresh bus |
//! | outcome kind | success · `Cancelled` · handler error (`ErrorReport`) · a divergence (refused) |
//!
//! # What the original ten goldens cover
//!
//! Unary completion; memory load and append; a streamed turn with events
//! and one tool; two tools in one turn under serial serving; two tool-call
//! turns on id-less and on dual-id wires; an invalid call retried by a
//! hook; a consumer cancel. They do not cover retrieval, a hook that
//! patches, denies or replaces, model routing, structured output, a host's
//! bus, a resumed run, a handler error, `tool_choice`, `max_tokens`,
//! `additional_params`, static context, an appended or absent preamble, or
//! a prior history. Those are the matrices.

#![allow(dead_code)] // every test target uses a different subset

use std::time::Duration;

use futures::StreamExt;
use rig_agent::{
    AgentBuilder, AgentHook, HookContext,
    agent::{MultiTurnStreamItem, StreamingError},
    completion::PromptError,
    run::{
        AgentRun, AgentRunStep, InvalidToolCallAction, InvalidToolCallContext, ModelTurn,
        ModelTurnOutcome, PendingToolCall, RunSpec, StreamedTurnAssembler, prepare_request,
    },
    tool::{RegisteredTool, server::ToolServer},
};
use rig_bus::{Bus, Dispatcher, MemoryHandle, ModelHandle, ToolHandle};
use rig_core::{
    completion::{CompletionRequestBuilder, Document},
    effect::{EffectFamily, EffectRecord, HandlerKey},
    id::ConversationId,
    message::ToolChoice,
    message::{Message, UserContent},
    tool::ToolContext,
    transcript::tool_result_output,
};
use rig_effect_log::{EffectLog, EffectLogRecorder, EffectLogReplayer};

/// A hook the producer added, by type: the header names hooks by their
/// type's last path segment, so the replay's hook is a type of the same
/// name (defined here) making the same decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hook {
    /// `on_invalid_tool_call` → retry once with feedback naming `add`.
    RetryUnknownTool,
}

/// How the producer's run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ending {
    /// A final answer, the last completion's text.
    Answer,
    /// `PromptError::MaxTurnsError`: the model-call budget ran out with the
    /// model still calling tools (a per-run `tool_choice` that forces a
    /// call does this). Every record is a success; the run is not.
    MaxTurns,
}

/// The producer's tool choice, as data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    Auto,
    None,
    Required,
    Specific(&'static str),
}

impl Choice {
    pub fn tool_choice(self) -> ToolChoice {
        match self {
            Self::Auto => ToolChoice::Auto,
            Self::None => ToolChoice::None,
            Self::Required => ToolChoice::Required,
            Self::Specific(name) => ToolChoice::Specific {
                function_names: vec![name.to_owned()],
            },
        }
    }
}

/// One golden's program: what the producing root test built, verbatim.
pub struct Program {
    pub fixture: &'static str,
    pub owner: &'static str,
    /// `None` is `without_preamble()`.
    pub preamble: Option<&'static str>,
    /// `append_preamble(doc)` after `preamble`.
    pub append_preamble: Option<&'static str>,
    /// `context(doc)` static documents, in order.
    pub context: &'static [&'static str],
    pub prompt: &'static str,
    /// The history the runner was given (`history(..)`), if any.
    pub history: Option<fn() -> Vec<Message>>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub additional_params: Option<fn() -> serde_json::Value>,
    pub tool_choice: Option<Choice>,
    /// `output_schema_raw(schema)`.
    pub output_schema: Option<fn() -> serde_json::Value>,
    /// The builder's `default_max_turns`, part of the run spec the header
    /// hashes; a runner-level `max_turns` is not.
    pub default_max_turns: Option<usize>,
    pub max_turns: Option<usize>,
    pub tool_concurrency: Option<usize>,
    /// The producer ran `stream_prompt`: the model is asked for a stream.
    pub streamed: bool,
    /// The producer attached conversation memory under this id.
    pub conversation: Option<&'static str>,
    /// The producer's hooks, in registration order.
    pub hooks: &'static [Hook],
    pub invalid_retries: usize,
    /// The producer dropped the stream after its first text delta: the
    /// one record is a `Cancelled` completion and the run never finishes.
    /// On replay the replayer answers that record as the cancel it was, so
    /// the consumer sees the cancel as its first item, never a delta.
    pub cancel_after_first_delta: bool,
    pub ending: Ending,
}

impl Program {
    pub const DEFAULT: Program = Program {
        fixture: "",
        owner: "golden",
        preamble: Some(""),
        append_preamble: None,
        context: &[],
        prompt: "",
        history: None,
        temperature: None,
        max_tokens: None,
        additional_params: None,
        tool_choice: None,
        output_schema: None,
        default_max_turns: None,
        max_turns: None,
        tool_concurrency: None,
        streamed: false,
        conversation: None,
        hooks: &[],
        invalid_retries: 0,
        cancel_after_first_delta: false,
        ending: Ending::Answer,
    };

    /// The preamble the run spec holds: the builder's, with any appended
    /// document.
    fn spec_preamble(&self) -> Option<String> {
        let base = self.preamble.map(str::to_owned);
        match self.append_preamble {
            Some(doc) => Some(format!("{}\n{doc}", base.unwrap_or_default())),
            None => base,
        }
    }

    fn static_context(&self) -> Vec<Document> {
        self.context
            .iter()
            .enumerate()
            .map(|(n, text)| Document {
                id: format!("static_doc_{n}"),
                text: (*text).to_owned(),
                additional_props: Default::default(),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The hooks, by name.

/// The producer's hook, verbatim (`tests/common/goldens.rs`): the header
/// names it by type name, so the replay's hook is the same type.
struct RetryUnknownTool;

impl AgentHook for RetryUnknownTool {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        context: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(retry_feedback(&context.tool_name))
    }
}

fn retry_feedback(tool_name: &str) -> InvalidToolCallAction {
    InvalidToolCallAction::Retry {
        feedback: format!("there is no tool named {tool_name}; use add"),
    }
}

fn add_hooks<S>(mut builder: AgentBuilder<S>, hooks: &[Hook]) -> AgentBuilder<S> {
    for hook in hooks {
        builder = match hook {
            Hook::RetryUnknownTool => builder.add_hook(RetryUnknownTool),
        };
    }
    builder
}

// ---------------------------------------------------------------------------
// Goldens and the oracle.

pub async fn within<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("a replay never hangs")
}

pub fn golden(fixture: &str) -> EffectLog {
    let path = format!(
        "{}/fixtures/{fixture}.effects.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("the golden fixture is committed");
    serde_json::from_str(&text).expect("the golden fixture loads")
}

/// A record as data: its kind, its outcome and its events, if any.
pub fn as_data(record: &EffectRecord) -> serde_json::Value {
    serde_json::json!({
        "key": record.key,
        "kind": record.kind,
        "outcome": record.outcome,
        "events": record.events,
    })
}

pub fn assert_same_records(replayed: &EffectLog, log: &EffectLog, interpreter: &str) {
    let replayed: Vec<_> = replayed.iter().map(as_data).collect();
    let recorded: Vec<_> = log.iter().map(as_data).collect();
    for (position, (got, want)) in replayed.iter().zip(&recorded).enumerate() {
        assert_eq!(
            got, want,
            "{interpreter}: record {position} differs from the golden"
        );
    }
    assert_eq!(
        replayed.len(),
        recorded.len(),
        "{interpreter}: the golden has {} records, the replay {}",
        recorded.len(),
        replayed.len()
    );
}

pub fn keeps_events(log: &EffectLog) -> bool {
    log.iter().any(|record| record.events.is_some())
}

pub fn golden_answer(log: &EffectLog) -> String {
    log.iter()
        .rev()
        .find_map(|record| match &record.outcome {
            Ok(rig_core::effect::Outcome::Completion(response)) => Some(
                response
                    .choice
                    .iter()
                    .filter_map(|content| match content {
                        rig_core::message::AssistantContent::Text(text) => Some(text.text.clone()),
                        _ => None,
                    })
                    .collect::<String>(),
            ),
            _ => None,
        })
        .expect("the golden ends in a completion")
}

/// The bus, with the golden's policy, the model replayer registered and a
/// recorder attached; the tool replayers in a server the agent advertises
/// from (the required row's tools, dispatched or not).
pub struct Replay {
    pub log: EffectLog,
    pub dispatcher: Dispatcher,
    pub registrar: rig_bus::Registrar,
    pub recorder: EffectLogRecorder,
    pub driver: tokio::task::JoinHandle<()>,
    pub model_key: HandlerKey,
    pub memory_key: HandlerKey,
}

impl Replay {
    pub fn open(program: &Program) -> Self {
        let log = golden(program.fixture);
        EffectLogReplayer::check_header(&log).expect("a current format");
        let bus = log.header.bus.expect("the header names the bus policy");
        let (dispatcher, registrar, mut driver) = Bus::channel_with(bus);
        let model_key = HandlerKey::from(format!("{}/model:default", program.owner));
        let memory_key = HandlerKey::from(format!("{}/memory", program.owner));
        let model = EffectLogReplayer::for_key(&log, &model_key).expect("the model's records");
        driver
            .register_erased(
                model_key.clone(),
                rig_core::serve::ErasedHandler::new(model),
            )
            .expect("a fresh key");
        let recorder = if keeps_events(&log) {
            EffectLogRecorder::keeping_stream_events()
        } else {
            EffectLogRecorder::new()
        };
        driver.record_to(recorder.clone());
        let driver = tokio::spawn(driver);
        Self {
            log,
            dispatcher,
            registrar,
            recorder,
            driver,
            model_key,
            memory_key,
        }
    }

    pub fn tool_keys(&self) -> Vec<HandlerKey> {
        self.log
            .header
            .required
            .iter()
            .filter(|(_, family)| **family == EffectFamily::Tool)
            .map(|(key, _)| key.clone())
            .collect()
    }

    pub fn tool_server(&self) -> rig_agent::tool::server::ToolServerHandle {
        let server = ToolServer::new().run();
        for key in self.tool_keys() {
            let replayer =
                EffectLogReplayer::for_key(&self.log, &key).expect("a required tool is described");
            server.add_registered_tool(
                RegisteredTool::from_handler(replayer).expect("a tool-family replayer"),
            );
        }
        server
    }

    pub async fn close(self) -> EffectLog {
        drop((self.dispatcher, self.registrar));
        within(self.driver).await.expect("driver task");
        self.recorder.take()
    }
}

// ---------------------------------------------------------------------------
// The bus engine.

pub async fn bus_engine_reproduces(program: &Program) {
    let replay = Replay::open(program);
    let server = replay.tool_server();
    let mut builder = AgentBuilder::over_bus(
        replay.dispatcher.clone(),
        replay.registrar.clone(),
        program.owner,
        replay.model_key.clone(),
    )
    .name(program.owner)
    .tool_server_handle(server);
    builder = match program.preamble {
        Some(preamble) => builder.preamble(preamble),
        None => builder.without_preamble(),
    };
    if let Some(doc) = program.append_preamble {
        builder = builder.append_preamble(doc);
    }
    for doc in program.context {
        builder = builder.context(*doc);
    }
    if let Some(temperature) = program.temperature {
        builder = builder.temperature(temperature);
    }
    if let Some(max_tokens) = program.max_tokens {
        builder = builder.max_tokens(max_tokens);
    }
    if let Some(params) = program.additional_params {
        builder = builder.additional_params(params());
    }
    if let Some(choice) = program.tool_choice {
        builder = builder.tool_choice(choice.tool_choice());
    }
    if let Some(schema) = program.output_schema {
        builder = builder.output_schema_raw(
            serde_json::from_value(schema()).expect("the producer's schema is a schema"),
        );
    }
    if let Some(default_max_turns) = program.default_max_turns {
        builder = builder.default_max_turns(default_max_turns);
    }
    builder = add_hooks(builder, program.hooks);
    if let Some(conversation) = program.conversation {
        let memory = EffectLogReplayer::for_key(&replay.log, &replay.memory_key)
            .expect("the conversation's records");
        builder = builder.memory_handler(memory).conversation(conversation);
    }
    let agent = builder.build();
    agent
        .check_replayable(&replay.log)
        .expect("the same program as the one recorded");

    let output = if program.streamed {
        let mut runner = agent.stream_prompt(program.prompt);
        if let Some(history) = program.history {
            runner = runner.history(history());
        }
        if let Some(max_turns) = program.max_turns {
            runner = runner.max_turns(max_turns);
        }
        if let Some(concurrency) = program.tool_concurrency {
            runner = runner.tool_concurrency(concurrency);
        }
        let mut stream = runner.stream().await;
        let mut output = None;
        let mut max_turns_reached = false;
        while let Some(item) = within(stream.next()).await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    output = Some(response.output);
                }
                Err(StreamingError::Report(report))
                    if program.cancel_after_first_delta
                        && report.kind == rig_core::error::ErrorKind::Cancelled =>
                {
                    break;
                }
                Err(StreamingError::Prompt(error))
                    if program.ending == Ending::MaxTurns
                        && matches!(*error, PromptError::MaxTurnsError { .. }) =>
                {
                    max_turns_reached = true;
                }
                Err(error) => {
                    panic!("the replayer answered every request it recognised: {error:?}")
                }
                Ok(_) => {}
            }
        }
        drop(stream);
        if program.cancel_after_first_delta {
            // The driver resolves the cancelled dispatch on its own task.
            for _ in 0..64 {
                tokio::task::yield_now().await;
            }
            None
        } else if program.ending == Ending::MaxTurns {
            assert!(max_turns_reached, "the run ends in MaxTurnsError");
            None
        } else {
            Some(output.expect("the stream yields a final response"))
        }
    } else {
        let mut runner = agent.prompt(program.prompt);
        if let Some(history) = program.history {
            runner = runner.history(history());
        }
        if let Some(max_turns) = program.max_turns {
            runner = runner.max_turns(max_turns);
        }
        if let Some(concurrency) = program.tool_concurrency {
            runner = runner.tool_concurrency(concurrency);
        }
        runner = runner.max_invalid_tool_call_retries(program.invalid_retries);
        match (within(runner.run()).await, program.ending) {
            (Ok(response), Ending::Answer) => Some(response.output),
            (Err(PromptError::MaxTurnsError { .. }), Ending::MaxTurns) => None,
            (Ok(response), Ending::MaxTurns) => {
                panic!("the run ends in MaxTurnsError, not an answer: {response:?}")
            }
            (Err(error), _) => {
                panic!("the replayer answered every request it recognised: {error:?}")
            }
        }
    };
    if let Some(output) = output {
        assert_eq!(output, golden_answer(&replay.log));
    }
    drop(agent);
    let log = replay.log.clone();
    let replayed = replay.close().await;
    assert_same_records(&replayed, &log, "bus engine");
}

// ---------------------------------------------------------------------------
// The hand driver: `AgentRun` stepped by this test, every step dispatched
// over the bus handles the engine would use.

/// The tool handles the program advertises, by tool name.
pub fn tool_handles(replay: &Replay) -> Vec<(String, ToolHandle)> {
    replay
        .tool_keys()
        .into_iter()
        .map(|key| {
            let handle: ToolHandle = replay.dispatcher.handle(&key).expect("a tool handle");
            (handle.name(), handle)
        })
        .collect()
}

pub async fn call_tools(
    calls: Vec<PendingToolCall>,
    tools: &[(String, ToolHandle)],
    concurrency: usize,
) -> Vec<UserContent> {
    let dispatch = |call: PendingToolCall| async move {
        if let Some(preresolved) = call.preresolved_result {
            return preresolved;
        }
        let name = call.tool_call.function.name.clone();
        let (_, handle) = tools
            .iter()
            .find(|(tool, _)| *tool == name)
            .unwrap_or_else(|| panic!("the program advertises `{name}`"));
        let args = call.tool_call.function.arguments.to_string();
        let answer = within(handle.call(name.clone(), args, ToolContext::new()))
            .await
            .expect("the replayer answered the recorded call");
        let output = answer
            .result
            .into_result()
            .expect("every tool in the corpus succeeded");
        // The engine's own shaping of a result (`rig_core::transcript`).
        tool_result_output(
            call.tool_call.id.clone(),
            call.tool_call.provider.clone(),
            name,
            output,
        )
    };
    futures::stream::iter(calls)
        .map(dispatch)
        .buffered(concurrency.max(1))
        .collect()
        .await
}

/// The run spec the producer's builder and runner amount to.
pub fn run_spec(program: &Program) -> RunSpec {
    RunSpec {
        preamble: program.spec_preamble(),
        static_context: program.static_context(),
        additional_params: program.additional_params.map(|params| params()),
        max_tokens: program.max_tokens,
        temperature: program.temperature,
        tool_choice: program.tool_choice.map(Choice::tool_choice),
        max_turns: program.max_turns,
        max_invalid_tool_call_retries: program.invalid_retries,
        output_schema: program.output_schema.map(|schema| schema()),
        ..RunSpec::new()
    }
}

pub async fn hand_driver_reproduces(program: &Program) {
    let replay = Replay::open(program);
    let server = replay.tool_server();
    server.attach(&replay.registrar);
    let tools = tool_handles(&replay);
    let model: ModelHandle = replay
        .dispatcher
        .handle(&replay.model_key)
        .expect("the model");
    let memory: Option<(MemoryHandle, ConversationId)> = program.conversation.map(|id| {
        let replayer = EffectLogReplayer::for_key(&replay.log, &replay.memory_key)
            .expect("the conversation's records");
        replay
            .registrar
            .register_erased(
                replay.memory_key.clone(),
                rig_core::serve::ErasedHandler::new(replayer),
            )
            .expect("a fresh key");
        let handle: MemoryHandle = replay
            .dispatcher
            .handle(&replay.memory_key)
            .expect("the memory");
        (handle, ConversationId::from(id))
    });
    let spec = run_spec(program);
    // Explicit history bypasses memory for the run, as the runner does.
    let history = match (program.history, &memory) {
        (Some(history), _) => Some(history()),
        (None, Some((handle, id))) => Some(
            within(handle.load(id.clone()))
                .await
                .expect("the replayer answered the load"),
        ),
        (None, None) => None,
    };
    let definitions = server.static_tool_defs();
    let mut run = AgentRun::from_spec(&spec, program.prompt, history);
    let response = loop {
        let step = match (run.next_step(), program.ending) {
            (Ok(step), _) => step,
            (Err(PromptError::MaxTurnsError { .. }), Ending::MaxTurns) => break None,
            (Err(error), _) => panic!("a step: {error:?}"),
        };
        match step {
            AgentRunStep::CallModel {
                prompt,
                history,
                turn,
            } => {
                let prepared = prepare_request(
                    &spec,
                    &model.capabilities(),
                    &history,
                    definitions.clone(),
                    run.output_tool_name(),
                    None,
                )
                .expect("prepared");
                run.set_output_tool_name(prepared.output_tool_name.clone());
                run.advertise_tools(turn, prepared.tools.clone());
                let executable = prepared.executable_tool_names.clone();
                let allowed = prepared.allowed_tool_names.clone();
                let request = prepared
                    .apply(CompletionRequestBuilder::unbound(prompt))
                    .build();
                let turn = if program.streamed {
                    let mut stream = model.stream(request);
                    let mut assembler = StreamedTurnAssembler::new(executable, allowed);
                    while let Some(event) = within(stream.next()).await {
                        let event = match event {
                            Ok(event) => event,
                            Err(report)
                                if program.cancel_after_first_delta
                                    && report.kind == rig_core::error::ErrorKind::Cancelled =>
                            {
                                break;
                            }
                            Err(report) => {
                                panic!("the replayer re-emitted the recorded stream: {report:?}")
                            }
                        };
                        assembler.ingest(&event).expect("a well-formed stream");
                    }
                    if program.cancel_after_first_delta {
                        drop(stream);
                        for _ in 0..64 {
                            tokio::task::yield_now().await;
                        }
                        break None;
                    }
                    let usage = stream.usage();
                    let snapshot = stream.snapshot();
                    let streamed = assembler.finish(stream.message_id.clone(), &snapshot);
                    ModelTurn::new(
                        streamed.message_id,
                        streamed.choice,
                        usage,
                        streamed.executable_tool_names,
                        streamed.allowed_tool_names,
                    )
                } else {
                    let response = within(model.complete(request))
                        .await
                        .expect("the replayer recognised the request");
                    ModelTurn::from_response_parts(&response, executable, allowed)
                };
                let mut outcome = run.model_response(turn).expect("a model turn");
                while let ModelTurnOutcome::NeedsResolution(invalid) = outcome {
                    assert!(
                        program.hooks.contains(&Hook::RetryUnknownTool),
                        "only the recovery program sees an invalid call"
                    );
                    outcome = run
                        .resolve_invalid_tool_call(retry_feedback(&invalid.tool_name))
                        .expect("the retry is resolved");
                }
            }
            AgentRunStep::CallTools { calls } => {
                let results =
                    call_tools(calls, &tools, program.tool_concurrency.unwrap_or(1)).await;
                run.tool_results(results).expect("results for every call");
            }
            AgentRunStep::Done(response) => break Some(response),
        }
    };
    let Some(response) = response else {
        drop((model, tools, memory));
        let log = replay.log.clone();
        let replayed = replay.close().await;
        assert_same_records(&replayed, &log, "hand driver");
        return;
    };
    if let (Some((handle, id)), None) = (&memory, program.history) {
        within(handle.append(id.clone(), response.messages.clone().unwrap_or_default()))
            .await
            .expect("the replayer answered the append");
    }
    assert_eq!(response.output, golden_answer(&replay.log));
    drop((model, tools, memory));
    let log = replay.log.clone();
    let replayed = replay.close().await;
    assert_same_records(&replayed, &log, "hand driver");
}

/// Both interpreters, as two tests each, for the rows named: `test: PROGRAM`.
#[macro_export]
macro_rules! both_interpreters {
    ($($test:ident: $program:ident),* $(,)?) => {
        mod bus_engine {
            $(
                #[tokio::test]
                async fn $test() {
                    $crate::corpus::bus_engine_reproduces(&super::$program).await;
                }
            )*
        }
        mod hand_driver {
            $(
                #[tokio::test]
                async fn $test() {
                    $crate::corpus::hand_driver_reproduces(&super::$program).await;
                }
            )*
        }
    };
}

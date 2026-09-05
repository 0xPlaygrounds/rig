//! The workspace-maintenance consumer, shared by the executable and tests.
//! Host inputs are configuration; observations are outputs, never replay inputs.

#![allow(
    dead_code,
    reason = "the CLI and each integration suite use different parts of the shared consumer"
)]

pub(crate) mod artifacts;
mod custom;
mod identity;
pub(crate) mod persistence;
mod providers;
mod repair;
use crate::cassettes::consumer_registry as registry;
pub(crate) mod runner;
mod scheduled;
mod workspace;

use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use bevy_app::App;
use bevy_ecs::{prelude::*, schedule::LogLevel};
use rig_core::{
    completion::{CompletionResponse, ModelRef, ProviderCapabilities, Usage},
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    message::{AssistantContent, Message, UserContent},
    serve::{OutcomeSink, Serve, ServingPolicy},
    tool::{ContextValue, ToolContext, ToolOutput, ToolResult},
};
use rig_ecs::{
    agent::*,
    bus::{
        self, Bound, BusPlugin, BusSet, EffectLogResource, EffectOutcome, Handlers, InFlight,
        Issued, PendingEffect, Replay, RigSchedule, WorldOutcome,
    },
    systems::{AgentPlugin, spawn_run},
};
use rig_effect_log::{EffectLog, EffectLogRecorder};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub(crate) use registry::{Approval, Arrival, Case, Fault, Provider, cases};
use workspace::{INITIAL, TARGET, Workspace};

const MODEL: &str = "maintenance/model:default";
const PREAMBLE: &str = "You maintain a disposable project. First call read_file and search_file (needle: Helo). Next call propose_edit with content exactly Hello, Rig! followed by a newline. Next call apply_edit with an empty object. If approved, call validate_file. Finally report the tool results briefly. Do not claim an edit or validation happened without calling its tool. A denied edit ends the task. Use only the supplied tools, and do not repeat a completed operation.";
const PROMPT: &str = "Fix the greeting typo in greeting.txt, using the approval-controlled tools.";

#[derive(Debug, Serialize, Deserialize)]
struct WriteReceipt {
    operation: String,
    applied: bool,
}

impl ContextValue for WriteReceipt {
    const KEY: &'static str = "maintenance.write-receipt.v1";
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Runtime(#[from] ErrorReport),
    #[error(transparent)]
    Transport(#[from] rig_core::http_client::Error),
    #[error(transparent)]
    Client(#[from] rig_core::client::ProviderClientError),
    #[error("consumer invariant: {0}")]
    Invariant(String),
    #[error("invalid invocation: {0}")]
    Invocation(String),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct Observation {
    pub boundary: String,
    pub effect: Option<u64>,
    pub data: Value,
}

#[derive(Resource)]
struct Host {
    case: Case,
    workspace: Workspace,
    replay: bool,
    observations: Vec<Observation>,
    seen: BTreeMap<u64, (usize, bool)>,
    proposal: Option<String>,
    approval: Option<Approval>,
    validated: bool,
    failure: Option<String>,
    primary: Option<String>,
}

impl Host {
    fn observe(&mut self, boundary: &str, effect: Option<u64>, data: Value) {
        self.observations.push(Observation {
            boundary: boundary.into(),
            effect,
            data,
        });
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Evidence {
    pub effects: EffectLog,
    pub observations: Vec<Observation>,
    pub files: BTreeMap<String, String>,
    pub writes: usize,
    pub result: String,
    pub checkpoints: Vec<persistence::Checkpoint>,
}

/// Move transport futures onto the explicitly supplied Tokio runtime. Dropping
/// an ECS task aborts its transport task too, rather than detaching a live call.
pub(crate) struct TokioHandler<S> {
    pub handler: Arc<S>,
    pub runtime: tokio::runtime::Handle,
    failures: ExecutionFailures,
}

#[derive(Clone, Default, Resource)]
struct ExecutionFailures(Arc<std::sync::Mutex<Vec<String>>>);

impl ExecutionFailures {
    fn record(&self, id: u64) {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(format!(
                "handler task failed for effect {id}, even if it emitted a terminal answer"
            ));
    }
    fn check(&self) -> Result<(), Error> {
        match self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .first()
        {
            Some(error) => Err(Error::Invariant(error.clone())),
            None => Ok(()),
        }
    }
}

struct AbortOnDrop(tokio::task::AbortHandle);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl<S: Serve + 'static> Serve for TokioHandler<S> {
    type Family = S::Family;
    fn descriptor(&self) -> HandlerDescriptor {
        self.handler.descriptor()
    }
    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        let id = sink.id().as_u64();
        let handler = Arc::clone(&self.handler);
        let task = self.runtime.spawn(async move {
            handler.serve(kind, sink).await;
        });
        let _abort = AbortOnDrop(task.abort_handle());
        if task.await.is_err() {
            self.failures.record(id);
        }
    }
}

fn tool_family(name: &str) -> FamilyDescriptor {
    let (description, parameters) = match name {
        "read_file" => (
            "Read greeting.txt",
            json!({"type":"object","properties":{}}),
        ),
        "search_file" => (
            "Find matching lines in greeting.txt",
            json!({"type":"object","properties":{"needle":{"type":"string"}},"required":["needle"]}),
        ),
        "propose_edit" => (
            "Propose replacement contents for greeting.txt without writing",
            json!({"type":"object","properties":{"content":{"type":"string"}},"required":["content"]}),
        ),
        "apply_edit" => (
            "Apply the proposed edit after host approval",
            json!({"type":"object","properties":{}}),
        ),
        _ => (
            "Validate the greeting file after an approved write",
            json!({"type":"object","properties":{}}),
        ),
    };
    FamilyDescriptor::Tool {
        name: name.into(),
        description: description.into(),
        parameters,
        embedding: None,
    }
}

const TOOLS: [&str; 5] = [
    "read_file",
    "search_file",
    "propose_edit",
    "apply_edit",
    "validate_file",
];
fn tool_key(name: &str) -> String {
    format!("maintenance/tool:{name}")
}

fn bound(world: &mut World, key: &str) -> Result<Entity, Error> {
    world
        .query::<(Entity, &Bound)>()
        .iter(world)
        .find(|(_, b)| b.key.as_str() == key)
        .map(|(e, _)| e)
        .ok_or_else(|| Error::Invariant(format!("missing bound handler {key}")))
}

fn build(case: &Case, replay: Option<&EffectLog>) -> Result<App, Error> {
    build_with_replay(case, replay, Replay::policy_visible())
}

fn build_with_replay(case: &Case, replay: Option<&EffectLog>, mode: Replay) -> Result<App, Error> {
    let mut app = App::new();
    // A complete controlled group must fit before Collect is allowed to run.
    let serving = ServingPolicy {
        command_capacity: case.intake,
        serial_per_handler: case.serial_keys,
        stream_capacity: 4096,
    };
    app.add_plugins((
        BusPlugin::with_policy(serving).ambiguity_detection(LogLevel::Error),
        AgentPlugin::default(),
    ));
    app.finish();
    app.cleanup();
    app.init_resource::<scheduled::DeliveryControl>();
    app.init_resource::<ExecutionFailures>();
    persistence::install(app.world_mut())?;
    // Perturb entity/archetype history independently of logical effect identity.
    #[derive(Component)]
    struct Noise;
    for index in 0..(case.spawn_noise + usize::from(replay.is_some()) * 7) {
        let entity = app.world_mut().spawn(Noise).id();
        if index % 2 == 0 {
            app.world_mut().despawn(entity);
        }
    }
    app.insert_resource(Host {
        case: case.clone(),
        workspace: Workspace::new()?,
        replay: replay.is_some(),
        observations: Vec::new(),
        seen: BTreeMap::new(),
        proposal: None,
        approval: None,
        validated: false,
        failure: None,
        primary: None,
    });
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::keeping_stream_events());
    if let Some(log) = replay {
        Handlers::with(app.world_mut(), |h| mode.register(h, log))??;
    } else {
        custom::install(app.world_mut(), case)?;
        for name in TOOLS {
            Handlers::with(app.world_mut(), |h| {
                h.register_open(tool_key(name), tool_family(name))
            })??;
        }
        if case.identity_checks {
            Handlers::with(app.world_mut(), |h| {
                h.register_open(tool_key("unused_audit"), tool_family("unused_audit"))
            })??;
        }
    }
    custom::stamp(app.world_mut(), case, replay)?;
    app.world_mut().resource_mut::<Schedules>().add_systems(
        RigSchedule,
        (
            approval_policy.in_set(BusSet::Gate),
            lifecycle_input
                .before(rig_ecs::systems::RigSet::Advance)
                .before(bus::record::begin_delivery_pass),
            serve_tools.after(BusSet::Dispatch).before(BusSet::Collect),
            persistence::capture_early
                .after(BusSet::Dispatch)
                .before(serve_tools),
            (observe, persistence::capture)
                .chain()
                .after(BusSet::Collect)
                .before(BusSet::Judge),
        ),
    );
    Ok(app)
}

fn program(app: &mut App, case: &Case) -> Result<Entity, Error> {
    let model = bound(app.world_mut(), MODEL)?;
    let agent = app
        .world_mut()
        .spawn((
            Owner("maintenance".into()),
            Preamble(Some(PREAMBLE.into())),
            Temperature(None),
            MaxTokens(Some(512)),
            AdditionalParams(
                (case.provider == Provider::Gemini)
                    .then(|| json!({"generationConfig":{"thinkingConfig":{"thinkingBudget":0}}})),
            ),
            ToolChoiceSpec(None),
            Output::default(),
            DefaultMaxTurns(None),
            MaxTurns(8),
            InvalidCalls::default(),
            UsesModel(model),
            ToolPolicy {
                concurrency: case.concurrency,
            },
            PolicyVersion("maintenance/v1/collect-observation".into()),
        ))
        .id();
    for (order, name) in TOOLS.iter().enumerate() {
        let tool = bound(app.world_mut(), &tool_key(name))?;
        app.world_mut()
            .spawn((Grant(tool), Order(order as u64), ChildOf(agent)));
    }
    if case.identity_checks {
        let tool = bound(app.world_mut(), &tool_key("unused_audit"))?;
        app.world_mut()
            .spawn((Grant(tool), Order(TOOLS.len() as u64), ChildOf(agent)));
    }
    let run = spawn_run(app.world_mut(), agent, &[], PROMPT, case.stream, None);
    app.world_mut()
        .entity_mut(run)
        .insert(persistence::MaintenanceRun {
            operation: "greeting-fix-v1".into(),
        });
    let recorder = app.world().resource::<EffectLogResource>().0.clone();
    rig_ecs::replay::stamp_run(app.world_mut(), run, &recorder);
    custom::start(app.world_mut(), run, case)?;
    if case.interleaved {
        app.world_mut().spawn((
            PendingEffect::new(
                MODEL,
                EffectKind::Completion {
                    stream: true,
                    request: rig_core::completion::CompletionRequest {
                        model: None,
                        chat_history: vec![Message::user(
                            "Explain the maintenance task while its tools run.",
                        )],
                        documents: vec![],
                        tools: vec![],
                        temperature: None,
                        max_tokens: Some(512),
                        tool_choice: None,
                        additional_params: (case.fault == Fault::CancelBackground)
                            .then(|| json!({"synthetic_background_chunks":64})),
                        output_schema: None,
                        record_telemetry_content: false,
                    },
                },
            ),
            ChildOf(run),
        ));
    }
    Ok(run)
}

fn tool_answer(host: &mut Host, name: &str, args: &str) -> Result<Value, Error> {
    let args: Value = serde_json::from_str(args)?;
    match name {
        "read_file" => Ok(json!({"path":"greeting.txt", "content":host.workspace.read()?})),
        "search_file" => {
            let needle = args
                .get("needle")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::Invariant("search needs needle".into()))?;
            let contents = host.workspace.read()?;
            let lines: Vec<_> = contents
                .lines()
                .enumerate()
                .filter(|(_, line)| line.contains(needle))
                .map(|(index, text)| json!({"line":index + 1,"text":text}))
                .collect();
            Ok(json!({"matches":lines}))
        }
        "propose_edit" => {
            let content = args
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::Invariant("proposal needs content".into()))?;
            Ok(json!({"proposed":content,"path":"greeting.txt"}))
        }
        "apply_edit" => {
            if host.case.fault == Fault::WriteError {
                return Err(Error::Invariant("controlled write failure".into()));
            }
            if host.proposal.is_none() {
                return Err(Error::Invariant("write without observed proposal".into()));
            }
            let approval = host
                .approval
                .ok_or_else(|| Error::Invariant("write has no approval decision".into()))?;
            if approval != Approval::Approve {
                return Ok(json!({"applied":false,"decision":approval}));
            }
            let content = host
                .proposal
                .clone()
                .ok_or_else(|| Error::Invariant("proposal disappeared".into()))?;
            host.workspace.apply(&content)?;
            Ok(
                json!({"applied":true,"path":"greeting.txt","content":content,"operation":"greeting-fix-v1"}),
            )
        }
        "validate_file" => {
            let content = host.workspace.read()?;
            Ok(json!({"valid": content == TARGET, "content":content}))
        }
        _ => Err(Error::Invariant(format!("unexpected tool {name}"))),
    }
}

fn lifecycle_input(world: &mut World) {
    if world.resource::<Host>().case.fault == Fault::CancelBeforeServe {
        cancel_consumer(world, "before-serve");
    }
    let host = world.resource::<Host>();
    let remove = host.case.fault == Fault::RemoveModelBeforeDispatch
        || (host.case.fault == Fault::RemoveModelBetweenTurns && host.primary.is_some());
    if remove && let Ok(model) = bound(world, MODEL) {
        world.despawn(model);
        world
            .resource_mut::<Host>()
            .observe("input.remove-model", None, json!({"key":MODEL}));
        world.resource_mut::<bus::Progress>().mark();
    }
}

fn cancel_consumer(world: &mut World, boundary: &str) {
    let runs: Vec<_> = world
        .query_filtered::<Entity, (
            With<persistence::MaintenanceRun>,
            Without<Failed>,
            Without<Settled>,
        )>()
        .iter(world)
        .collect();
    for run in runs {
        world
            .resource_mut::<Host>()
            .observe("input.cancel", None, json!({"boundary":boundary}));
        world.entity_mut(run).insert(Cancelled(boundary.into()));
    }
}

fn approval_policy(world: &mut World) {
    if world.resource::<Host>().approval.is_some() {
        return;
    }
    let requested = world
        .query_filtered::<&PendingEffect, Without<Issued>>()
        .iter(world)
        .any(|effect| effect.key.as_str() == tool_key("apply_edit"));
    if requested {
        let decision = {
            let mut host = world.resource_mut::<Host>();
            let decision = host.case.approval;
            host.approval = Some(decision);
            host.observe("approval", None, json!({"decision":decision}));
            decision
        };
        if decision == Approval::Cancel {
            let runs: Vec<_> = world
                .query_filtered::<Entity, With<persistence::MaintenanceRun>>()
                .iter(world)
                .collect();
            for run in runs {
                world
                    .entity_mut(run)
                    .insert(Cancelled("host cancelled while awaiting approval".into()));
            }
        }
    }
}

fn serve_tools(world: &mut World) {
    if world.resource::<Host>().replay {
        return;
    }
    let mut pending: Vec<_> = world
        .query_filtered::<(Entity, &PendingEffect, &Issued), (
            With<InFlight>,
            Without<EffectOutcome>,
            Without<WorldOutcome>,
        )>()
        .iter(world)
        .filter_map(|(entity, effect, id)| match &effect.kind {
            EffectKind::ToolCall { args, .. } => {
                Some((id.0.as_u64(), entity, effect.key.clone(), args.clone()))
            }
            _ => None,
        })
        .collect();
    pending.sort_by_key(|(id, ..)| *id);
    for (_, entity, key, args) in pending {
        let name = key.as_str().strip_prefix("maintenance/tool:").unwrap_or("");
        let host = world.resource::<Host>();
        if host.primary.is_none() && matches!(name, "read_file" | "search_file") {
            match host.case.arrival {
                Arrival::ReadFirst if name != "read_file" => continue,
                Arrival::SearchFirst if name != "search_file" => continue,
                _ => (),
            }
        }
        let result = tool_answer(&mut world.resource_mut::<Host>(), name, &args)
            .map(|answer| Outcome::ToolResult {
                result: ToolResult::success(ToolOutput::json(answer)),
            })
            .map_err(|error| ErrorReport::new(ErrorKind::Request, error.to_string()));
        if name == "apply_edit" && result.is_ok() {
            let mut output = ToolContext::new();
            let applied = world.resource::<Host>().workspace.writes == 1;
            if let Err(error) = output.insert_result(WriteReceipt {
                operation: "greeting-fix-v1".into(),
                applied,
            }) {
                world.resource_mut::<Host>().failure = Some(error.to_string());
            }
            world.entity_mut(entity).insert(bus::ToolOutputs(output));
        }
        world.entity_mut(entity).insert(WorldOutcome::new(result));
    }
}

fn observe(world: &mut World) {
    let mut visible: Vec<_> = world
        .query::<(
            &Issued,
            &PendingEffect,
            Option<&bus::Streamed>,
            Option<&EffectOutcome>,
            Option<&bus::ToolOutputs>,
        )>()
        .iter(world)
        .map(|(id, effect, stream, outcome, published)| {
            (
                id.0.as_u64(),
                effect.key.clone(),
                stream.cloned(),
                outcome.cloned(),
                published.cloned(),
            )
        })
        .collect();
    visible.sort_by_key(|(id, ..)| *id);
    let cancel_partial = {
        let mut host = world.resource_mut::<Host>();
        for (id, key, stream, outcome, published) in visible {
            let prior = host.seen.get(&id).copied().unwrap_or_default();
            let events = stream.as_ref().map_or(0, |s| s.events.len());
            if events != prior.0 {
                host.observe(
                    "collect.stream",
                    Some(id),
                    json!({"events":events,"text":stream.as_ref().map(|s| &s.text)}),
                );
            }
            if let Some(outcome) = &outcome
                && !prior.1
            {
                host.observe(
                    "collect.outcome",
                    Some(id),
                    json!({"key":key,"outcome":outcome.0}),
                );
                if key.as_str() == tool_key("apply_edit") && outcome.0.is_ok() {
                    match published
                        .as_ref()
                        .map(|output| output.0.result::<WriteReceipt>())
                    {
                        Some(Ok(Some(receipt)))
                            if receipt.operation == "greeting-fix-v1"
                                && receipt.applied == (host.case.approval == Approval::Approve) =>
                        {
                            host.observe("collect.publication", Some(id), json!(receipt));
                        }
                        _ => {
                            host.failure =
                                Some("write receipt missing or incorrect at Collect".into())
                        }
                    }
                }
                if let Ok(Outcome::ToolResult { result }) = &outcome.0 {
                    let parsed = serde_json::from_str::<Value>(&result.output().render());
                    match parsed {
                        Ok(value) => {
                            if host.primary.is_none()
                                && matches!(
                                    key.as_str(),
                                    "maintenance/tool:read_file" | "maintenance/tool:search_file"
                                )
                            {
                                host.primary = Some(key.as_str().into());
                                // Publish the first available inspection as the provisional view.
                                host.observe(
                                    "preview.primary",
                                    Some(id),
                                    json!({"source":key,"preview":value}),
                                );
                            }
                            if let Some(proposal) = value.get("proposed").and_then(Value::as_str) {
                                host.proposal = Some(proposal.into());
                            }
                            if key.as_str() == tool_key("apply_edit") {
                                // Replay applies recorded data only inside its own disposable
                                // projection workspace. It never invokes the external tool.
                                if host.replay {
                                    let decision = host.case.approval;
                                    if value.get("applied") == Some(&Value::Bool(true)) {
                                        if decision != Approval::Approve {
                                            host.failure =
                                                Some("replayed write without approval".into());
                                        } else if let Some(content) =
                                            value.get("content").and_then(Value::as_str)
                                            && let Err(error) = host.workspace.apply(content)
                                        {
                                            host.failure = Some(error.to_string());
                                        }
                                    }
                                }
                            }
                            if value.get("valid") == Some(&Value::Bool(true)) {
                                host.validated = true;
                            }
                        }
                        Err(error) => {
                            host.failure = Some(format!("tool output is not JSON: {error}"))
                        }
                    }
                }
            }
            host.seen.insert(id, (events, outcome.is_some()));
        }
        host.case.fault == Fault::CancelPartial
    };
    if world.resource::<Host>().case.fault == Fault::CancelBackground
        && world.resource::<Host>().primary.is_some()
    {
        let losers: Vec<_> = world.query_filtered::<(Entity, &Issued, &PendingEffect), With<InFlight>>().iter(world)
            .filter(|(_, _, effect)| matches!(&effect.kind, EffectKind::Completion { request, .. } if request.tools.is_empty()))
            .map(|(entity, id, _)| (entity, id.0.as_u64())).collect();
        for (entity, id) in losers {
            world.resource_mut::<Host>().observe(
                "policy.cancel-loser",
                Some(id),
                json!({"reason":"inspection is available"}),
            );
            world.despawn(entity);
        }
    }
    if cancel_partial {
        let partial: Vec<_> = world
            .query_filtered::<(Entity, &bus::Streamed), With<InFlight>>()
            .iter(world)
            .filter(|(_, stream)| !stream.text.is_empty())
            .map(|(entity, _)| entity)
            .collect();
        if !partial.is_empty() {
            cancel_consumer(world, "partial-progress");
            for entity in partial {
                world.despawn(entity);
            }
        }
    }
}

async fn drive(mut app: App, run: Entity) -> Result<Evidence, Error> {
    let result = drive_world(&mut app, run).await;
    if let Err(error) = &result {
        let host = app.world().resource::<Host>();
        let bundle = artifacts::candidate(&host.case)?;
        let effects = app.world().resource::<EffectLogResource>().log();
        let observations = host.observations.clone();
        let writes = host.workspace.writes;
        let pending: Vec<_> = app
            .world_mut()
            .query::<(&Issued, &PendingEffect, Option<&EffectOutcome>)>()
            .iter(app.world())
            .filter(|(_, _, outcome)| outcome.is_none())
            .map(|(issued, pending, _)| json!({"effect":issued.0,"pending":pending}))
            .collect();
        artifacts::write(
            &bundle.join("runtime-failure.json"),
            &crate::cassettes::scrub_artifact(&json!({
                "error":error.to_string(),"boundary":observations.last().map(|item|&item.boundary),
                "observations":observations,"effects":effects,"pending":pending,"writes":writes
            })),
        )?;
        return Err(Error::Invariant(format!(
            "{error}; runtime evidence {}",
            bundle.display()
        )));
    }
    result
}

async fn drive_world(app: &mut App, run: Entity) -> Result<Evidence, Error> {
    let deadline = Instant::now() + Duration::from_secs(120);
    let control = app.world().resource::<scheduled::DeliveryControl>().clone();
    loop {
        control.release();
        if control.ready(app.world_mut()) {
            app.update();
            control.collected();
        }
        app.world().resource::<ExecutionFailures>().check()?;
        if let Some(failure) = app.world().get_resource::<bus::ReplayFailure>() {
            return Err(Error::Runtime(failure.0.clone()));
        }
        if let Some(Failed(failure)) = app.world().get::<Failed>(run) {
            let expected_cancel = matches!(failure, Failure::Cancelled(_))
                && (app.world().resource::<Host>().case.approval == Approval::Cancel
                    || matches!(
                        app.world().resource::<Host>().case.fault,
                        Fault::CancelBeforeServe | Fault::CancelPartial
                    ));
            let expected_stream_error = app.world().resource::<Host>().case.fault
                == Fault::StreamErrorBeforeFinal
                && matches!(failure, Failure::Provider(report) if report.kind == ErrorKind::Provider && report.message == "controlled stream error");
            let expected_missing_model = matches!(
                app.world().resource::<Host>().case.fault,
                Fault::RemoveModelBeforeDispatch | Fault::RemoveModelBetweenTurns
            ) && matches!(failure, Failure::Provider(report) if report.kind == ErrorKind::HandlerUnavailable);
            if !expected_cancel && !expected_stream_error && !expected_missing_model {
                return Err(Error::Invariant(format!("run failed: {failure:?}")));
            }
            // Keep draining an already issued stream so its record retains
            // every item, including data after the first terminal signal.
            if app
                .world_mut()
                .query::<&InFlight>()
                .iter(app.world())
                .next()
                .is_none()
            {
                break;
            }
        }
        if app.world().get::<Settled>(run).is_some() {
            break;
        }
        if Instant::now() >= deadline {
            return Err(Error::Invariant("120s consumer deadline exceeded".into()));
        }
        tokio::task::yield_now().await;
    }
    custom::validate(app.world_mut())?;
    let terminal = match app.world().get::<Failed>(run) {
        Some(Failed(failure)) => json!({"failed":failure}),
        None => json!({"settled":true}),
    };
    app.world_mut()
        .resource_mut::<Host>()
        .observe("run.terminal", None, terminal);
    let host = app.world().resource::<Host>();
    if let Some(error) = &host.failure {
        return Err(Error::Invariant(error.clone()));
    }
    let contents = host.workspace.read()?;
    if matches!(
        host.case.fault,
        Fault::StreamErrorBeforeFinal | Fault::StreamErrorAfterFinal
    ) {
        let log = app.world().resource::<EffectLogResource>().log();
        let mut checked = 0;
        for record in &log.records {
            let EffectKind::Completion { stream: true, .. } = &record.kind else {
                continue;
            };
            let final_index = record
                .events
                .as_deref()
                .unwrap_or_default()
                .iter()
                .position(|event| matches!(event, rig_core::streaming::StreamEvent::Final(_)))
                .ok_or_else(|| Error::Invariant("faulted stream lost its Final event".into()))?;
            let errors = log
                .header
                .stream_errors
                .get(&record.id)
                .ok_or_else(|| Error::Invariant("faulted stream lost its error item".into()))?;
            let before = host.case.fault == Fault::StreamErrorBeforeFinal;
            if errors.len() != 1
                || !errors.iter().all(|error| {
                    error.error.message == "controlled stream error"
                        && (error.item <= final_index) == before
                })
                || record.outcome.is_err() != before
            {
                return Err(Error::Invariant(
                    "stream error position or first-terminal outcome changed".into(),
                ));
            }
            checked += 1;
        }
        if checked == 0 {
            return Err(Error::Invariant(
                "stream fault did not exercise a completion".into(),
            ));
        }
    }
    if host.case.fault == Fault::CancelBackground
        && !host
            .observations
            .iter()
            .any(|item| item.boundary == "policy.cancel-loser")
    {
        return Err(Error::Invariant(
            "consumer did not cancel its unfinished background loser".into(),
        ));
    }
    if host.case.interleaved {
        let log = app.world().resource::<EffectLogResource>().log();
        let mut batches = BTreeMap::<u64, std::collections::BTreeSet<u64>>::new();
        for delivery in log.header.deliveries.as_deref().unwrap_or_default() {
            if matches!(delivery.kind, rig_core::effect::DeliveryKind::Stream { .. }) {
                batches
                    .entry(delivery.batch)
                    .or_default()
                    .insert(delivery.id.as_u64());
            }
        }
        // Resume's tail starts after both initial streams have completed.
        let restoring = !app
            .world()
            .resource::<persistence::Checkpoints>()
            .0
            .is_empty()
            && log
                .records
                .first()
                .is_some_and(|record| record.id.as_u64() > 1);
        if !restoring && batches.values().filter(|ids| ids.len() > 1).count() < 2 {
            return Err(Error::Invariant(
                "interleaved case requires repeated Collect batches containing both streams".into(),
            ));
        }
    }
    if host.case.provider == Provider::Synthetic
        && !matches!(
            host.case.fault,
            Fault::StreamErrorBeforeFinal
                | Fault::RemoveModelBeforeDispatch
                | Fault::CancelBeforeServe
                | Fault::CancelPartial
        )
    {
        let expected = match host.case.arrival {
            Arrival::SearchFirst => "maintenance/tool:search_file",
            _ => "maintenance/tool:read_file",
        };
        if host.primary.as_deref() != Some(expected) {
            return Err(Error::Invariant(format!(
                "first visible inspection must be {expected}"
            )));
        }
    }
    let failed_before_write = matches!(
        host.case.fault,
        Fault::WriteError
            | Fault::StreamErrorBeforeFinal
            | Fault::RemoveModelBeforeDispatch
            | Fault::RemoveModelBetweenTurns
            | Fault::CancelBeforeServe
            | Fault::CancelPartial
    );
    if failed_before_write && (contents != INITIAL || host.workspace.writes != 0 || host.validated)
    {
        return Err(Error::Invariant(
            "failed operation changed or validated the workspace".into(),
        ));
    }
    if host.case.fault == Fault::WriteError
        && !host.observations.iter().any(|observation| {
            observation
                .data
                .pointer("/outcome/Err/message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("controlled write failure"))
        })
    {
        return Err(Error::Invariant(
            "write failure scenario did not observe its injected error".into(),
        ));
    }
    match host.case.approval {
        Approval::Approve
            if !failed_before_write
                && (contents != TARGET || !host.validated || host.workspace.writes != 1) =>
        {
            return Err(Error::Invariant(format!(
                "approved edit must change and validate file exactly once: content={contents:?}, validated={}, writes={}",
                host.validated, host.workspace.writes
            )));
        }
        Approval::Deny | Approval::Cancel if contents != INITIAL || host.workspace.writes != 0 => {
            return Err(Error::Invariant("unapproved file changed".into()));
        }
        _ => (),
    }
    let result = app
        .world()
        .get::<RunResult>(run)
        .map(|r| r.0.clone())
        .unwrap_or_default();
    Ok(Evidence {
        effects: app.world().resource::<EffectLogResource>().log(),
        observations: host.observations.clone(),
        files: [("greeting.txt".into(), contents)].into(),
        writes: host.workspace.writes,
        result,
        checkpoints: app.world().resource::<persistence::Checkpoints>().0.clone(),
    })
}

pub(crate) async fn execute(case: &Case, model: impl Serve + 'static) -> Result<Evidence, Error> {
    let mut app = build(case, None)?;
    let model = TokioHandler {
        handler: Arc::new(scheduled::Scheduled {
            handler: Arc::new(model),
            control: app.world().resource::<scheduled::DeliveryControl>().clone(),
            batch_size: case.stream_batch,
            fault: case.fault,
            failures: app.world().resource::<ExecutionFailures>().clone(),
        }),
        runtime: tokio::runtime::Handle::current(),
        failures: app.world().resource::<ExecutionFailures>().clone(),
    };
    Handlers::with(app.world_mut(), |h| h.register(MODEL, model))??;
    let run = program(&mut app, case)?;
    drive(app, run).await
}

pub(crate) async fn replay(case: &Case, log: &EffectLog) -> Result<Evidence, Error> {
    let mut app = build(case, Some(log))?;
    let run = program(&mut app, case)?;
    rig_ecs::replay::check_replayable(app.world_mut(), run, log)?;
    drive(app, run).await
}

/// Synthetic provider-independent policy stimulus, explicitly not a cassette.
pub(crate) struct Scripted;
impl Serve for Scripted {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(MODEL),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new("synthetic/maintenance"),
                capabilities: ProviderCapabilities::default(),
            },
            layers: Vec::new(),
        }
    }
    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        let EffectKind::Completion { request, stream } = kind else {
            return;
        };
        let results: Vec<_> = request
            .chat_history
            .iter()
            .flat_map(|m| match m {
                Message::User { content } => content
                    .iter()
                    .filter(|c| matches!(c, UserContent::ToolResult(_)))
                    .collect(),
                _ => Vec::new(),
            })
            .collect();
        let call = |id, name, args| AssistantContent::tool_call(id, name, args);
        let choice = if request.tools.is_empty() {
            vec![AssistantContent::text(
                "The workspace tools inspect, propose, approve, edit and validate.",
            )]
        } else {
            match results.len() {
                0 => vec![
                    call("read", "read_file", json!({})),
                    call("search", "search_file", json!({"needle":"Helo"})),
                ],
                2 => vec![call("proposal", "propose_edit", json!({"content":TARGET}))],
                3 => vec![call("write", "apply_edit", json!({}))],
                4 => vec![call("validation", "validate_file", json!({}))],
                _ => vec![AssistantContent::text(
                    "Maintenance finished; inspect the tool results.",
                )],
            }
        };
        if stream {
            let mut writer = sink.writer();
            if let Some(count) = request
                .additional_params
                .as_ref()
                .and_then(|params| params.get("synthetic_background_chunks"))
                .and_then(Value::as_u64)
            {
                for _ in 0..count.min(64) {
                    if writer
                        .text("Checking the maintenance task. ")
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            }
            if results.len() < 5 && writer.text("Inspecting the project. ").await.is_err() {
                return;
            }
            for part in choice {
                let sent = match part {
                    AssistantContent::Text(text) => writer.text(text.text).await,
                    AssistantContent::ToolCall(call) => {
                        writer
                            .tool_call(call.function.name, call.function.arguments)
                            .await
                    }
                    AssistantContent::Reasoning(_) | AssistantContent::Image(_) => return,
                };
                if sent.is_err() {
                    return;
                }
            }
            let _ = writer
                .finish(rig_core::streaming::StreamFinal::new(
                    "synthetic",
                    Usage::new(),
                ))
                .await;
        } else {
            sink.resolve(Ok(Outcome::Completion(CompletionResponse::new(
                choice,
                Usage::new(),
                "synthetic",
            ))))
            .await;
        }
    }
}

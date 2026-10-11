//! Plugin corpus: 54. OpenTelemetry tracing (spec `54-otel-tracing.md`).
//!
//! Each turn is exported as an OpenTelemetry trace: a `turn` span with a
//! `model.request` child for every model call (the turn's own and the
//! [`ModelRequest`]s plugins make for it) and a `tool.call` child for every
//! tool call. A subagent's turns are children of the `task` call that
//! briefed it. Attribute names follow the GenAI semantic conventions where
//! one exists (`gen_ai.*`), and `rig.*` otherwise. Spans are exported over
//! OTLP/HTTP by the SDK's batch processor, on a thread of its own, so a
//! collector that does not answer never holds the agent up; the spans of a
//! failed export are counted and `/telemetry` reports them.
//!
//! Enabled by `OTEL_EXPORTER_OTLP_ENDPOINT` (and
//! `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` for argument
//! and result bodies), or by an [`OtelConfig`] or a [`Tracing`] resource
//! the app inserts before adding the plugin. Without any, it does nothing.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use bevy_log::warn;
use opentelemetry::trace::{SpanKind, Status, TraceContextExt, Tracer as _, TracerProvider as _};
use opentelemetry::{Context, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{SdkTracer, SdkTracerProvider, SpanData, SpanExporter};
use rig_core::completion::Message as Chat;
use rig_ecs::StopTurns;
use rig_ecs::turn::ModelCall;
use rig_harness::prelude::*;

/// The collector's base URL, as OpenTelemetry names it.
pub const ENDPOINT_VAR: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";
/// Whether argument and result bodies go in the spans, as the GenAI
/// conventions name it.
pub const CAPTURE_VAR: &str = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";

/// How long an export, or the last flush on exit, may wait for the
/// collector.
const EXPORT_TIMEOUT: Duration = Duration::from_secs(3);

/// Where spans go and what they carry.
#[derive(Resource, Clone, Debug, Default)]
pub struct OtelConfig {
    /// The collector's base URL, such as `http://localhost:4318`; `None`
    /// turns tracing off.
    pub endpoint: Option<String>,
    /// Whether tool argument and result bodies are attributes.
    pub capture_content: bool,
}

impl OtelConfig {
    /// From [`ENDPOINT_VAR`] and [`CAPTURE_VAR`].
    pub fn from_env() -> Self {
        let endpoint = std::env::var(ENDPOINT_VAR).ok();
        let capture = std::env::var(CAPTURE_VAR).ok();
        Self {
            endpoint: endpoint.filter(|endpoint| !endpoint.trim().is_empty()),
            capture_content: capture.is_some_and(|value| value.eq_ignore_ascii_case("true")),
        }
    }
}

/// Spans exported and dropped so far.
#[derive(Debug, Default)]
pub struct ExportCounts {
    /// Spans the collector took.
    pub exported: AtomicU64,
    /// Spans of exports that failed, such as to a collector that is down.
    pub dropped: AtomicU64,
}

/// The tracer the plugin records with, and its export.
#[derive(Resource, Clone)]
pub struct Tracing {
    provider: SdkTracerProvider,
    tracer: SdkTracer,
    /// Whether tool argument and result bodies are attributes.
    pub capture_content: bool,
    /// Where spans go, for `/telemetry`.
    pub destination: String,
    /// What the export did so far.
    pub counts: Arc<ExportCounts>,
}

impl Tracing {
    /// Spans batched in the background to `exporter`, which `destination`
    /// names.
    pub fn new<E: SpanExporter + 'static>(
        exporter: E,
        destination: impl Into<String>,
        capture_content: bool,
    ) -> Self {
        let counts = Arc::new(ExportCounts::default());
        let exporter = Counted {
            inner: exporter,
            counts: counts.clone(),
        };
        let resource = Resource::builder().with_service_name("rig").build();
        let provider = SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(resource)
            .build();
        Self {
            tracer: provider.tracer("rig"),
            provider,
            capture_content,
            destination: destination.into(),
            counts,
        }
    }

    /// Spans exported over OTLP/HTTP to the collector at `endpoint`.
    pub fn otlp(endpoint: &str, capture_content: bool) -> Result<Self, String> {
        let base = endpoint.trim_end_matches('/');
        let url = match base.ends_with("/v1/traces") {
            true => base.to_owned(),
            false => format!("{base}/v1/traces"),
        };
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_endpoint(url.clone())
            .with_timeout(EXPORT_TIMEOUT)
            .build()
            .map_err(|error| error.to_string())?;
        Ok(Self::new(exporter, url, capture_content))
    }

    /// Exports every ended span now, waiting for the exporter.
    pub fn flush(&self) -> OTelSdkResult {
        self.provider.force_flush()
    }

    /// A span `name` of `kind`, child of `parent`, with `attributes`.
    fn start(
        &self,
        name: &'static str,
        kind: SpanKind,
        parent: &Context,
        attributes: Vec<KeyValue>,
    ) -> Context {
        let span = self
            .tracer
            .span_builder(name)
            .with_kind(kind)
            .with_attributes(attributes)
            .start_with_context(&self.tracer, parent);
        parent.with_span(span)
    }
}

/// An exporter that counts what it exported and what it lost.
#[derive(Debug)]
struct Counted<E> {
    inner: E,
    counts: Arc<ExportCounts>,
}

impl<E: SpanExporter> SpanExporter for Counted<E> {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        let spans = u64::try_from(batch.len()).unwrap_or(u64::MAX);
        let result = self.inner.export(batch).await;
        let counter = match result {
            Ok(()) => &self.counts.exported,
            Err(_) => &self.counts.dropped,
        };
        counter.fetch_add(spans, Ordering::Relaxed);
        result
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.inner.shutdown_with_timeout(timeout)
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}

/// Exports each turn as a trace; see the crate's documentation.
#[derive(Default)]
pub struct OtelTracingPlugin;

impl Plugin for OtelTracingPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<Tracing>() {
            let config = app.world().get_resource::<OtelConfig>().cloned();
            let config = config.unwrap_or_else(OtelConfig::from_env);
            let Some(endpoint) = config.endpoint else {
                return;
            };
            match Tracing::otlp(&endpoint, config.capture_content) {
                Ok(tracing) => app.insert_resource(tracing),
                Err(error) => {
                    warn!("OpenTelemetry tracing is off: {error}");
                    return;
                }
            };
        }
        app.init_resource::<Traces>()
            .add_observer(start_turn)
            .add_observer(note_request)
            .add_observer(end_turn)
            .add_observer(start_model_call)
            .add_observer(start_plugin_request)
            .add_observer(end_model_call)
            .add_observer(start_tool_call)
            .add_observer(end_tool_call)
            .add_observer(end_cancelled)
            .add_observer(link_subagent)
            .add_systems(Update, first_token.after(PollCalls))
            .add_systems(
                Last,
                flush_on_exit.after(StopTurns).run_if(on_message::<AppExit>),
            )
            .add_command(
                "telemetry",
                "Show where OpenTelemetry spans go, and how many were exported or dropped",
                telemetry,
            );
    }
}

/// The open turn spans and what links subagents to the calls that briefed
/// them.
#[derive(Resource, Default)]
struct Traces {
    /// Each busy agent's turn span, and whether its user message was
    /// measured.
    turns: HashMap<Entity, (Context, bool)>,
    /// Each tool call's span by the calling agent's id and the call's id,
    /// while the caller's turn runs: a message an agent sends for a call
    /// names that call as its request.
    tool_calls: HashMap<(String, String), Context>,
    /// The span a subagent's next turn goes under: the call whose message
    /// started it.
    parents: HashMap<Entity, Context>,
}

/// A call's span; removing it, as despawning a stopped turn's calls does,
/// ends the span as cancelled if it is still open.
#[derive(Component)]
struct Traced {
    context: Context,
    started: SystemTime,
}

/// On a model call whose first streamed text was timed.
#[derive(Component)]
struct FirstToken;

fn text_len(attribute: &'static str, len: usize) -> KeyValue {
    KeyValue::new(attribute, i64::try_from(len).unwrap_or(i64::MAX))
}

fn count(attribute: &'static str, value: u64) -> KeyValue {
    KeyValue::new(attribute, i64::try_from(value).unwrap_or(i64::MAX))
}

/// Opens a turn's span, under the call that started a subagent's work.
fn start_turn(
    add: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    agents: Query<(&AgentId, Option<&Name>, Has<SpawnedBy>)>,
    session: Option<Res<SessionPaths>>,
    tracing: Res<Tracing>,
    mut traces: ResMut<Traces>,
) {
    let Ok(&TurnOf(agent)) = turns.get(add.entity) else {
        return;
    };
    let Ok((id, name, spawned)) = agents.get(agent) else {
        return;
    };
    let mut attributes = vec![
        KeyValue::new("gen_ai.operation.name", "invoke_agent"),
        KeyValue::new("gen_ai.agent.id", id.0.clone()),
        KeyValue::new("rig.agent.subagent", spawned),
    ];
    if let Some(name) = name {
        attributes.push(KeyValue::new("gen_ai.agent.name", name.as_str().to_owned()));
    }
    let session = session.and_then(|paths| Some(paths.path().file_name()?.to_str()?.to_owned()));
    if let Some(session) = session {
        attributes.push(KeyValue::new("gen_ai.conversation.id", session));
    }
    let parent = traces.parents.remove(&agent).unwrap_or_default();
    let context = tracing.start("turn", SpanKind::Internal, &parent, attributes);
    if let Some((stale, _)) = traces.turns.insert(agent, (context, false)) {
        stale.span().set_status(Status::error("superseded"));
        stale.span().end();
    }
}

/// Measures the user's message the turn's first request answers.
fn note_request(prepare: On<PrepareRequest>, mut traces: ResMut<Traces>) {
    let Some((context, measured)) = traces.turns.get_mut(&prepare.agent) else {
        return;
    };
    if *measured {
        return;
    }
    *measured = true;
    if let Some(Chat::User { content }) = prepare.messages.last() {
        let len: usize = content
            .iter()
            .filter_map(|item| match item {
                UserContent::Text(text) => Some(text.text.len()),
                _ => None,
            })
            .sum();
        context
            .span()
            .set_attribute(text_len("rig.user_message.length", len));
    }
}

/// Ends a turn's span with how the turn ended.
fn end_turn(ended: On<TurnEnded>, ids: Query<&AgentId>, mut traces: ResMut<Traces>) {
    let agent = ended.entity;
    if let Ok(id) = ids.get(agent) {
        traces.tool_calls.retain(|(caller, _), _| *caller != id.0);
    }
    let Some((context, _)) = traces.turns.remove(&agent) else {
        return;
    };
    let span = context.span();
    let (outcome, status) = match &ended.outcome {
        TurnOutcome::Answered(_) => ("answered", Status::Ok),
        TurnOutcome::Failed(why) => ("failed", Status::error(why.clone())),
        TurnOutcome::Stopped => ("interrupted", Status::error("interrupted")),
    };
    span.set_attribute(KeyValue::new("rig.turn.outcome", outcome));
    span.set_status(status);
    span.end();
}

/// The model a call of `turn` goes to: its own connection, else its
/// turn's, else its agent's.
fn model_of<'a>(
    own: Option<&'a Connection>,
    routed: Option<&'a Connection>,
    agent: Option<&'a Connection>,
) -> Option<&'a Connection> {
    own.or(routed).or(agent)
}

/// Opens the span of a model call, child of its turn's.
fn start_model(
    world: (&Res<Tracing>, &Traces),
    (call, agent, turn_retries): (Entity, Entity, u32),
    connection: Option<&Connection>,
    made_by: Option<&Name>,
    commands: &mut Commands,
) {
    let (tracing, traces) = world;
    let Some((parent, _)) = traces.turns.get(&agent) else {
        return;
    };
    let mut attributes = vec![
        KeyValue::new("gen_ai.operation.name", "chat"),
        count("rig.retries", u64::from(turn_retries)),
    ];
    if let Some(connection) = connection {
        let spec = &connection.spec;
        attributes.push(KeyValue::new("gen_ai.request.model", spec.id.clone()));
        attributes.push(KeyValue::new(
            "gen_ai.provider.name",
            spec.provider.vendor().to_string(),
        ));
    }
    if let Some(name) = made_by {
        attributes.push(KeyValue::new(
            "rig.plugin_request",
            name.as_str().to_owned(),
        ));
    }
    let context = tracing.start("model.request", SpanKind::Client, parent, attributes);
    commands.entity(call).insert(Traced {
        context,
        started: SystemTime::now(),
    });
}

/// A turn's own model call.
fn start_model_call(
    add: On<Add<ModelCall>>,
    calls: Query<&CallOf>,
    turns: Query<(&TurnOf, &Recovery, Option<&Connection>)>,
    agents: Query<Option<&Connection>>,
    tracing: Res<Tracing>,
    traces: Res<Traces>,
    mut commands: Commands,
) {
    let Ok(&CallOf(turn)) = calls.get(add.entity) else {
        return;
    };
    let Ok((&TurnOf(agent), recovery, routed)) = turns.get(turn) else {
        return;
    };
    let own = agents.get(agent).ok().flatten();
    start_model(
        (&tracing, &traces),
        (add.entity, agent, recovery.retries),
        model_of(None, routed, own),
        None,
        &mut commands,
    );
}

/// A plugin's model call for a turn, such as a compaction's summary.
fn start_plugin_request(
    add: On<Add<ModelRequest>>,
    calls: Query<(&CallOf, Option<&Connection>, Option<&Name>)>,
    turns: Query<(&TurnOf, &Recovery, Option<&Connection>)>,
    agents: Query<Option<&Connection>>,
    tracing: Res<Tracing>,
    traces: Res<Traces>,
    mut commands: Commands,
) {
    let Ok((&CallOf(turn), own, name)) = calls.get(add.entity) else {
        return;
    };
    let Ok((&TurnOf(agent), recovery, routed)) = turns.get(turn) else {
        return;
    };
    let agent_model = agents.get(agent).ok().flatten();
    // Who made the call is not on it: its `Name`, when the plugin gave one.
    let name = name.cloned().unwrap_or_else(|| Name::new("plugin"));
    start_model(
        (&tracing, &traces),
        (add.entity, agent, recovery.retries),
        model_of(own, routed, agent_model),
        Some(&name),
        &mut commands,
    );
}

/// Ends a model call's span with its usage, cost and finish, or its error.
fn end_model_call(done: On<Add<Done<ModelReply>>>, calls: Query<(&Traced, &Done<ModelReply>)>) {
    let Ok((traced, Done(reply))) = calls.get(done.entity) else {
        return;
    };
    let span = traced.context.span();
    match reply {
        Ok(response) => {
            let usage = &response.usage;
            let tokens = [
                ("gen_ai.usage.input_tokens", usage.input_tokens),
                ("gen_ai.usage.output_tokens", usage.output_tokens),
                (
                    "gen_ai.usage.cache_read.input_tokens",
                    usage.cached_input_tokens,
                ),
                (
                    "gen_ai.usage.cache_creation.input_tokens",
                    usage.cache_creation_input_tokens,
                ),
                ("gen_ai.usage.reasoning_tokens", usage.reasoning_tokens),
            ];
            for (attribute, value) in tokens {
                if let Some(value) = value {
                    span.set_attribute(count(attribute, value));
                }
            }
            if let Some(cost) = &usage.cost {
                span.set_attribute(KeyValue::new("rig.cost.usd", cost.total));
            }
            if let Some(model) = response.model() {
                span.set_attribute(KeyValue::new("gen_ai.response.model", model.to_owned()));
            }
            if let Some(id) = response.response_id() {
                span.set_attribute(KeyValue::new("gen_ai.response.id", id.to_owned()));
            }
            if let Some(reason) = response.finish_reason() {
                let reason = format!("{reason:?}").to_lowercase();
                span.set_attribute(KeyValue::new("gen_ai.response.finish_reasons", reason));
            }
            match &response.error {
                Some(error) => {
                    span.set_attribute(KeyValue::new("error.type", "provider_refusal"));
                    span.set_status(Status::error(error.clone()));
                }
                None => span.set_status(Status::Ok),
            }
        }
        Err(report) => {
            span.set_attribute(KeyValue::new("error.type", report.kind.code()));
            span.set_status(Status::error(report.message.clone()));
        }
    }
    span.end();
}

/// Times a model call's first streamed text or reasoning, when it has any.
fn first_token(
    calls: Query<(Entity, &Partial, &Traced), (Changed<Partial>, Without<FirstToken>)>,
    mut commands: Commands,
) {
    for (call, partial, traced) in &calls {
        if partial.text.is_empty() && partial.reasoning.is_empty() {
            continue;
        }
        let waited = traced.started.elapsed().unwrap_or_default();
        let ms = u64::try_from(waited.as_millis()).unwrap_or(u64::MAX);
        let span = traced.context.span();
        span.set_attribute(count("rig.time_to_first_token_ms", ms));
        commands.entity(call).insert(FirstToken);
    }
}

/// Opens a tool call's span, child of its turn's.
fn start_tool_call(
    add: On<Add<ToolCallRun>>,
    calls: Query<(&ToolCallRun, &CallOf)>,
    turns: Query<&TurnOf>,
    ids: Query<&AgentId>,
    tracing: Res<Tracing>,
    mut traces: ResMut<Traces>,
    mut commands: Commands,
) {
    let Ok((run, &CallOf(turn))) = calls.get(add.entity) else {
        return;
    };
    let Ok(&TurnOf(agent)) = turns.get(turn) else {
        return;
    };
    let Some((parent, _)) = traces.turns.get(&agent) else {
        return;
    };
    let call = &run.call;
    let arguments = call.function.raw_arguments();
    let mut attributes = vec![
        KeyValue::new("gen_ai.operation.name", "execute_tool"),
        KeyValue::new("gen_ai.tool.name", call.function.name.as_str().to_owned()),
        KeyValue::new("gen_ai.tool.call.id", call.id.to_string()),
        text_len("rig.tool.arguments.size", arguments.len()),
    ];
    if tracing.capture_content {
        attributes.push(KeyValue::new("gen_ai.tool.call.arguments", arguments));
    }
    let context = tracing.start("tool.call", SpanKind::Internal, parent, attributes);
    if let Ok(id) = ids.get(agent) {
        let key = (id.0.clone(), call.id.to_string());
        traces.tool_calls.insert(key, context.clone());
    }
    commands.entity(add.entity).insert(Traced {
        context,
        started: SystemTime::now(),
    });
}

/// Ends a tool call's span with its result: an error status for a failure,
/// and `denied` for a call refused before it ran.
fn end_tool_call(
    done: On<Add<ToolOutput>>,
    calls: Query<(&Traced, &ToolOutput)>,
    tracing: Res<Tracing>,
) {
    let Ok((traced, ToolOutput(result))) = calls.get(done.entity) else {
        return;
    };
    let span = traced.context.span();
    let output = result.output().render();
    span.set_attribute(text_len("rig.tool.result.size", output.len()));
    span.set_attribute(KeyValue::new("rig.tool.status", result.status_name()));
    span.set_attribute(KeyValue::new("rig.tool.is_error", !result.is_success()));
    if tracing.capture_content {
        span.set_attribute(KeyValue::new("gen_ai.tool.call.result", output));
    }
    let status = match (result.refusal(), result.error()) {
        (Some(_), _) => Status::error("denied"),
        (_, Some(error)) => {
            span.set_attribute(KeyValue::new("error.type", error.kind().as_str()));
            Status::error(error.message().to_owned())
        }
        (None, None) => Status::Ok,
    };
    span.set_status(status);
    span.end();
}

/// Ends the span of a call that went away unfinished, such as one of an
/// interrupted turn.
fn end_cancelled(remove: On<Remove<Traced>>, calls: Query<&Traced>) {
    let Ok(traced) = calls.get(remove.entity) else {
        return;
    };
    let span = traced.context.span();
    if span.is_recording() {
        span.set_status(Status::error("cancelled"));
        span.end();
    }
}

/// A message an agent sends for one of its tool calls, such as a `task`
/// call's brief to the subagent it started, puts the receiver's next turn
/// under that call.
fn link_subagent(sent: On<Deliver>, mut traces: ResMut<Traces>) {
    let origin = &sent.origin;
    let (Some(from), Some(request)) = (&origin.from, &origin.request) else {
        return;
    };
    let key = (from.0.clone(), request.0.clone());
    if let Some(call) = traces.tool_calls.get(&key).cloned() {
        traces.parents.insert(sent.entity, call);
    }
}

/// On exit, ends the turns the exit stopped, which end without a
/// [`TurnEnded`], and sends what is batched.
fn flush_on_exit(tracing: Res<Tracing>, mut traces: ResMut<Traces>) {
    for (_, (context, _)) in traces.turns.drain() {
        let span = context.span();
        span.set_attribute(KeyValue::new("rig.turn.outcome", "exit"));
        span.set_status(Status::error("the app exited"));
        span.end();
    }
    if let Err(error) = tracing.provider.shutdown_with_timeout(EXPORT_TIMEOUT) {
        warn!("OpenTelemetry: the last spans were not sent: {error}");
    }
}

/// `/telemetry`: where spans go, and how many were exported or dropped.
fn telemetry(
    In(args): In<CommandArgs>,
    tracing: Res<Tracing>,
    traces: Res<Traces>,
    mut notices: MessageWriter<Notice>,
) {
    let counts = &tracing.counts;
    let text = format!(
        "OpenTelemetry tracing to {}: {} spans exported, {} dropped, {} turns open. Content \
         capture is {}.",
        tracing.destination,
        counts.exported.load(Ordering::Relaxed),
        counts.dropped.load(Ordering::Relaxed),
        traces.turns.len(),
        if tracing.capture_content { "on" } else { "off" },
    );
    notices.write(Notice::info(args.agent, text));
}

#[cfg(test)]
mod tests;

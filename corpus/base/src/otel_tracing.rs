//! Plugin corpus: 54. OpenTelemetry tracing (spec `54-otel-tracing.md`).
//!
//! Each turn is exported as an OpenTelemetry trace: a `turn` span with a
//! `model.request` child for every model call of the turn (its own and
//! those plugins make for it) and a `tool.call` child for every tool call.
//! A model call of no turn is a root span named after its purpose, such as
//! `compaction`. An agent whose work is recorded under an effect
//! ([`EffectParent`]), such as the subagent of a `task` call, has its turns
//! under that effect's span. Spans live on the entities they trace: a call
//! or turn that goes away unfinished ends its span as such.
//!
//! Attribute names follow the GenAI semantic conventions where one exists
//! (`gen_ai.*`), and `rig.*` otherwise. Spans are exported over OTLP/HTTP by
//! the SDK's batch processor, on a thread of its own, so a collector that
//! does not answer never holds the agent up; the spans of a failed export
//! are counted and `/telemetry` reports them.
//!
//! Enabled by the `[otel]` settings table (`endpoint`, `capture_content`)
//! or by `OTEL_EXPORTER_OTLP_ENDPOINT` (and
//! `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`), which win, or
//! by a [`Tracing`] resource the app inserts before adding the plugin.
//! Without any, it does nothing.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::world::DeferredWorld;
use bevy_log::warn;
use opentelemetry::trace::{SpanKind, Status, TraceContextExt, Tracer as _, TracerProvider as _};
use opentelemetry::{Context, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{SdkTracer, SdkTracerProvider, SpanData, SpanExporter};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionResponse;
use rig_harness::rig_core::effect::EffectId;
use rig_harness::rig_ecs::StopTurnsOnExit;
use serde::Deserialize;

/// The collector's base URL, as OpenTelemetry names it.
pub const ENDPOINT_VAR: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";
/// Whether argument and result bodies go in the spans, as the GenAI
/// conventions name it.
pub const CAPTURE_VAR: &str = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";

/// How long an export, or the last flush on exit, may wait for the
/// collector.
const EXPORT_TIMEOUT: Duration = Duration::from_secs(3);

/// The `[otel]` settings table.
#[derive(Resource, Deserialize, Clone, Debug, Default)]
#[serde(default, deny_unknown_fields)]
pub struct OtelSettings {
    /// The collector's base URL, such as `http://localhost:4318`; none
    /// turns tracing off.
    pub endpoint: Option<String>,
    /// Whether tool argument and result bodies are attributes.
    pub capture_content: bool,
}

impl OtelSettings {
    /// These settings, with [`ENDPOINT_VAR`] and [`CAPTURE_VAR`] in place
    /// of what they set.
    fn with_env(mut self) -> Self {
        if let Ok(endpoint) = std::env::var(ENDPOINT_VAR) {
            self.endpoint = Some(endpoint);
        }
        if let Ok(capture) = std::env::var(CAPTURE_VAR) {
            self.capture_content = capture.eq_ignore_ascii_case("true");
        }
        self.endpoint = self.endpoint.filter(|endpoint| !endpoint.trim().is_empty());
        self
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

    /// A span `name` of `kind`, under `parent` (a root span without one).
    fn start(
        &self,
        name: impl Into<std::borrow::Cow<'static, str>>,
        kind: SpanKind,
        parent: Option<&Context>,
        attributes: Vec<KeyValue>,
    ) -> Span {
        let parent = parent.cloned().unwrap_or_else(Context::new);
        let span = self
            .tracer
            .span_builder(name)
            .with_kind(kind)
            .with_attributes(attributes)
            .start_with_context(&self.tracer, &parent);
        Span(parent.with_span(span))
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
            app.add_setting::<OtelSettings>("otel");
            let settings = app.world().resource::<OtelSettings>().clone().with_env();
            let Some(endpoint) = settings.endpoint else {
                return;
            };
            match Tracing::otlp(&endpoint, settings.capture_content) {
                Ok(tracing) => app.insert_resource(tracing),
                Err(error) => {
                    warn!("OpenTelemetry tracing is off: {error}");
                    return;
                }
            };
        }
        app.add_observer(nest_agent)
            .add_observer(start_turn)
            .add_observer(end_turn)
            .add_observer(start_call)
            .add_observer(end_model_call)
            .add_observer(end_tool_call)
            .add_systems(
                Last,
                flush_on_exit
                    .after(StopTurnsOnExit)
                    .run_if(on_message::<AppExit>),
            )
            .add_command(
                "telemetry",
                "Show where OpenTelemetry spans go, and how many were exported or dropped",
                telemetry,
            );
    }
}

/// The span of the turn or call it is on. Removed while the span is still
/// open, as despawning an unfinished call or a turn stopped on exit does,
/// it ends it as unfinished.
#[derive(Component)]
#[component(on_remove = end_unfinished)]
struct Span(Context);

/// On an agent: the span its turns go under, that of the effect its work is
/// recorded under ([`EffectParent`]).
#[derive(Component)]
struct ParentSpan(Context);

fn end_unfinished(world: DeferredWorld, context: HookContext) {
    if let Some(Span(traced)) = world.get::<Span>(context.entity) {
        let span = traced.span();
        if span.is_recording() {
            span.set_status(Status::error("unfinished"));
            span.end();
        }
    }
}

fn size(attribute: &'static str, len: usize) -> KeyValue {
    KeyValue::new(attribute, i64::try_from(len).unwrap_or(i64::MAX))
}

fn count(attribute: &'static str, value: u64) -> KeyValue {
    KeyValue::new(attribute, i64::try_from(value).unwrap_or(i64::MAX))
}

/// The span of effect `id`, or of the nearest effect it was asked for by
/// that has one, among the effects in flight.
fn span_of_effect(id: EffectId, effects: &Query<(&Effect, Option<&Span>)>) -> Option<Context> {
    let mut next = Some(id);
    // Effect parents form a tree; the bound only guards against a cycle.
    for _ in 0..32 {
        let (effect, span) = effects
            .iter()
            .find(|(effect, _)| Some(effect.id()) == next)?;
        if let Some(Span(context)) = span {
            return Some(context.clone());
        }
        next = effect.parent;
    }
    None
}

/// Puts an agent's turns under the span of the effect its work is recorded
/// under, such as the `task` call that started a subagent.
fn nest_agent(
    insert: On<Insert<EffectParent>>,
    parents: Query<&EffectParent>,
    effects: Query<(&Effect, Option<&Span>)>,
    mut commands: Commands,
) {
    let agent = insert.entity;
    let Ok(&EffectParent(id)) = parents.get(agent) else {
        return;
    };
    match span_of_effect(id, &effects) {
        Some(context) => commands.entity(agent).insert(ParentSpan(context)),
        None => commands.entity(agent).remove::<ParentSpan>(),
    };
}

/// Opens a turn's span.
fn start_turn(
    add: On<Add<TurnOf>>,
    turns: Query<(&TurnOf, &Turn, &Name)>,
    agents: Query<(&AgentId, Option<&Name>, Has<SpawnedBy>, Option<&ParentSpan>)>,
    session: Option<Res<SessionPaths>>,
    tracing: Res<Tracing>,
    mut commands: Commands,
) {
    let Ok((&TurnOf(agent), turn, turn_name)) = turns.get(add.entity) else {
        return;
    };
    let Ok((id, name, spawned, parent)) = agents.get(agent) else {
        return;
    };
    let mut attributes = vec![
        KeyValue::new("gen_ai.operation.name", "invoke_agent"),
        KeyValue::new("gen_ai.agent.id", id.0.clone()),
        KeyValue::new("rig.agent.subagent", spawned),
        KeyValue::new("rig.turn.name", turn_name.as_str().to_owned()),
        count("rig.turn.index", turn.index),
    ];
    if let Some(name) = name {
        attributes.push(KeyValue::new("gen_ai.agent.name", name.as_str().to_owned()));
    }
    if let Some(session) = session.and_then(|paths| paths.id()) {
        attributes.push(KeyValue::new("gen_ai.conversation.id", session.to_string()));
    }
    let parent = parent.map(|ParentSpan(context)| context);
    let span = tracing.start("turn", SpanKind::Internal, parent, attributes);
    commands.entity(add.entity).insert(span);
}

/// Ends a turn's span with how the turn ended, and the length of the
/// message that asked for it.
fn end_turn(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &Turn, &ForAgent, &Span)>,
    conversations: Query<&Conversation>,
) {
    let Ok((outcome, turn, &ForAgent(agent), Span(context))) = turns.get(settled.entity) else {
        return;
    };
    let span = context.span();
    let asked = conversations
        .get(agent)
        .ok()
        .and_then(|conversation| conversation.request(turn.start).map(|entry| entry.message));
    if let Some(message::Message::User { content }) = asked {
        let len: usize = content
            .iter()
            .filter_map(|item| match item {
                UserContent::Text(text) => Some(text.text.len()),
                _ => None,
            })
            .sum();
        span.set_attribute(size("rig.user_message.length", len));
    }
    let (how, status) = match outcome {
        TurnOutcome::Answered(_) => ("answered", Status::Ok),
        TurnOutcome::Failed(why) => ("failed", Status::error(why.clone())),
        TurnOutcome::Stopped(stop) => (
            "interrupted",
            Status::error(match stop.cause() {
                StopCause::User => "interrupted".to_owned(),
                cause => cause.to_string(),
            }),
        ),
    };
    span.set_attribute(KeyValue::new("rig.turn.outcome", how));
    span.set_status(status);
    span.end();
}

/// A call as it is admitted: a model call or a tool call.
type Calls<'w, 's> = Query<
    'w,
    's,
    (&'static Effect, Option<&'static CallOf>, Has<ToolRequest>),
    Or<(
        With<ModelRequest>,
        With<TurnCall>,
        With<SideCall>,
        With<ToolRequest>,
    )>,
>;

/// Opens a call's span as it is admitted, so calls refused before they
/// start have one: under its turn's span, or a root span named after its
/// purpose.
fn start_call(
    admitted: On<Add<Admitted>>,
    calls: Calls,
    turns: Query<(&Span, &Recovery)>,
    tracing: Res<Tracing>,
    mut commands: Commands,
) {
    let Ok((effect, of, tool)) = calls.get(admitted.entity) else {
        return;
    };
    let turn = of.and_then(|&CallOf(turn)| turns.get(turn).ok());
    let parent = turn.map(|(Span(context), _)| context);
    let purpose = effect.purpose.clone();
    let mut attributes = Vec::new();
    if let Some(purpose) = &purpose {
        attributes.push(KeyValue::new("rig.purpose", purpose.to_string()));
    }
    let span = match tool {
        true => {
            attributes.push(KeyValue::new("gen_ai.operation.name", "execute_tool"));
            tracing.start("tool.call", SpanKind::Internal, parent, attributes)
        }
        false => {
            attributes.push(KeyValue::new("gen_ai.operation.name", "chat"));
            if let Some((_, recovery)) = turn {
                attributes.push(count("rig.retries", u64::from(recovery.retries)));
            }
            let name = match (parent, purpose) {
                (None, Some(purpose)) => purpose,
                _ => "model.request".into(),
            };
            tracing.start(name, SpanKind::Client, parent, attributes)
        }
    };
    commands.entity(admitted.entity).insert(span);
}

/// The usage, model and finish of a model call's response.
fn response_attributes(response: &CompletionResponse) -> Vec<KeyValue> {
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
    let mut attributes: Vec<KeyValue> = tokens
        .into_iter()
        .filter_map(|(attribute, value)| Some(count(attribute, value?)))
        .collect();
    if let Some(cost) = &usage.cost {
        attributes.push(KeyValue::new("rig.cost.usd", cost.total));
    }
    if let Some(model) = response.model() {
        attributes.push(KeyValue::new("gen_ai.response.model", model.to_owned()));
    }
    if let Some(id) = response.response_id() {
        attributes.push(KeyValue::new("gen_ai.response.id", id.to_owned()));
    }
    if let Some(reason) = response.finish_reason() {
        let reason = format!("{reason:?}").to_lowercase();
        attributes.push(KeyValue::new("gen_ai.response.finish_reasons", reason));
    }
    attributes
}

/// Ends a model call's span as it settles: the model it went to, the time
/// to its first streamed item, its usage, cost and finish, or its error,
/// retried or not.
fn end_model_call(
    settled: On<Add<Settled>>,
    calls: Query<(
        &Span,
        &ModelRequest,
        &Outcome,
        &ForAgent,
        Option<(&Started, &StreamTiming)>,
        Option<&RetryVerdict>,
        Has<SupersededBy>,
    )>,
    agents: Query<&Connection>,
) {
    let Ok((Span(context), request, outcome, &ForAgent(agent), first, verdict, superseded)) =
        calls.get(settled.entity)
    else {
        return;
    };
    let span = context.span();
    if let Some((started, first)) = first {
        let waited = first.first.duration_since(started.at).unwrap_or_default();
        let ms = u64::try_from(waited.as_millis()).unwrap_or(u64::MAX);
        span.set_attribute(count("gen_ai.response.time_to_first_chunk_ms", ms));
    }
    if let Some(connection) = request.connection.as_ref().or(agents.get(agent).ok()) {
        let spec = &connection.spec;
        span.set_attribute(KeyValue::new("gen_ai.request.model", spec.id.clone()));
        let vendor = spec.provider.vendor().to_string();
        span.set_attribute(KeyValue::new("gen_ai.provider.name", vendor));
    }
    span.set_attribute(KeyValue::new("rig.superseded", superseded));
    if let Some(RetryVerdict(verdict)) = verdict {
        let verdict = format!("{verdict:?}").to_lowercase();
        span.set_attribute(KeyValue::new("rig.retry.verdict", verdict));
    }
    match &outcome.0 {
        Ok(_) => {
            let Some(response) = outcome.completion() else {
                span.set_status(Status::error("not a completion"));
                span.end();
                return;
            };
            for attribute in response_attributes(response) {
                span.set_attribute(attribute);
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

/// Ends a tool call's span as it settles: its name, arguments and result,
/// an error status for a failure and `denied` for a refusal.
fn end_tool_call(
    settled: On<Add<Settled>>,
    calls: Query<(&Span, &ToolRequest, &Outcome)>,
    tracing: Res<Tracing>,
) {
    let Ok((Span(context), request, outcome)) = calls.get(settled.entity) else {
        return;
    };
    let span = context.span();
    let call = &request.call;
    let arguments = call.function.raw_arguments();
    let result = outcome.tool_result();
    let output = result.output().render();
    span.set_attributes([
        KeyValue::new("gen_ai.tool.name", call.function.name.as_str().to_owned()),
        KeyValue::new("gen_ai.tool.call.id", call.id.to_string()),
        size("rig.tool.arguments.size", arguments.len()),
        size("rig.tool.result.size", output.len()),
        KeyValue::new("rig.tool.status", result.status_name()),
        KeyValue::new("rig.tool.is_error", !result.is_success()),
    ]);
    if tracing.capture_content {
        span.set_attribute(KeyValue::new("gen_ai.tool.call.arguments", arguments));
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

/// On exit, after the turns the exit stopped were despawned (which ends
/// their spans as unfinished), sends what is batched.
fn flush_on_exit(tracing: Res<Tracing>) {
    if let Err(error) = tracing.provider.shutdown_with_timeout(EXPORT_TIMEOUT) {
        warn!("OpenTelemetry: the last spans were not sent: {error}");
    }
}

/// `/telemetry`: where spans go, and how many were exported or dropped.
fn telemetry(
    In(args): In<CommandArgs>,
    tracing: Res<Tracing>,
    turns: Query<(), (With<Span>, With<TurnOf>)>,
    mut notices: MessageWriter<Notice>,
) {
    let counts = &tracing.counts;
    let text = format!(
        "OpenTelemetry tracing to {}: {} spans exported, {} dropped, {} turns open. Content \
         capture is {}.",
        tracing.destination,
        counts.exported.load(Ordering::Relaxed),
        counts.dropped.load(Ordering::Relaxed),
        turns.iter().count(),
        if tracing.capture_content { "on" } else { "off" },
    );
    notices.write(Notice::info(args.agent, text));
}

#[cfg(test)]
mod tests;

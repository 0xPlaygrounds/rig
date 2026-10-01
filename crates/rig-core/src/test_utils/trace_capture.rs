//! A `tracing` layer that records spans and events for test assertions.

use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Map, Value};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;

/// Records every span and event it sees. Clones share one record, so a test
/// keeps a clone and installs another with [`TraceCapture::subscriber`].
///
/// Field values are JSON: strings and `Debug`/`Display` renderings are
/// strings, integers are numbers and booleans are booleans.
#[derive(Clone, Default)]
pub struct TraceCapture(Arc<Mutex<Captured>>);

#[derive(Default)]
struct Captured {
    spans: Vec<CapturedSpan>,
    events: Vec<CapturedEvent>,
}

impl Captured {
    fn span_mut(&mut self, id: &Id) -> Option<&mut CapturedSpan> {
        self.spans
            .iter_mut()
            .rev()
            .find(|span| span.id == id.into_u64())
    }
}

/// One span as [`TraceCapture`] saw it.
#[derive(Clone, Debug)]
pub struct CapturedSpan {
    /// The span's id, unique while the span is open.
    pub id: u64,
    /// The span's name.
    pub name: &'static str,
    /// The span's target.
    pub target: &'static str,
    /// The explicit parent, or the current span for a contextual span.
    pub parent: Option<u64>,
    /// The parent's name.
    pub parent_name: Option<&'static str>,
    /// Every field the span declares, in declaration order.
    pub declared: Vec<&'static str>,
    /// The values given when the span was created.
    pub initial: Map<String, Value>,
    /// Every value recorded later, in order, repeats included.
    pub recorded: Vec<(String, Value)>,
    /// The ids this span follows from.
    pub follows_from: Vec<u64>,
}

impl CapturedSpan {
    /// The latest value of `field`: its last recording, else its initial value.
    pub fn value(&self, field: &str) -> Option<&Value> {
        self.recorded
            .iter()
            .rev()
            .find(|(name, _)| name == field)
            .map(|(_, value)| value)
            .or_else(|| self.initial.get(field))
    }

    /// [`Self::value`] as text: a string as it is, anything else as JSON.
    pub fn text(&self, field: &str) -> Option<String> {
        self.value(field).map(text)
    }

    /// The latest value of `field` when it is an unsigned integer.
    pub fn u64(&self, field: &str) -> Option<u64> {
        self.value(field).and_then(Value::as_u64)
    }

    /// The latest value recorded after creation, ignoring initial values.
    pub fn recorded_value(&self, field: &str) -> Option<&Value> {
        self.recorded
            .iter()
            .rev()
            .find(|(name, _)| name == field)
            .map(|(_, value)| value)
    }

    /// How many times `field` was recorded after creation.
    pub fn record_count(&self, field: &str) -> usize {
        self.recorded
            .iter()
            .filter(|(name, _)| name == field)
            .count()
    }

    /// The initial values with every later recording applied in order.
    pub fn values(&self) -> Map<String, Value> {
        let mut values = self.initial.clone();
        values.extend(self.recorded.iter().cloned());
        values
    }

    /// The span without its id, which differs between runs: name, target,
    /// parent name, declared fields and [`Self::values`].
    pub fn summary(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "target": self.target,
            "parent": self.parent_name,
            "fields": self.declared,
            "values": self.values(),
        })
    }
}

/// One event as [`TraceCapture`] saw it.
#[derive(Clone, Debug)]
pub struct CapturedEvent {
    /// The event's level.
    pub level: Level,
    /// The event's target.
    pub target: &'static str,
    /// The event's fields, the message under `message`.
    pub fields: Map<String, Value>,
}

impl CapturedEvent {
    /// The event's message.
    pub fn message(&self) -> String {
        self.fields.get("message").map(text).unwrap_or_default()
    }
}

fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

impl TraceCapture {
    /// A registry with this capture as its only layer.
    pub fn subscriber(&self) -> impl Subscriber + Send + Sync + 'static {
        tracing_subscriber::registry().with(self.clone())
    }

    /// Every span seen so far, in creation order.
    pub fn spans(&self) -> Vec<CapturedSpan> {
        self.lock().spans.clone()
    }

    /// Every event seen so far, in order.
    pub fn events(&self) -> Vec<CapturedEvent> {
        self.lock().events.clone()
    }

    /// The last span opened so far.
    pub fn last_span(&self) -> Option<CapturedSpan> {
        self.lock().spans.last().cloned()
    }

    /// Every value `field` took on any span, at creation or later, span by
    /// span in creation order.
    pub fn values_of(&self, field: &str) -> Vec<Value> {
        let captured = self.lock();
        let mut values = Vec::new();
        for span in &captured.spans {
            values.extend(span.initial.get(field).cloned());
            values.extend(
                span.recorded
                    .iter()
                    .filter(|(name, _)| name == field)
                    .map(|(_, value)| value.clone()),
            );
        }
        values
    }

    /// The events at WARN, in order, each as its message followed by
    /// ` name=value` for every other field.
    pub fn warnings(&self) -> Vec<String> {
        self.lock()
            .events
            .iter()
            .filter(|event| event.level == Level::WARN)
            .map(|event| {
                let mut rendered = event.message();
                for (name, value) in event.fields.iter().filter(|(name, _)| *name != "message") {
                    rendered.push_str(&format!(" {name}={}", text(value)));
                }
                rendered
            })
            .collect()
    }

    /// Forgets every span and event seen so far.
    pub fn clear(&self) {
        let mut captured = self.lock();
        captured.spans.clear();
        captured.events.clear();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Captured> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<S> Layer<S> for TraceCapture
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let parent = match attrs.parent() {
            Some(parent) => ctx.span(parent),
            None if attrs.is_contextual() => ctx.lookup_current(),
            None => None,
        };
        let mut initial = Map::new();
        attrs.record(&mut Values(&mut initial));
        let metadata = attrs.metadata();
        self.lock().spans.push(CapturedSpan {
            id: id.into_u64(),
            name: metadata.name(),
            target: metadata.target(),
            parent: parent.as_ref().map(|span| span.id().into_u64()),
            parent_name: parent.as_ref().map(|span| span.name()),
            declared: metadata.fields().iter().map(|field| field.name()).collect(),
            initial,
            recorded: Vec::new(),
            follows_from: Vec::new(),
        });
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, _: Context<'_, S>) {
        let mut fields = Map::new();
        values.record(&mut Values(&mut fields));
        if let Some(span) = self.lock().span_mut(id) {
            span.recorded.extend(fields);
        }
    }

    fn on_follows_from(&self, id: &Id, follows: &Id, _: Context<'_, S>) {
        if let Some(span) = self.lock().span_mut(id) {
            span.follows_from.push(follows.into_u64());
        }
    }

    fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
        let mut fields = Map::new();
        event.record(&mut Values(&mut fields));
        self.lock().events.push(CapturedEvent {
            level: *event.metadata().level(),
            target: event.metadata().target(),
            fields,
        });
    }
}

struct Values<'a>(&'a mut Map<String, Value>);

impl Visit for Values<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().into(), Value::String(format!("{value:?}")));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().into(), Value::from(value));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().into(), Value::from(value));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.0.insert(field.name().into(), Value::from(value));
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.0.insert(field.name().into(), Value::from(value));
    }
}

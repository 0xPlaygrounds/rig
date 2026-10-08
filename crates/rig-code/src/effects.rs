//! The one path every model call and tool call takes, and the session's
//! effect log.
//!
//! [`EffectHub::dispatch`] mints an [`EffectId`], opens a record in the
//! shared [`EffectLogRecorder`] scoped to the agent's stable id, and spawns
//! the handler on a Bevy task pool. A `Last` system appends resolved records
//! to `effects.jsonl` in the session directory.

use std::{
    collections::HashMap,
    io::Write as _,
    panic::AssertUnwindSafe,
    sync::{Arc, mpsc::Sender},
};

use bevy_ecs::prelude::*;
use bevy_tasks::{AsyncComputeTaskPool, IoTaskPool, Task};
use futures::{FutureExt as _, StreamExt as _};
use rig_cassette::effect_log::{EffectLogRecorder, LogHeader};
use rig_core::{
    ErrorKind, ErrorReport,
    catalog::ModelSpec,
    effect::{EffectId, EffectKind, EffectRecord, HandlerKey, Outcome},
    providers::registry::ModelSelector,
    serve::{
        Dispatch, ErasedHandler, Observe, Origin, Recorder as _, Reply, StreamTap,
        adapters::ModelAdapter, stream_truncated,
    },
    streaming::{Item, PartKind, Relayed, StreamEvent},
};

use crate::{
    agent::{AgentId, Feed},
    session::Session,
    tools::ToolDef,
};

/// The dispatcher: the effect recorder, the next effect id, and the model
/// handlers built so far, by catalog reference.
#[derive(Resource, Default)]
pub struct EffectHub {
    recorder: EffectLogRecorder,
    pub(crate) next_id: u64,
    models: HashMap<String, ErasedHandler>,
    header_handlers: Option<usize>,
}

impl EffectHub {
    /// The handler serving `reference`, built from the environment's
    /// credentials on first use.
    pub fn model(&mut self, reference: &str, spec: &ModelSpec) -> Result<ErasedHandler, String> {
        if let Some(handler) = self.models.get(reference) {
            return Ok(handler.clone());
        }
        let model = ModelSelector::from(spec)
            .provider_ref()
            .map_err(|error| error.to_string())?
            .completion_model()
            .map_err(|error| error.to_string())?;
        let handler = ErasedHandler::new(ModelAdapter::new(reference, model));
        self.recorder.handlers(vec![handler.descriptor()]);
        self.models.insert(reference.to_owned(), handler.clone());
        Ok(handler)
    }

    /// Record and start one effect for `agent`. `handler` is `None` when
    /// nothing serves `key`; the effect then resolves as unavailable. A
    /// streaming completion sends its text to `feed` as it arrives. Tool
    /// calls run on the async compute pool, model calls on the IO pool.
    pub fn dispatch(
        &mut self,
        agent: &AgentId,
        key: HandlerKey,
        handler: Option<ErasedHandler>,
        kind: EffectKind,
        feed: Option<Sender<Feed>>,
    ) -> Task<Result<Outcome, ErrorReport>> {
        self.next_id += 1;
        let id = EffectId::from_raw(self.next_id);
        let origin = Origin {
            parent: None,
            scope: Some(Arc::from(agent.0.as_str())),
        };
        let recorder = self.recorder.clone();
        recorder.begin(id, key.clone(), kind.clone(), origin);
        let dispatch = Dispatch::new(id, kind.streams()).with_observer(Box::new(Recorded {
            recorder: recorder.clone(),
            id,
        }));
        let is_tool = matches!(kind, EffectKind::ToolCall { .. });
        let work = async move {
            let Some(handler) = handler else {
                let error = ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    format!("nothing serves `{}`", key.as_str()),
                );
                recorder.resolve(id, Err(error.clone()));
                return Err(error);
            };
            let served = AssertUnwindSafe(serve(&handler, kind, dispatch, feed))
                .catch_unwind()
                .await;
            served.unwrap_or_else(|panic| {
                let message = panic
                    .downcast_ref::<&str>()
                    .map(|text| (*text).to_owned())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_owned());
                let error = ErrorReport::new(
                    ErrorKind::Internal,
                    format!("`{}` panicked: {message}", key.as_str()),
                );
                recorder.resolve(id, Err(error.clone()));
                Err(error)
            })
        };
        if is_tool {
            AsyncComputeTaskPool::get_or_init(Default::default).spawn(work)
        } else {
            IoTaskPool::get_or_init(Default::default).spawn(work)
        }
    }
}

/// Serve one effect, sending a streamed reply's text to `feed` and folding
/// it to its outcome.
async fn serve(
    handler: &ErasedHandler,
    kind: EffectKind,
    dispatch: Dispatch,
    feed: Option<Sender<Feed>>,
) -> Result<Outcome, ErrorReport> {
    let (mut stream, feed) = match (handler.handle(kind, dispatch).await, feed) {
        (Reply::Stream(stream), Some(feed)) => (stream, feed),
        (reply, _) => return reply.into_outcome().await,
    };
    let mut tap = StreamTap::new();
    while let Some(item) = stream.next().await {
        let piece = match &item {
            Ok(Relayed::Origin(origin)) => Some(Feed::Origin(origin.clone())),
            Ok(Relayed::Item(Item::Event(StreamEvent::Text { text, .. }))) => {
                Some(Feed::Text(text.clone()))
            }
            Ok(Relayed::Item(Item::Event(StreamEvent::Reasoning { text, .. }))) => {
                Some(Feed::Reasoning(text.clone()))
            }
            Ok(Relayed::Item(Item::Event(StreamEvent::Start {
                kind: PartKind::ToolCall,
                name: Some(name),
                ..
            }))) => Some(Feed::ToolStart(name.as_str().to_owned())),
            _ => None,
        };
        if let Some(piece) = piece {
            // The call entity is gone when nobody listens; the fold goes on.
            let _ = feed.send(piece);
        }
        if let Some(outcome) = tap.observe(&item) {
            return outcome;
        }
    }
    Err(stream_truncated())
}

/// Tells the recorder what happens to one dispatch.
struct Recorded {
    recorder: EffectLogRecorder,
    id: EffectId,
}

impl Observe for Recorded {
    fn outcome(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        self.recorder.resolve(self.id, outcome.clone());
    }

    fn keep_events(&self) -> bool {
        self.recorder.keep_events()
    }

    fn event(&mut self, item: &Item<StreamEvent>) {
        self.recorder.event(self.id, item);
    }

    fn stream_error(&mut self, error: &ErrorReport) {
        self.recorder.stream_error(self.id, error);
    }

    fn origin(&mut self, origin: &rig_core::message::Origin) {
        self.recorder.origin(self.id, origin);
    }

    fn discard(&mut self, _layer: &str) {
        self.recorder.discard(self.id);
    }

    fn patch(&mut self, kind: &EffectKind) {
        self.recorder.patch(self.id, kind.clone());
    }
}

/// Describe every registered tool to the recorder, once at startup.
pub(crate) fn describe_tools(hub: Res<EffectHub>, tools: Query<&ToolDef>) {
    hub.recorder
        .handlers(tools.iter().map(|tool| tool.handler.descriptor()).collect());
}

/// One line of `effects.jsonl`: `{"header": ..}` or `{"record": ..}`.
#[derive(serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Line<'a> {
    Header(&'a LogHeader),
    Record(&'a EffectRecord),
}

/// Append resolved records to the session's `effects.jsonl`, preceded by
/// the log header whenever the set of described handlers grew.
pub(crate) fn flush_effects(mut hub: ResMut<EffectHub>, session: Res<Session>) {
    let log = hub.recorder.take();
    let handlers = log.header.handlers.len();
    let header_due = hub.header_handlers != Some(handlers);
    if log.records.is_empty() && !header_due {
        return;
    }
    let mut lines = String::new();
    let mut push = |line: Line<'_>| match serde_json::to_string(&line) {
        Ok(line) => {
            lines.push_str(&line);
            lines.push('\n');
        }
        Err(error) => bevy_log::warn!("an effect record does not serialize: {error}"),
    };
    if header_due {
        push(Line::Header(&log.header));
    }
    for record in &log.records {
        push(Line::Record(record));
    }
    let path = session.dir.join("effects.jsonl");
    let written = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut file| file.write_all(lines.as_bytes()));
    match written {
        Ok(()) => hub.header_handlers = Some(handlers),
        Err(error) => bevy_log::warn!("cannot write {}: {error}", path.display()),
    }
}

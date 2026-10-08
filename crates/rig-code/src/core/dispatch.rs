//! The one path every model call and tool call takes: `Effects::dispatch`
//! records the call with rig-core's effect types into an
//! `EffectLogRecorder`, scoped by the agent's stable id, and the recorded
//! effects are appended to the session's `effects.jsonl`.

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy::prelude::*;
use rig_cassette::effect_log::{EffectLogRecorder, LogHeader};
use rig_core::ErrorReport;
use rig_core::effect::{EffectId, EffectKind, Outcome};
use rig_core::serve::{Dispatch, ErasedHandler, Observe, Origin, Recorder, Reply};
use rig_core::streaming::{Item, StreamEvent};

use super::agent::AgentId;
use super::session::Session;

/// Records every dispatch of the app. Shared: the recorder and the id
/// counter are handles.
#[derive(Resource, Clone, Default)]
pub struct Effects {
    recorder: EffectLogRecorder,
    next: Arc<AtomicU64>,
}

impl Effects {
    /// Serve `kind` with `handler` for the agent `scope`, recording it. The
    /// returned id names the effect; `parent` is the effect that asked for
    /// this one. The future owns everything it needs; dropping it cancels
    /// the call, which is recorded as cancelled.
    pub fn dispatch(
        &self,
        scope: &AgentId,
        parent: Option<EffectId>,
        handler: ErasedHandler,
        kind: EffectKind,
    ) -> (EffectId, impl Future<Output = Reply> + Send + 'static) {
        let id = EffectId::from_raw(self.next.fetch_add(1, Ordering::Relaxed));
        let descriptor = handler.descriptor();
        let streaming = matches!(kind, EffectKind::Completion { stream: true, .. });
        self.recorder.handlers(vec![descriptor.clone()]);
        self.recorder.begin(
            id,
            descriptor.key,
            kind.clone(),
            Origin {
                parent,
                scope: Some(Arc::from(scope.0.as_str())),
            },
        );
        let dispatch = Dispatch::new(id, streaming).with_observer(Box::new(Bridge {
            recorder: self.recorder.clone(),
            id,
        }));
        (id, async move { handler.handle(kind, dispatch).await })
    }

    /// The id the next dispatch gets.
    pub(crate) fn next_id(&self) -> u64 {
        self.next.load(Ordering::Relaxed)
    }

    /// Continue numbering after a restored session's last effect.
    pub(crate) fn resume_at(&self, next: u64) {
        self.next.fetch_max(next, Ordering::Relaxed);
    }
}

/// Tells the recorder what one dispatch's handler answered.
struct Bridge {
    recorder: EffectLogRecorder,
    id: EffectId,
}

impl Observe for Bridge {
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

/// Appends the resolved effects to `effects.jsonl` and rewrites
/// `effects-header.json` in the session directory.
pub(crate) fn flush_effects(effects: Res<Effects>, session: Res<Session>) {
    let log = effects.recorder.take();
    if log.records.is_empty() {
        return;
    }
    if let Err(error) = write_log(&session, &log) {
        error!("cannot write the effect log of {}: {error}", session.id);
    }
}

fn write_log(session: &Session, log: &rig_cassette::effect_log::EffectLog) -> std::io::Result<()> {
    std::fs::create_dir_all(&session.dir)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(session.dir.join("effects.jsonl"))?;
    let mut lines = Vec::new();
    for record in &log.records {
        serde_json::to_writer(&mut lines, record)?;
        lines.push(b'\n');
    }
    file.write_all(&lines)?;
    let path = session.dir.join("effects-header.json");
    let mut header = log.header.clone();
    if let Ok(bytes) = std::fs::read(&path) {
        match serde_json::from_slice::<LogHeader>(&bytes) {
            Ok(previous) => keep_previous(&mut header, previous),
            Err(error) => warn!("replacing the unreadable {}: {error}", path.display()),
        }
    }
    super::session::write_atomic(&path, &serde_json::to_vec_pretty(&header)?)
}

/// The recorder's header covers only this process's dispatches since the
/// last flush; keep what earlier flushes and earlier builds (before a
/// `/reload`) wrote, so the header describes the whole `effects.jsonl`.
fn keep_previous(header: &mut LogHeader, previous: LogHeader) {
    let mut handlers = previous.handlers;
    for handler in std::mem::take(&mut header.handlers) {
        match handlers.iter_mut().find(|known| known.key == handler.key) {
            Some(known) => *known = handler,
            None => handlers.push(handler),
        }
    }
    header.handlers = handlers;
    for (key, family) in previous.signature {
        header.signature.insert_if_absent(key, family);
    }
    for (id, errors) in previous.stream_errors {
        header.stream_errors.entry(id).or_insert(errors);
    }
}

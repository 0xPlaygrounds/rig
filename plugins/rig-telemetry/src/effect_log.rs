//! The effect log: every effect the agents' work settles, model calls,
//! tool calls, delivered messages and plugins' own effects, recorded with
//! rig-core's effect types into the session's [`SESSION_EFFECT_LOG`], a
//! file of its store ([`Files`]): each effect's [`EffectRecorded`] record, in id order, as it settles. Its
//! header describes the tools and the models the agents connect to, and
//! names each key the records use. Ids continue after the highest one the
//! log already holds, so a resumed session keeps adding to the same log.
//! A plugin reads it back with the [`EffectLog`] resource, narrowed by an
//! [`EffectFilter`]. With the default `tui` feature, in a run of the
//! terminal view, a restored session's log is read back once, off the main
//! thread, and sent to the view as `rig_tui::PastEffects`, so a resumed
//! session's tool calls show the calls they started and what they
//! published. When the session is rewritten ([`SessionRewritten`]), the
//! file was rewritten with the rest of the store: the writer forgets what
//! it knew of the old lines, and the records not written yet go through
//! the same map.

use std::io;
use std::sync::Arc;

use bevy_log::error;
#[cfg(feature = "tui")]
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_cassette::effect_log::jsonl::SESSION_EFFECT_LOG;
use rig_cassette::effect_log::{self, LogHeader, jsonl};
use rig_core::effect::{EffectRecord, HandlerDescriptor};
use rig_core::serde_json;
use rig_ecs::WriteJournal;
use rig_ecs::effects::Effects;
#[cfg(feature = "tui")]
use rig_ecs::journal::SessionRestored;
use rig_ecs::journal::SessionStore;
use rig_ecs::prelude::*;
use rig_ecs::tools::ToolDef;
#[cfg(feature = "tui")]
use rig_harness::prelude::AppCliExt;
use rig_harness::rig_memory::Place;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Records every effect into the session's effect log, in its store, once
/// the harness opened it; without a store it records nothing.
#[derive(Default)]
pub struct EffectLogPlugin;

impl Plugin for EffectLogPlugin {
    fn build(&self, _: &mut App) {}

    fn finish(&self, app: &mut App) {
        let Some(SessionStore(store)) = app.world().get_resource::<SessionStore>().cloned() else {
            return;
        };
        let store: Arc<dyn Files> = store;
        let mut writer = jsonl::Writer::on(store.clone(), SESSION_EFFECT_LOG);
        let last = writer.last_id().ok().flatten();
        let header = LogHeader::default();
        app.insert_resource(Effects::continue_after(last))
            .insert_resource(EffectLog { store })
            .insert_resource(EffectLogFile { header, writer })
            .add_systems(Startup, describe_tools)
            .add_systems(Last, write_effects.in_set(WriteJournal))
            .add_observer(describe_model);
        #[cfg(feature = "tui")]
        if app.runs_front(rig_tui::FRONT) {
            app.add_systems(Update, read_past.run_if(resource_added::<SessionRestored>))
                .add_observer(send_past);
        }
    }
}

/// The past effects [`read_past`] read and worked out off the main thread,
/// until [`send_past`] takes them.
#[cfg(feature = "tui")]
#[derive(Component)]
struct PastRead(Option<io::Result<rig_tui::PastEffects>>);

#[cfg(feature = "tui")]
impl From<io::Result<rig_tui::PastEffects>> for PastRead {
    fn from(read: io::Result<rig_tui::PastEffects>) -> Self {
        Self(Some(read))
    }
}

/// Reads the log of the session just restored off the main thread, once,
/// for the terminal view's call tree ([`rig_tui::PastEffects`]): a resumed
/// session's calls show what they started and published, as this run's do.
/// The task reads only what the tree needs, no request's history, and does
/// all the work per record, so the main thread only hands the result over.
#[cfg(feature = "tui")]
fn read_past(log: Res<EffectLog>, wake: Res<Wake>, mut commands: Commands) {
    let log = log.clone();
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let read = async move { log.read_outcomes().map(rig_tui::PastEffects::new) };
    commands.spawn((
        Name::new("effect log: past effects"),
        Running::spawn_into::<PastRead, _>(pool, &wake, read),
    ));
}

/// Sends what [`read_past`] read to the terminal view; a log that could not
/// be read leaves the view without the past calls' children and facts, and
/// says so.
#[cfg(feature = "tui")]
fn send_past(
    read: On<Insert<PastRead>>,
    mut reads: Query<&mut PastRead>,
    mut past: MessageWriter<rig_tui::PastEffects>,
    mut commands: Commands,
) {
    match reads
        .get_mut(read.entity)
        .ok()
        .and_then(|mut read| read.0.take())
    {
        Some(Ok(effects)) => {
            past.write(effects);
        }
        Some(Err(failure)) => error!("reading the effect log failed: {failure}"),
        None => {}
    }
    commands.entity(read.entity).despawn();
}

/// The session's effect log, to read back; there when
/// [`EffectLogPlugin`] records one. It holds what was written by the end
/// of the last frame: the records of this frame's effects are written in
/// its `Last`.
#[derive(Resource, Clone)]
pub struct EffectLog {
    store: Arc<dyn Files>,
}

impl EffectLog {
    /// Every record written so far, in the order written; none before the
    /// first. It reads the whole file and rebuilds every model call's
    /// request: off the main thread, such as in a task, for a long session.
    pub fn read(&self) -> io::Result<Vec<EffectRecord>> {
        let bytes = self.bytes()?;
        Ok(jsonl::read_from(bytes.as_slice())?.records)
    }

    /// Every record written so far, as [`read`](Self::read) gives them but
    /// with each model call's request history and tools left empty
    /// ([`jsonl::read_outcomes_from`]): what each effect was and how it
    /// ended, read in time linear in the log's size.
    pub fn read_outcomes(&self) -> io::Result<Vec<EffectRecord>> {
        jsonl::read_outcomes_from(self.bytes()?.as_slice())
    }

    /// The log's bytes; none before the first record.
    fn bytes(&self) -> io::Result<Vec<u8>> {
        match self.store.read_file(SESSION_EFFECT_LOG) {
            Ok(bytes) => Ok(bytes),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(error) => Err(error),
        }
    }

    /// The records `filter` matches, in the order written.
    pub fn read_matching(&self, filter: &EffectFilter) -> io::Result<Vec<EffectRecord>> {
        let mut records = self.read()?;
        records.retain(|record| filter.matches(record));
        Ok(records)
    }
}

/// Which records of the [`EffectLog`] to read: each part set narrows it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EffectFilter {
    /// The agent whose work it was, by its id (the record's scope).
    pub agent: Option<String>,
    /// The effect's kind by name: `completion`, `tool_call`, or a plugin's
    /// own, such as `subagent`.
    pub kind: Option<String>,
    /// The turn of its agent it was made in (the record's turn; the first
    /// is 0). Turns are numbered per agent, so pair it with
    /// [`agent`](Self::agent).
    pub turn: Option<u64>,
}

impl EffectFilter {
    /// The work of the agent with the id `agent`.
    #[must_use]
    pub fn agent(self, agent: impl Into<String>) -> Self {
        Self {
            agent: Some(agent.into()),
            ..self
        }
    }

    /// Effects of the kind named `kind`.
    #[must_use]
    pub fn kind(self, kind: impl Into<String>) -> Self {
        Self {
            kind: Some(kind.into()),
            ..self
        }
    }

    /// Effects made in turn `turn` of their agent.
    #[must_use]
    pub fn turn(self, turn: u64) -> Self {
        Self {
            turn: Some(turn),
            ..self
        }
    }

    /// Whether it keeps `record`.
    pub fn matches(&self, record: &EffectRecord) -> bool {
        let agent = record.scope.as_deref();
        self.agent
            .as_deref()
            .is_none_or(|wanted| agent == Some(wanted))
            && (self.kind.as_deref()).is_none_or(|kind| record.kind.name() == kind)
            && self.turn.is_none_or(|turn| record.turn == Some(turn))
    }
}

/// The log's header as it grows, and the file it is written to.
#[derive(Resource)]
struct EffectLogFile {
    header: LogHeader,
    writer: jsonl::Writer,
}

impl EffectLogFile {
    /// Describes each of `handlers` in the header: one already described
    /// in place, a new one after the others.
    fn describe(&mut self, handlers: impl IntoIterator<Item = HandlerDescriptor>) {
        let known = &mut self.header.handlers;
        for handler in handlers {
            match known.iter_mut().find(|known| known.key == handler.key) {
                Some(known) => *known = handler,
                None => known.push(handler),
            }
        }
    }
}

/// Describes every registered tool in the log's header, by name.
fn describe_tools(mut log: ResMut<EffectLogFile>, tools: Query<&ToolDef>) {
    let mut tools: Vec<_> = tools.iter().map(|tool| &tool.0).collect();
    tools.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
    log.describe(tools.into_iter().map(|tool| {
        HandlerDescriptor::tool(
            tool.name.as_str(),
            &tool.description,
            tool.parameters.clone(),
        )
    }));
}

/// Describes each model an agent connects to in the log's header.
fn describe_model(
    connected: On<Insert<Connection>>,
    agents: Query<&Connection>,
    mut log: ResMut<EffectLogFile>,
) {
    if let Ok(connection) = agents.get(connected.entity) {
        log.describe([connection.handler.0.descriptor()]);
    }
}

/// Writes the effects settled since the last frame, in id order, mapped
/// through each rewrite of the session this frame. A failure is shown once.
fn write_effects(
    mut log: ResMut<EffectLogFile>,
    (mut recorded, mut rewrites): (
        MessageReader<EffectRecorded>,
        MessageReader<SessionRewritten>,
    ),
    mut failed: Local<bool>,
    mut notices: MessageWriter<Notice>,
) {
    let mut records: Vec<_> = recorded.read().map(|recorded| recorded.0.clone()).collect();
    let EffectLogFile { header, writer } = &mut *log;
    for rewritten in rewrites.read() {
        // The file pass rewrote the file: nothing of its old lines is kept.
        writer.forget();
        let with = &*rewritten.with;
        *header = mapped(header.clone(), with).unwrap_or_default();
        let before = records.len();
        records = records
            .into_iter()
            .filter_map(|record| mapped(record, with))
            .collect();
        if records.len() < before {
            error!("effect records the rewrite left unreadable were not logged");
        }
    }
    if records.is_empty() {
        return;
    }
    records.sort_by_key(|record| record.id);
    for record in &records {
        let family = record.kind.family();
        header
            .signature
            .insert_if_absent(record.key.clone(), family);
    }
    // The writer states only what the header adds to the file's.
    let taken = effect_log::EffectLog {
        header: header.clone(),
        records,
    };
    if let Err(failure) = writer.append(&taken)
        && !*failed
    {
        *failed = true;
        error!("writing the effect log failed: {failure}");
        notices.write(Notice::error(
            None,
            format!("Writing the effect log failed: {failure}"),
        ));
    }
}

/// `item` mapped through `with` by its serde form, as the rewrite's file
/// pass maps a line of the log; `None` when the mapped form does not read
/// back.
fn mapped<T: Serialize + DeserializeOwned>(item: T, with: &dyn Rewrite) -> Option<T> {
    let at = Place::File {
        name: SESSION_EFFECT_LOG,
    };
    let Ok(mut value) = serde_json::to_value(&item) else {
        return Some(item);
    };
    if !with.value(&at, SESSION_EFFECT_LOG, &mut value) {
        return Some(item);
    }
    serde_json::from_value(value).ok()
}

#[cfg(test)]
mod tests;

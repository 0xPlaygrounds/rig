//! The effect log: every settled effect (model calls, tool calls,
//! delivered messages, plugins' own) as an [`EffectRecorded`] record in the
//! session's [`SESSION_EFFECT_LOG`] ([`Files`]), in id order. Its header
//! describes the tools and models and names each key the records use.
//!
//! A resumed session appends after the highest id. Plugins read it with [`EffectLog`] and an [`EffectFilter`].
//! With the `tui` feature, a restored session's log is read once, off the
//! main thread, and sent to the view as `rig_tui::PastEffects`, so its
//! tool calls show what they started.

use std::io;
use std::sync::Arc;

use bevy_log::error;
#[cfg(feature = "tui")]
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_cassette::effect_log::jsonl::SESSION_EFFECT_LOG;
use rig_cassette::effect_log::{EffectFilter, LogHeader, jsonl};
use rig_core::effect::{EffectRecord, HandlerDescriptor};
use rig_ecs::WriteLogs;
use rig_ecs::effects::Effects;
use rig_ecs::prelude::*;
#[cfg(feature = "tui")]
use rig_ecs::session_log::SessionRestored;
use rig_ecs::session_log::SessionStore;
use rig_ecs::tools::ToolDef;
#[cfg(feature = "tui")]
use rig_harness::prelude::AppCliExt;

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
            .add_systems(
                Last,
                (describe_tools, write_effects).chain().in_set(WriteLogs),
            )
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
        Ok(jsonl::read_session(&*self.store)?.records)
    }

    /// Every record written so far, as [`read`](Self::read) gives them but
    /// with each model call's request history and tools left empty
    /// ([`jsonl::read_outcomes_from`]): what each effect was and how it
    /// ended, read in time linear in the log's size.
    pub fn read_outcomes(&self) -> io::Result<Vec<EffectRecord>> {
        jsonl::read_session_outcomes(&*self.store)
    }

    /// The records `filter` matches, in the order written.
    pub fn read_matching(&self, filter: &EffectFilter) -> io::Result<Vec<EffectRecord>> {
        let mut records = self.read()?;
        records.retain(|record| filter.matches(record));
        Ok(records)
    }
}

/// The log's header as it grows, and the file it is written to.
#[derive(Resource)]
struct EffectLogFile {
    header: LogHeader,
    writer: jsonl::Writer,
}

/// Describes each tool registered or changed since the last frame in the
/// log's header, by name, before its calls are written.
fn describe_tools(mut log: ResMut<EffectLogFile>, tools: Query<&ToolDef, Changed<ToolDef>>) {
    let mut tools: Vec<_> = tools.iter().map(|tool| &tool.0).collect();
    tools.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
    log.header.describe(tools.into_iter().map(|tool| {
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
        log.header.describe([connection.handler.0.descriptor()]);
    }
}

/// Writes the effects settled since the last frame, in id order. A failure
/// is shown once.
fn write_effects(
    mut log: ResMut<EffectLogFile>,
    mut recorded: MessageReader<EffectRecorded>,
    mut failed: Local<bool>,
    mut notices: MessageWriter<Notice>,
) {
    let mut records: Vec<&EffectRecord> = recorded.read().map(|recorded| &recorded.0).collect();
    if records.is_empty() {
        return;
    }
    records.sort_by_key(|record| record.id);
    let EffectLogFile { header, writer } = &mut *log;
    records.iter().for_each(|record| header.sign(record));
    // The writer states only what the header adds to the file's.
    if let Err(failure) = writer.append(header, records.iter().copied())
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

#[cfg(test)]
mod tests;

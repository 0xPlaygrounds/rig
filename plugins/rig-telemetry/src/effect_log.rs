//! The effect log: every effect the agents' work settles, model calls,
//! tool calls, delivered messages and plugins' own effects, recorded with
//! rig-core's effect types into the session directory's [`EFFECT_LOG`]:
//! each effect's [`Recorded`] record, in id order, as it settles. Its
//! header describes the tools and the models the agents connect to, and
//! names each key the records use. Ids continue after the highest one the
//! log already holds, so a resumed session keeps adding to the same log.

use bevy_log::error;
use rig_cassette::effect_log::{EffectLog, LogHeader, jsonl};
use rig_cassette::journal::EFFECT_LOG;
use rig_core::effect::HandlerDescriptor;
use rig_ecs::WriteJournal;
use rig_ecs::effects::Effects;
use rig_ecs::prelude::*;
use rig_ecs::tools::ToolDef;

use rig_harness::prelude::SessionPaths;

/// Records every effect into the session directory's effect log; without
/// a session directory it records nothing.
#[derive(Default)]
pub struct EffectLogPlugin;

impl Plugin for EffectLogPlugin {
    fn build(&self, app: &mut App) {
        let Some(paths) = app.world().get_resource::<SessionPaths>() else {
            return;
        };
        let mut writer = jsonl::Writer::new(paths.path().join(EFFECT_LOG));
        let last = writer.last_id().ok().flatten();
        let header = LogHeader::default();
        app.insert_resource(Effects::continue_after(last))
            .insert_resource(EffectLogFile { header, writer })
            .add_systems(Startup, describe_tools)
            .add_systems(Last, write_effects.in_set(WriteJournal))
            .add_observer(describe_model);
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

/// Writes the effects settled since the last frame, in id order. A failure
/// is shown once.
fn write_effects(
    mut log: ResMut<EffectLogFile>,
    mut recorded: MessageReader<Recorded>,
    mut failed: Local<bool>,
    mut notices: MessageWriter<Notice>,
) {
    let mut records: Vec<_> = recorded.read().map(|recorded| recorded.0.clone()).collect();
    if records.is_empty() {
        return;
    }
    records.sort_by_key(|record| record.id);
    let EffectLogFile { header, writer } = &mut *log;
    for record in &records {
        let family = record.kind.family();
        header
            .signature
            .insert_if_absent(record.key.clone(), family);
    }
    // The writer states only what the header adds to the file's.
    let taken = EffectLog {
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

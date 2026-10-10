//! The effect log: every model and tool call the agents make, recorded
//! with rig-core's effect types into the session directory's
//! [`EFFECT_LOG`], so rig-cassette's replayer replays the session. Its
//! header describes the tools and the models the agents connect to. Ids
//! continue after the highest one the log already holds, so a resumed
//! session keeps adding to the same log.

use std::sync::Arc;

use bevy_app::OnAppExitSystems;
use bevy_log::error;
use rig_cassette::effect_log::{EffectLogRecorder, jsonl};
use rig_cassette::journal::EFFECT_LOG;
use rig_core::effect::HandlerDescriptor;
use rig_core::serve::Recorder;
use rig_ecs::effects::Effects;
use rig_ecs::prelude::*;
use rig_ecs::tools::ToolDef;
use rig_ecs::{StopTurns, WriteJournal};

use crate::prelude::SessionPaths;

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
        let recorder = EffectLogRecorder::new();
        app.insert_resource(Effects::recorded_by(Arc::new(recorder.clone()), last))
            .insert_resource(EffectLog { recorder, writer })
            .add_systems(Startup, describe_tools)
            .add_systems(
                Last,
                write_effects
                    .in_set(OnAppExitSystems)
                    .in_set(WriteJournal)
                    .after(StopTurns),
            )
            .add_observer(describe_model);
    }
}

/// The recorder and the file it is written to.
#[derive(Resource)]
struct EffectLog {
    recorder: EffectLogRecorder,
    writer: jsonl::Writer,
}

/// Describes every registered tool in the log's header, by name.
fn describe_tools(log: Res<EffectLog>, tools: Query<&ToolDef>) {
    let mut tools: Vec<_> = tools.iter().map(|tool| &tool.0).collect();
    tools.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
    let described = tools.into_iter().map(|tool| {
        HandlerDescriptor::tool(
            tool.name.as_str(),
            &tool.description,
            tool.parameters.clone(),
        )
    });
    log.recorder.handlers(described.collect());
}

/// Describes each model an agent connects to in the log's header.
fn describe_model(
    connected: On<Insert<Connection>>,
    agents: Query<&Connection>,
    log: Res<EffectLog>,
) {
    if let Ok(connection) = agents.get(connected.entity) {
        log.recorder
            .handlers(vec![connection.handler.0.descriptor()]);
    }
}

/// Writes the effects resolved by the end of the frame. A failure is shown
/// once.
fn write_effects(
    mut log: ResMut<EffectLog>,
    mut failed: Local<bool>,
    mut notices: MessageWriter<Notice>,
) {
    // Taking copies the header, tools' schemas and all, every frame.
    if log.recorder.resolved() == 0 {
        return;
    }
    let EffectLog { recorder, writer } = &mut *log;
    if let Err(failure) = writer.append(&recorder.take())
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

use std::time::{Duration, Instant};

use bevy_app::TaskPoolPlugin;
use rig_core::effect::{Answered as EffectOutcome, EffectId};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::{ToolFunction, ToolName};
use rig_core::tool::ChangeKind;
use rig_tools::Write;

use super::{Preview, PreviewPlugin, Previewed};
use rig_harness::per_call::AppPerCallExt;
use rig_harness::prelude::*;

/// A gate that asks for a preview of every call, and refuses each once it
/// is previewed, so none runs.
fn preview_then_refuse(
    mut calls: Query<(Entity, Option<&Previewed>, &mut Decision), (ToolCalls, ReadyToStart)>,
    mut commands: Commands,
) {
    for (call, previewed, mut decision) in &mut calls {
        match previewed {
            None => {
                commands.entity(call).insert(Preview);
            }
            Some(_) => {
                let refused = ErrorReport::new(ErrorKind::Denied, "refused");
                decision.decide(Decider::plugin("test"), Err::<EffectOutcome, _>(refused));
            }
        }
    }
}

#[test]
fn a_previewed_write_lists_the_file_it_would_create_and_leaves_the_disk_alone() {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("preview-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, PreviewPlugin))
        .insert_resource(ProjectRoot(dir.clone()))
        .add_tool_per_call(Write::default(), ToolOptions::default())
        .add_systems(AgentPass, preview_then_refuse.in_set(Stage::Gate));
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    let args = serde_json::json!({ "path": "new.txt", "content": "one\ntwo\n" });
    let Ok(name) = ToolName::new("write") else {
        return;
    };
    let call = ToolCall::from_wire("w1", ToolFunction::new(name, args));
    let parent = Effect::with_parent_id(Some(EffectId::from_raw(1)));
    let call = world
        .spawn((ToolRequest { call }, parent, ForAgent(agent)))
        .id();
    let started = Instant::now();
    while app.world().get::<Outcome>(call).is_none() && started.elapsed() < Duration::from_secs(5) {
        app.update();
        std::thread::sleep(Duration::from_millis(5));
    }
    let previewed = app.world().get::<Previewed>(call).map(|it| it.0.clone());
    let changes = previewed.clone().and_then(Result::ok).map(|changes| {
        changes
            .into_iter()
            .map(|change| (change.path, change.kind, change.added))
            .collect::<Vec<_>>()
    });
    let path = dir.join("new.txt").to_string_lossy().into_owned();
    assert_eq!(
        changes,
        Some(vec![(path, ChangeKind::Created, 2)]),
        "{previewed:?}"
    );
    assert!(!dir.join("new.txt").exists());
    std::fs::remove_dir_all(&dir).ok();
}

//! The `assets` feature: the agent's preamble is a Markdown file and its
//! tools a JSON file, loaded by `bevy_asset` (from memory here; from a
//! directory with the default source). A handle on the agent becomes its
//! `Preamble` and `Grant`s the tick the asset loads; the run is spawned
//! once both applied.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::type_complexity,
    reason = "an example: user code, thirty lines, a mock behind it"
)]

mod support;

use bevy_app::{Startup, Update};
use bevy_asset::{
    AssetApp, AssetPlugin, AssetServer,
    io::{
        AssetSourceBuilder, AssetSourceId,
        memory::{Dir, MemoryAssetReader},
    },
};
use bevy_ecs::prelude::*;
use rig_core::message::AssistantContent;
use rig_ecs::{
    agent::{Grant, Run},
    assets::{Applied, AssetsPlugin, Prompt, PromptHandle, ToolDefinitions, ToolsHandle},
    bus::Handlers,
    systems::spawn_run,
};

fn main() {
    let dir = Dir::default();
    dir.insert_asset_text(
        std::path::Path::new("agent.md"),
        "You are a calculator here to help the user perform arithmetic operations.\n",
    );
    dir.insert_asset_text(
        std::path::Path::new("agent.tools.json"),
        r#"[{"name": "subtract", "description": "Subtract y from x", "parameters": {"type": "object"}}]"#,
    );
    let mut app = support::app();
    app.register_asset_source(
        AssetSourceId::Default,
        AssetSourceBuilder::new(move || Box::new(MemoryAssetReader { root: dir.clone() })),
    )
    .add_plugins((
        AssetPlugin {
            watch_for_changes_override: Some(false),
            use_asset_processor_override: Some(false),
            ..Default::default()
        },
        AssetsPlugin,
    ))
    .add_systems(Startup, ask)
    .add_systems(Update, start_when_applied)
    .add_observer(support::print_the_answer_and_exit)
    .run();
}

fn ask(mut handlers: Handlers, mut commands: Commands, server: Res<AssetServer>) {
    let model = support::Scripted::new(vec![
        vec![support::call(
            "subtract",
            serde_json::json!({"x": 2, "y": 5}),
        )],
        vec![AssistantContent::text("-3")],
    ]);
    let tools = vec![support::add(), support::subtract()];
    let (model, _) = support::register(&mut handlers, model, tools);
    let agent = support::agent(&mut commands, model, "", 2);
    commands.entity(agent).insert((
        PromptHandle(server.load("agent.md")),
        ToolsHandle(server.load("agent.tools.json")),
    ));
}

/// Once both assets applied, one run — the granted tools counted from
/// the agent's `Grant` children.
fn start_when_applied(
    agents: Query<Entity, (With<Applied<Prompt>>, With<Applied<ToolDefinitions>>)>,
    grants: Query<&ChildOf, With<Grant>>,
    runs: Query<(), With<Run>>,
    mut commands: Commands,
) {
    if !runs.is_empty() {
        return;
    }
    for agent in &agents {
        let granted = grants.iter().filter(|link| link.parent() == agent).count();
        println!("granted {granted} tool(s) from agent.tools.json");
        commands.queue(move |world: &mut World| {
            spawn_run(world, agent, &[], "Calculate 2 - 5.", false, None);
        });
    }
}

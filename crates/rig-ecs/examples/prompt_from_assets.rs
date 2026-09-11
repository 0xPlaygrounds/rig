//! The `assets` feature: the agent's preamble is a Markdown file and its
//! tools a JSON file, loaded by `bevy_asset` (from memory here; from a
//! directory with the default source). A handle on the agent becomes its
//! `Preamble` and `Grant`s the tick the asset loads; the run is spawned
//! once both applied.

mod support;

use bevy_app::{App, AppExit, ScheduleRunnerPlugin, Update};
use bevy_asset::{
    AssetApp, AssetPlugin, AssetServer,
    io::{
        AssetSourceBuilder, AssetSourceId,
        memory::{Dir, MemoryAssetReader},
    },
};
use bevy_ecs::prelude::*;
use rig_core::{message::AssistantContent, serve::ServingPolicy};
use rig_ecs::{
    agent::Grant,
    assets::{
        Applied, AssetsPlugin, AssetsSet, Prompt as PromptAsset, PromptHandle, ToolDefinitions,
        ToolsHandle,
    },
    bus::{Handlers, run_to_quiescence},
    commands::{Agent, CommandFailures, Prompt, RigCommands, install},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Dir::default();
    dir.insert_asset_text(
        std::path::Path::new("agent.md"),
        "You are a calculator here to help the user perform arithmetic operations.\n",
    );
    dir.insert_asset_text(
        std::path::Path::new("agent.tools.json"),
        r#"[{"name": "subtract", "description": "Subtract y from x", "parameters": {"type": "object"}}]"#,
    );
    let mut app = App::new();
    install(app.world_mut(), ServingPolicy::default())?;
    app.add_plugins(ScheduleRunnerPlugin::default());
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
    .add_systems(
        Update,
        (
            start_when_applied.after(AssetsSet),
            report_command_failures,
            run_to_quiescence,
        ),
    )
    .configure_sets(Update, AssetsSet.before(run_to_quiescence))
    .add_observer(support::print_the_answer_and_exit)
    .add_observer(support::exit_when_failed);

    let model = support::Scripted::new(vec![
        vec![support::call(
            "subtract",
            serde_json::json!({"x": 2, "y": 5}),
        )],
        vec![AssistantContent::text("-3")],
    ]);
    let world = app.world_mut();
    let model = Handlers::register_in(world, support::MODEL, model)?;
    Handlers::register_in(world, "demo/add", support::add())?;
    Handlers::register_in(world, "demo/subtract", support::subtract())?;
    let agent = Agent::new(model).max_turns(2).spawn(world)?;
    let server = world.resource::<AssetServer>();
    let handles = (
        PromptHandle(server.load("agent.md")),
        ToolsHandle(server.load("agent.tools.json")),
        WaitingForAssets,
    );
    world.entity_mut(agent).insert(handles);

    if app.run().is_success() {
        Ok(())
    } else {
        Err(std::io::Error::other("the asset example failed").into())
    }
}

#[derive(Component)]
struct WaitingForAssets;

type ReadyToSubmit = (
    With<WaitingForAssets>,
    With<Applied<PromptAsset>>,
    With<Applied<ToolDefinitions>>,
);

/// Once both assets applied, one run — the granted tools counted from
/// the agent's `Grant` children.
fn start_when_applied(
    agents: Query<(Entity, Option<&Children>), ReadyToSubmit>,
    grants: Query<(), With<Grant>>,
    mut commands: Commands,
) {
    for (agent, children) in &agents {
        let granted = children
            .into_iter()
            .flat_map(|children| children.iter())
            .filter(|child| grants.contains(*child))
            .count();
        println!("granted {granted} tool(s) from agent.tools.json");
        commands.prompt(Prompt::new(agent, "Calculate 2 - 5."));
        commands.entity(agent).remove::<WaitingForAssets>();
    }
}

// Deferred operations report application-time failures through this resource.
fn report_command_failures(
    mut failures: ResMut<CommandFailures>,
    mut exit: MessageWriter<AppExit>,
) {
    for failure in failures.drain() {
        eprintln!("request {:?} failed: {}", failure.entity, failure.error);
        exit.write(AppExit::error());
    }
}

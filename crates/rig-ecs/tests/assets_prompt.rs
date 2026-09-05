//! The `assets` feature: a prompt file becomes the agent's preamble, a
//! tool-definition file its grants, in file order, to the bound handlers
//! the definitions name; a definition no handler serves is not granted.
//!
//! | claim | test |
//! |---|---|
//! | a loaded prompt is the preamble; loaded definitions are grants in file order, unserved names skipped; the run sees both | `a_prompt_and_tool_definitions_become_the_preamble_and_the_grants` |

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing
)]

mod run_support;

use bevy_app::App;
use bevy_asset::{
    AssetApp, AssetPlugin, AssetServer,
    io::{
        AssetSourceBuilder, AssetSourceId,
        memory::{Dir, MemoryAssetReader},
    },
};
use bevy_ecs::{prelude::*, schedule::LogLevel};
use rig_core::serve::ServingPolicy;
use rig_ecs::{
    agent::{Grant, Grants, Order, Preamble, Settled},
    assets::{Applied, AssetsPlugin, Prompt, PromptHandle, ToolDefinitions, ToolsHandle},
    bus::BusPlugin,
    systems::{AgentPlugin, spawn_run},
};
use run_support::*;

#[test]
fn a_prompt_and_tool_definitions_become_the_preamble_and_the_grants() {
    let dir = Dir::default();
    dir.insert_asset_text(std::path::Path::new("agent.md"), "Be brief.\n");
    dir.insert_asset_text(
        std::path::Path::new("agent.tools.json"),
        r#"[
            {"name": "unknown", "description": "nothing serves it", "parameters": {}},
            {"name": "add", "description": "adds", "parameters": {"type": "object"}}
        ]"#,
    );
    let mut app = App::new();
    app.register_asset_source(
        AssetSourceId::Default,
        AssetSourceBuilder::new(move || Box::new(MemoryAssetReader { root: dir.clone() })),
    )
    .add_plugins((
        BusPlugin::with_policy(ServingPolicy::default()).ambiguity_detection(LogLevel::Error),
        AgentPlugin::default(),
        AssetPlugin {
            watch_for_changes_override: Some(false),
            use_asset_processor_override: Some(false),
            ..Default::default()
        },
        AssetsPlugin,
    ));
    app.finish();
    app.cleanup();
    let (model, requests) = Capturing::new("t/model:default", "ok");
    let model = register(&mut app, "t/model:default", model);
    let add = register(&mut app, "t/tool:add#0", Adder::new("t/tool:add#0"));
    let agent = spawn_agent(app.world_mut(), "t", model);
    let server = app.world().resource::<AssetServer>().clone();
    app.world_mut().entity_mut(agent).insert((
        Preamble(None),
        PromptHandle(server.load("agent.md")),
        ToolsHandle(server.load("agent.tools.json")),
    ));
    tick_until(&mut app, "the assets applied", |world| {
        world.get::<Applied<Prompt>>(agent).is_some()
            && world.get::<Applied<ToolDefinitions>>(agent).is_some()
    });
    let world = app.world_mut();
    assert_eq!(
        world
            .get::<Preamble>(agent)
            .and_then(|p| p.0.clone())
            .as_deref(),
        Some("Be brief.")
    );
    // `unknown` is skipped; `add` is granted to the handler describing it.
    let mut grants: Vec<(u64, Entity)> = world
        .query::<(&Grant, &Order, &ChildOf)>()
        .iter(world)
        .filter(|(_, _, child_of)| child_of.parent() == agent)
        .map(|(grant, order, _)| (order.0, grant.0))
        .collect();
    grants.sort_unstable();
    assert_eq!(grants.len(), 1, "{grants:?}");
    assert_eq!(grants[0].1, add);
    // The relationship's target is the tool: it lists the one grant.
    assert_eq!(world.get::<Grants>(add).map(|g| g.len()), Some(1));
    let run = spawn_run(world, agent, &[], "go", false, None);
    tick_until(&mut app, "the run", |world| {
        world.get::<Settled>(run).is_some()
    });
    let requests = requests.lock().unwrap();
    assert_eq!(texts(&requests[0])[0], "system:Be brief.");
    assert_eq!(requests[0].tools.len(), 1);
}

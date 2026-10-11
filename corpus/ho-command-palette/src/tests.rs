use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ToolAdapter;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{app_in, first_agent, scratch_home};
use rig_tui::{KeyBinding, KeyPressed, TuiInput};

use super::*;
use crate::tui::Palette;

/// Set by `/zebra-stripes`.
#[derive(Resource, Default)]
struct Striped(bool);

fn stripe(_: In<CommandArgs>, mut striped: ResMut<Striped>) {
    striped.0 = true;
}

/// A tool found while the app runs.
struct ZzRuntime;

impl PortableTool for ZzRuntime {
    const NAME: &'static str = "zz_runtime";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "A tool registered while the agent runs.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }

    async fn call(&self, _: serde_json::Value) -> Result<String, ToolExecutionError> {
        Ok("ran".to_owned())
    }
}

/// The kernel, the palette and the test's commands on a session of `home`,
/// after the first frame, with its agent.
fn start(home: &Home) -> (App, Entity) {
    let (mut app, _wakes) = app_in(home, &SessionId::generate());
    app.add_plugins((AgentPlugin, JournalPlugin, CommandPalettePlugin))
        .init_resource::<Striped>()
        .add_command("zebra-stripes", "Paint the stripes", stripe)
        .add_command("gone", "Gone soon", |_: In<CommandArgs>| {});
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    (app, agent)
}

/// Presses the keys of the binding labelled `label` with `agent` shown.
fn press(app: &mut App, label: &str, agent: Entity) {
    let world = app.world_mut();
    let mut bindings = world.query::<(Entity, &KeyBinding)>();
    let found: Vec<Entity> = bindings
        .iter(world)
        .filter(|(_, binding)| binding.label == label)
        .map(|(entity, _)| entity)
        .collect();
    assert_eq!(found.len(), 1, "{label}");
    for entity in found {
        world.trigger(KeyPressed {
            entity,
            agent: Some(agent),
            released: false,
        });
    }
    app.update();
}

/// Types `filter` in the open palette.
fn type_filter(app: &mut App, filter: &str) -> Vec<String> {
    app.world_mut().resource_mut::<TuiInput>().set(filter);
    app.update();
    let palette = app.world().resource::<Palette>();
    palette
        .rows
        .iter()
        .map(|entry| entry.label.clone())
        .collect()
}

fn first(world: &mut World, filter: &str) -> Option<String> {
    ranked(world, filter)
        .into_iter()
        .next()
        .map(|entry| entry.label)
}

#[test]
fn ranks_live_entries_runs_a_pick_and_lists_it_first_next_time() {
    let home = scratch_home("command-palette");
    let (mut app, agent) = start(&home);
    app.world_mut().resource_mut::<TuiInput>().set("my draft");

    // Opened, the palette ranks what exists whenever its filter changes.
    press(&mut app, "Command palette", agent);
    assert_eq!(app.world().resource::<TuiInput>().text(), "");
    let shown = type_filter(&mut app, "zst");
    assert_eq!(
        shown.first().map(String::as_str),
        Some("/zebra-stripes"),
        "{shown:?}"
    );

    // A tool registered and a command disabled while it is open.
    let tool = ZzRuntime;
    let name = ToolName::new(<ZzRuntime as PortableTool>::NAME);
    assert!(name.is_ok());
    let Ok(name) = name else { return };
    let definition = ToolDefinition::new(
        name,
        PortableTool::description(&tool),
        PortableTool::parameters(&tool),
    );
    let handler = ErasedHandler::new(ToolAdapter::new(tool));
    let world = app.world_mut();
    world.spawn_tool(definition, handler, ToolOptions::default());
    let mut commands = world.query::<(Entity, &Name, &SlashCommand)>();
    let gone = commands
        .iter(world)
        .find(|(_, name, _)| name.as_str() == "/gone")
        .map(|(entity, ..)| entity);
    assert!(gone.is_some());
    if let Some(gone) = gone {
        world.entity_mut(gone).insert(Disabled);
    }
    let shown = type_filter(&mut app, "zz_r");
    assert!(shown.contains(&"zz_runtime".to_owned()), "{shown:?}");
    let everything = ranked(app.world_mut(), "");
    assert!(everything.iter().all(|entry| entry.label != "/gone"));
    assert!(
        ranked(app.world_mut(), "gone")
            .iter()
            .all(|entry| entry.label != "/gone")
    );

    // Enter runs the first row, and gives the draft back.
    type_filter(&mut app, "zst");
    press(&mut app, "Pick the palette's entry", agent);
    assert!(app.world().resource::<Striped>().0);
    assert_eq!(app.world().resource::<TuiInput>().text(), "my draft");
    assert_eq!(
        first(app.world_mut(), "").as_deref(),
        Some("/zebra-stripes")
    );

    // The pick is kept in `RIG_HOME`: the next session lists it first too.
    drop(app);
    let (mut app, _) = start(&home);
    assert_eq!(
        first(app.world_mut(), "").as_deref(),
        Some("/zebra-stripes")
    );
}

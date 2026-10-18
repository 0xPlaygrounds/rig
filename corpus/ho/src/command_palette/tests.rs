use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ToolAdapter;
use rig_harness::harness_protocol::SessionId;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{app_in, first_agent, scratch_home};
use rig_tui::crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use rig_tui::{Editor, KeyBinding, KeyFocus, KeyInput, KeyPressed, MainEditor, PendingInput};

use super::*;
use crate::command_palette::tui::Palette;

/// Set by `/zebra-stripes`.
#[derive(Resource, Default)]
struct Striped(bool);

fn stripe(_: In<CommandArgs>, mut striped: ResMut<Striped>) {
    striped.0 = true;
}

/// A tool found while the app runs.
struct ZzRuntime;

impl Tool for ZzRuntime {
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

    async fn call(
        &self,
        _: &mut ToolContext,
        _: serde_json::Value,
    ) -> Result<String, ToolExecutionError> {
        Ok("ran".to_owned())
    }
}

/// The kernel, both halves of the palette and the test's commands on a session of `home`,
/// after the first frame, with its agent shown.
fn start(home: &Home) -> (App, Entity) {
    let (mut app, _wakes) = app_in(home, &SessionId::generate());
    app.add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        PickPlugin,
        CommandPalettePlugin,
        CommandPaletteView,
    ))
    .init_resource::<Striped>()
    .add_command("zebra-stripes", "Paint the stripes", stripe)
    .add_command("gone", "Gone soon", |_: In<CommandArgs>| {});
    // The app is its own view: the agent's catalog reaches the palette as
    // its event.
    rig_harness_test_support::view::loopback(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    (app, agent)
}

/// Presses Ctrl-P, the palette's binding, with `agent` shown.
fn open_palette(app: &mut App, agent: Entity) {
    let world = app.world_mut();
    let mut bindings = world.query::<(Entity, &KeyBinding)>();
    let found: Vec<Entity> = bindings
        .iter(world)
        .filter(|(_, binding)| binding.label == "Command palette")
        .map(|(entity, _)| entity)
        .collect();
    assert_eq!(found.len(), 1);
    for entity in found {
        world.trigger(KeyPressed {
            entity,
            agent: Some(agent),
            released: false,
        });
    }
    app.update();
}

/// The terminal reports `keys` in one batch; the palette's rows after it.
fn type_keys(app: &mut App, keys: &[KeyCode]) -> Vec<String> {
    let mut events = app.world_mut().resource_mut::<PendingInput>();
    for &code in keys {
        let key = KeyEvent::new(code, KeyModifiers::NONE);
        events.0.push(KeyInput::Event(Event::Key(key)));
    }
    app.update();
    let world = app.world_mut();
    let mut open = world.query::<&Palette>();
    open.iter(world)
        .flat_map(|palette| palette.rows.iter().map(|entry| entry.label.clone()))
        .collect()
}

/// `text` as typed keys.
fn chars(text: &str) -> Vec<KeyCode> {
    text.chars().map(KeyCode::Char).collect()
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
    let world = app.world_mut();
    let mut main = world.query_filtered::<&mut Editor, With<MainEditor>>();
    let drafted = main
        .single_mut(world)
        .map(|mut input| input.set("my draft"));
    assert!(drafted.is_ok(), "the default screen has a main editor");

    // Opened, the palette has the keys and ranks what exists as its filter
    // changes; the draft stays in the input.
    open_palette(&mut app, agent);
    assert!(app.world().resource::<KeyFocus>().0.is_some());
    let shown = type_keys(&mut app, &chars("zst"));
    assert_eq!(
        shown.first().map(String::as_str),
        Some("/zebra-stripes"),
        "{shown:?}"
    );

    // A tool registered and a command disabled while it is open.
    let tool = ZzRuntime;
    let name = ToolName::new(<ZzRuntime as Tool>::NAME);
    assert!(name.is_ok());
    let Ok(name) = name else { return };
    let definition = ToolDefinition::new(name, Tool::description(&tool), Tool::parameters(&tool));
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
    // The agent's catalog takes the tool at the end of a frame.
    app.update();
    let mut keys = vec![KeyCode::Backspace; 3];
    keys.extend(chars("zz_r"));
    let shown = type_keys(&mut app, &keys);
    assert!(shown.contains(&"zz_runtime".to_owned()), "{shown:?}");
    assert!(
        ranked(app.world_mut(), "")
            .iter()
            .chain(&ranked(app.world_mut(), "gone"))
            .all(|entry| entry.label != "/gone")
    );

    // Typed in one batch, Enter picks for the filter typed before it: it
    // runs the first row and leaves the draft as it was.
    let mut keys = vec![KeyCode::Backspace; 4];
    keys.extend(chars("zst"));
    keys.push(KeyCode::Enter);
    assert!(type_keys(&mut app, &keys).is_empty());
    app.update();
    assert!(app.world().resource::<Striped>().0);
    let world = app.world_mut();
    let mut main = world.query_filtered::<&Editor, With<MainEditor>>();
    let draft = main.single(world).map(Editor::text);
    assert_eq!(draft.ok(), Some("my draft"));
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

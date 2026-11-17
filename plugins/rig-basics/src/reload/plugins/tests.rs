use std::fs;

use rig_harness::harness_protocol::Invocation;
use rig_harness::host::launcher::Invoked;
use rig_harness::pick::{Highlighted, PickPreview};
use rig_harness::rig_ecs::inbox::Recalled;
use rig_harness::services::RigHome;
use rig_harness_test_support::scratch_home;

use super::*;

/// Every notice written so far, and why each refused line was.
#[derive(Resource, Default)]
struct Told(Vec<String>);

/// An app with `/plugins` in `home`, on a terminal or not, keeping its
/// notices, and an agent.
fn app(home: &rig_harness::harness_protocol::Home, terminal: bool) -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins((rig_harness::rig_ecs::AgentPlugin, PluginsPlugin))
        .insert_resource(RigHome(home.clone()))
        .insert_resource(Invoked {
            args: Invocation::default(),
            terminal,
        })
        .init_resource::<Told>()
        .add_systems(
            Last,
            |mut notices: MessageReader<Notice>,
             mut refused: MessageReader<Recalled>,
             mut told: ResMut<Told>| {
                told.0
                    .extend(notices.read().map(|notice| notice.text.clone()));
                told.0
                    .extend(refused.read().filter_map(|line| line.why.clone()));
            },
        );
    let agent = spawn_root(app.world_mut()).id();
    (app, agent)
}

/// Runs `/plugins` for `agent` and the frame after it.
fn open(app: &mut App, agent: Entity) {
    let world = app.world_mut();
    world.trigger(RunCommand::new(agent, "plugins".to_owned()));
    world.flush();
    app.update();
}

/// The text of the preview on `picker`.
fn preview(app: &App, picker: Entity) -> String {
    match app.world().get::<PickPreview>(picker) {
        Some(PickPreview::Text(text)) => text.clone(),
        _ => String::new(),
    }
}

#[test]
fn plugins_opens_from_settings_with_short_names_and_a_preview_from_the_start() {
    let home = scratch_home("plugins-opens");
    fs::create_dir_all(home.root()).ok();
    fs::write(
        home.settings(),
        "[plugins]\ndisable = [\"rig_basics::welcome::WelcomePlugin\"]\n",
    )
    .ok();
    let (mut app, agent) = app(&home, true);
    open(&mut app, agent);

    let world = app.world_mut();
    let pickers: Vec<(Entity, Picker)> = world
        .query::<(Entity, &Picker)>()
        .iter(world)
        .map(|(entity, picker)| (entity, picker.clone()))
        .collect();
    let told = &app.world().resource::<Told>().0;
    assert_eq!(pickers.len(), 1, "{told:?}");
    let Some((picker, shown)) = pickers.first() else {
        return;
    };
    let labels: Vec<&str> = shown.items.iter().map(|item| item.label.as_str()).collect();
    assert!(labels.contains(&"Terminal view"), "{labels:?}");
    assert!(
        labels.iter().all(|label| !label.contains("::")),
        "{labels:?}"
    );

    let first = preview(&app, *picker);
    // The first row is the first plugin of the first group.
    let first_built_in = BUILT_IN.iter().min_by_key(|built_in| built_in.group);
    let first_path = first_built_in.map_or("-", |built_in| built_in.plugin);
    assert!(first.contains(first_path), "{first:?} names {first_path}");
    let more = shown.items.first().and_then(|item| item.more.as_deref());
    assert!(
        more.is_some_and(|more| more.starts_with(first_path)),
        "typing the type path finds it: {more:?}"
    );

    let welcome = labels.iter().position(|label| *label == "Welcome");
    assert!(welcome.is_some(), "{labels:?}");
    let Some(welcome) = welcome else {
        return;
    };
    let checked = shown.items.get(welcome).map(|item| item.checked);
    assert_eq!(checked, Some(false), "off in settings, unticked");
    app.world_mut()
        .entity_mut(*picker)
        .insert(Highlighted(welcome));
    app.update();
    let text = preview(&app, *picker);
    assert!(
        text.contains("rig_basics::welcome::WelcomePlugin")
            && text.contains("The welcome at the top of the transcript")
            && text.contains("default on")
            && text.contains("off in settings"),
        "{text}"
    );

    // A filter that leaves no row leaves no plugin in the preview.
    app.world_mut().entity_mut(*picker).remove::<Highlighted>();
    app.update();
    assert_eq!(preview(&app, *picker), NONE_HIGHLIGHTED);
    app.world_mut()
        .entity_mut(*picker)
        .insert(Highlighted(welcome));
    app.update();
    assert_eq!(preview(&app, *picker), text);
}

/// The rows of the one `/plugins` picker in `app`.
fn picked_rows(app: &mut App) -> Vec<Row> {
    let world = app.world_mut();
    let mut pickers = world.query::<&PluginsPicker>();
    let rows = pickers.iter(world).next().map(|picker| picker.rows.clone());
    rows.unwrap_or_default()
}

#[test]
fn a_row_says_what_waits_for_the_next_reload_and_what_an_essential_takes_away() {
    let home = scratch_home("plugins-states");
    fs::create_dir_all(home.root()).ok();
    fs::write(
        home.settings(),
        "[plugins]\ndisable = [\"rig_basics::welcome::WelcomePlugin\"]\n\
         enable = [\"rig_basics::budget::BudgetPlugin\"]\n\
         [[plugins.extra]]\ncrate = \"viz\"\npath = \"viz\"\nplugin = \"viz::Charts\"\n\
         [[plugins.extra]]\ncrate = \"viz\"\npath = \"viz\"\nplugin = \"viz::Tables\"\n\
         [[plugins.extra]]\ncrate = \"viz\"\npath = \"viz\"\nplugin = \"viz::charts::Plugin\"\n\
         [[plugins.extra]]\ncrate = \"viz\"\npath = \"viz\"\nplugin = \"viz::tables::Plugin\"\n",
    )
    .ok();
    let (mut app, agent) = app(&home, true);
    // This build runs the terminal view and the welcome, which settings.toml
    // now turns off.
    for plugin in ["rig_tui::TuiPlugin", "rig_basics::welcome::WelcomePlugin"] {
        let source = PluginSource {
            package: String::new(),
            source: String::new(),
        };
        app.world_mut().spawn((Name::new(plugin), source));
    }
    open(&mut app, agent);
    let rows = picked_rows(&mut app);
    let row = |plugin: &str| rows.iter().find(|row| row.plugin == plugin);
    let shown = |plugin: &str| row(plugin).map(|row| (row.name.as_str(), row.state.as_str()));

    assert_eq!(
        shown("rig_basics::welcome::WelcomePlugin"),
        Some(("Welcome", "off · next reload"))
    );
    assert_eq!(
        shown("rig_basics::budget::BudgetPlugin"),
        Some(("Budget", "on · next reload"))
    );
    assert_eq!(shown("rig_tui::TuiPlugin"), Some(("Terminal view", "on")));
    let about = |plugin: &str| row(plugin).map_or("", |row| row.about.as_str());
    assert!(about("rig_basics::welcome::WelcomePlugin").contains("Runs until the next reload"));
    assert!(about("rig_basics::budget::BudgetPlugin").contains("Takes effect at the next reload"));
    assert!(
        about("rig_tui::TuiPlugin")
            .contains("Essential: turning it off takes away the terminal view"),
        "{}",
        about("rig_tui::TuiPlugin")
    );
    // Two extras of one crate are told apart.
    assert_eq!(
        shown("viz::Charts").map(|(name, _)| name),
        Some("viz Charts")
    );
    assert_eq!(
        shown("viz::Tables").map(|(name, _)| name),
        Some("viz Tables")
    );
    // Two of one crate with one type name are named by their type paths.
    assert_eq!(
        shown("viz::charts::Plugin").map(|(name, _)| name),
        Some("viz::charts::Plugin")
    );
    assert_eq!(
        shown("viz::tables::Plugin").map(|(name, _)| name),
        Some("viz::tables::Plugin")
    );
}

#[test]
fn an_unreadable_settings_file_is_named_with_what_to_do() {
    let home = scratch_home("plugins-unreadable");
    fs::create_dir_all(home.root()).ok();
    fs::write(
        home.settings(),
        "[plugins]\nenable = [\"nope::NopePlugin\"]\n",
    )
    .ok();
    let (mut app, agent) = app(&home, true);
    open(&mut app, agent);

    let world = app.world_mut();
    let pickers = world.query::<&Picker>().iter(world).count();
    assert_eq!(pickers, 0);
    let told = &app.world().resource::<Told>().0;
    let settings = home.settings().display().to_string();
    assert!(
        told.iter().any(|text| text.contains(&settings)
            && text.contains("nope::NopePlugin")
            && text.contains("rig plugin disable")),
        "{told:?}"
    );
}

#[test]
fn settings_that_are_not_toml_are_named_without_a_command_that_cannot_read_them() {
    let home = scratch_home("plugins-not-toml");
    fs::create_dir_all(home.root()).ok();
    fs::write(home.settings(), "[plugins\n").ok();
    let (mut app, agent) = app(&home, true);
    open(&mut app, agent);

    let told = &app.world().resource::<Told>().0;
    let settings = home.settings().display().to_string();
    assert!(
        told.iter().any(|text| text.contains(&settings)
            && text.contains("not TOML")
            && text.contains("Edit the file")
            && !text.contains("rig plugin disable")),
        "{told:?}"
    );
}

#[test]
fn every_row_has_a_name_of_its_own() {
    let home = scratch_home("plugins-names");
    fs::create_dir_all(home.root()).ok();
    let extras = [
        ("Welcome", "welcome::Plugin"),
        ("viz", "viz::charts::Plugin"),
        ("viz", "viz::tables::Plugin"),
        ("Plugin", "Other"),
        ("lone", "Plugin"),
        ("Model", "picker"),
        ("Model", "picker2"),
        ("picker", "x::picker"),
    ];
    let tables: String = (extras.iter())
        .map(|(krate, plugin)| {
            format!("[[plugins.extra]]\ncrate = \"{krate}\"\npath = \"{krate}\"\nplugin = \"{plugin}\"\n")
        })
        .collect();
    fs::write(home.settings(), tables).ok();
    let (mut app, agent) = app(&home, true);
    open(&mut app, agent);
    let rows = picked_rows(&mut app);
    assert_eq!(
        rows.len(),
        BUILT_IN.len() + extras.len(),
        "{:?}",
        app.world().resource::<Told>().0
    );
    let mut names: Vec<&str> = rows.iter().map(|row| row.name.as_str()).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(names.len(), before, "{names:?}");
}

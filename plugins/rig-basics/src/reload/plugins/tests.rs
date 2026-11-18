use std::fs;

use rig_harness::harness_protocol::Invocation;
use rig_harness::host::launcher::Invoked;
use rig_harness::pick::{Chosen, Highlighted, Pick, PickPreview, Ticked};
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
    app.add_plugins((
        rig_harness::rig_ecs::AgentPlugin,
        rig_harness::pick::PickPlugin,
        PluginsPlugin,
    ))
    .init_resource::<ReloadStatus>()
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
    let agent = spawn_root(app.world_mut(), ()).id();
    (app, agent)
}

/// Runs `/plugins` for `agent` and the frame after it.
fn open(app: &mut App, agent: Entity) {
    run(app, agent, "plugins");
}

/// Runs the command `line` for `agent` and the frame after it.
fn run(app: &mut App, agent: Entity, line: &str) {
    let world = app.world_mut();
    world.trigger(RunCommand::new(agent, line.to_owned()));
    world.flush();
    app.update();
}

/// The one picker open, if one is.
fn picker(app: &mut App) -> Option<(Entity, Picker)> {
    let world = app.world_mut();
    let mut pickers = world.query::<(Entity, &Picker)>();
    let open: Vec<(Entity, Picker)> = (pickers.iter(world))
        .map(|(entity, picker)| (entity, picker.clone()))
        .collect();
    match open.as_slice() {
        [one] => Some(one.clone()),
        _ => None,
    }
}

/// The position of the item labelled `label` in `picker`.
fn at(picker: &Picker, label: &str) -> Option<usize> {
    picker.items.iter().position(|item| item.label == label)
}

/// The detail of the item labelled `label` in `picker`.
fn detail(picker: &Picker, label: &str) -> String {
    let item = picker.items.iter().find(|item| item.label == label);
    item.and_then(|item| item.detail.clone())
        .unwrap_or_default()
}

/// Ticks the list as the view does, from what is on now: `on` ticked and
/// `off` unticked, then the frame after it.
fn tick(app: &mut App, list: Entity, on: &[&str], off: &[&str]) {
    let Some(shown) = app.world().get::<Picker>(list).cloned() else {
        return;
    };
    let ticked = (shown.items.iter().enumerate())
        .filter(|(_, item)| {
            on.contains(&item.label.as_str()) || item.checked && !off.contains(&item.label.as_str())
        })
        .map(|(index, _)| index)
        .collect();
    app.world_mut().entity_mut(list).insert(Ticked(ticked));
    app.update();
}

/// Closes or answers `entity` as a person, then the frame after it.
fn pick(app: &mut App, entity: Entity, chosen: Option<Chosen>) {
    let world = app.world_mut();
    world.trigger(Pick::person(entity, chosen));
    world.flush();
    app.update();
}

/// The settings file's text, empty when it is missing.
fn settings(home: &rig_harness::harness_protocol::Home) -> String {
    fs::read_to_string(home.settings()).unwrap_or_default()
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
    // The first row is rig's first plugin: its crate and area come first.
    let first_path = BUILT_IN.first().map_or("-", |built_in| built_in.plugin);
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
        ("Plugin", "other::Other"),
        ("lone", "lone::Plugin"),
        ("Model", "model::picker"),
        ("Model", "model::picker2"),
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

#[test]
fn enter_ticks_and_the_list_marks_and_counts_changes_against_the_settings() {
    let home = scratch_home("plugins-ticks");
    let (mut app, agent) = app(&home, true);
    open(&mut app, agent);
    let found = picker(&mut app);
    assert!(
        found.is_some(),
        "no picker: {:?}",
        app.world().resource::<Told>().0
    );
    let Some((list, _)) = found else {
        return;
    };
    #[cfg(feature = "tui")]
    {
        let keys = app.world().get::<rig_tui::ChoiceKeys>(list);
        let enter = (keys.iter().flat_map(|keys| keys.0.iter())).any(|(code, action)| {
            *code == rig_tui::crossterm::event::KeyCode::Enter
                && *action == rig_tui::ChoiceAction::Tick
        });
        assert!(enter, "Enter ticks in /plugins: {keys:?}");
    }

    tick(&mut app, list, &["Budget"], &["Welcome"]);
    let shown = app.world().get::<Picker>(list).cloned().unwrap_or_default();
    assert!(detail(&shown, "Welcome").contains("- off"), "{shown:#?}");
    assert!(detail(&shown, "Budget").contains("+ on"), "{shown:#?}");
    assert!(shown.body.contains("2 changes"), "{}", shown.body);
    assert!(shown.body.contains("Enter ticks"), "{}", shown.body);
    assert_eq!(settings(&home), "", "ticking writes nothing");

    // Back as it was: no change is counted.
    tick(&mut app, list, &["Welcome"], &["Budget"]);
    let shown = app.world().get::<Picker>(list).cloned().unwrap_or_default();
    assert!(!detail(&shown, "Welcome").contains("off"), "{shown:#?}");
    assert!(!shown.body.contains("change"), "{}", shown.body);
}

#[test]
fn esc_with_no_change_closes_and_asks_nothing() {
    let home = scratch_home("plugins-esc-unchanged");
    let (mut app, agent) = app(&home, true);
    open(&mut app, agent);
    let found = picker(&mut app);
    assert!(found.is_some(), "no picker");
    let Some((list, _)) = found else {
        return;
    };
    tick(&mut app, list, &[], &[]);
    pick(&mut app, list, None);

    assert!(picker(&mut app).is_none());
    assert_eq!(settings(&home), "");
    assert_eq!(app.world().resource::<Told>().0, Vec::<String>::new());
}

#[test]
fn esc_with_changes_asks_to_reload_now_or_later_naming_them() {
    let home = scratch_home("plugins-esc-changed");
    let (mut app, agent) = app(&home, true);
    open(&mut app, agent);
    let found = picker(&mut app);
    assert!(found.is_some(), "no picker");
    let Some((list, _)) = found else {
        return;
    };
    tick(&mut app, list, &["Budget"], &["Welcome", "Plugins picker"]);
    pick(&mut app, list, None);

    let found = picker(&mut app);
    assert!(found.is_some(), "no question");
    let Some((question, asked)) = found else {
        return;
    };
    let labels: Vec<&str> = asked.items.iter().map(|item| item.label.as_str()).collect();
    assert!(
        asked.body.contains("Turn on: Budget")
            && asked.body.contains("Turn off: ")
            && asked.body.contains("Welcome")
            && asked.body.contains("/plugins itself"),
        "{}",
        asked.body
    );
    // Without the launcher a reload cannot run, so it is not offered.
    assert_eq!(labels, ["Later, at the next start", "Keep editing"]);
    assert!(asked.body.contains("`rig`"), "{}", asked.body);
    assert_eq!(settings(&home), "", "asking writes nothing");

    // Keep editing: the list again, with the ticks kept.
    let keep = at(&asked, "Keep editing");
    pick(&mut app, question, keep.map(Chosen::item));
    let found = picker(&mut app);
    assert!(found.is_some(), "no list");
    let Some((list, again)) = found else {
        return;
    };
    let checked = |label: &str| {
        (again.items.iter().find(|item| item.label == label)).map(|item| item.checked)
    };
    assert_eq!(checked("Budget"), Some(true));
    assert_eq!(checked("Welcome"), Some(false));
    assert!(detail(&again, "Welcome").contains("- off"), "{again:#?}");
    assert_eq!(settings(&home), "");

    // Esc asks again; Later writes the changes and does not reload.
    tick(&mut app, list, &[], &["Plugins picker"]);
    let ticked = app.world().get::<Ticked>(list).cloned();
    assert!(ticked.is_some());
    pick(&mut app, list, None);
    let found = picker(&mut app);
    assert!(found.is_some(), "no question");
    let Some((question, asked)) = found else {
        return;
    };
    let later = at(&asked, "Later, at the next start");
    pick(&mut app, question, later.map(Chosen::item));
    assert!(picker(&mut app).is_none());
    let written = settings(&home);
    assert!(
        written.contains("rig_basics::budget::BudgetPlugin")
            && written.contains("rig_basics::welcome::WelcomePlugin")
            && written.contains("rig_basics::reload::plugins::PluginsPlugin"),
        "{written}"
    );
    let chosen = config::chosen(&home).unwrap_or_default();
    assert!(
        chosen
            .on
            .iter()
            .any(|on| on == "rig_basics::budget::BudgetPlugin")
    );
    assert!(
        !chosen
            .on
            .iter()
            .any(|on| on == "rig_basics::welcome::WelcomePlugin")
    );
    assert_eq!(*app.world().resource::<ReloadStatus>(), ReloadStatus::Idle);
}

#[test]
fn plugins_defaults_asks_to_go_back_to_rigs_defaults() {
    let home = scratch_home("plugins-defaults");
    fs::create_dir_all(home.root()).ok();
    let changed = "[plugins]\ndisable = [\"rig_basics::welcome::WelcomePlugin\"]\n\
                   enable = [\"rig_basics::budget::BudgetPlugin\"]\n";
    fs::write(home.settings(), changed).ok();
    let (mut app, agent) = app(&home, true);
    run(&mut app, agent, "plugins defaults");

    let found = picker(&mut app);
    assert!(
        found.is_some(),
        "no question: {:?}",
        app.world().resource::<Told>().0
    );
    let Some((question, asked)) = found else {
        return;
    };
    assert!(
        asked.body.contains("Turn on: Welcome") && asked.body.contains("Turn off: Budget"),
        "{}",
        asked.body
    );
    let later = at(&asked, "Later, at the next start");
    pick(&mut app, question, later.map(Chosen::item));
    let back = config::chosen(&home).unwrap_or_default();
    let defaults: Vec<&str> = (BUILT_IN.iter())
        .filter(|built_in| built_in.on && built_in.available())
        .map(|built_in| built_in.plugin)
        .collect();
    assert_eq!(back.on, defaults);

    // At the defaults already, it says so and asks nothing.
    run(&mut app, agent, "plugins defaults");
    assert!(picker(&mut app).is_none());
    let told = &app.world().resource::<Told>().0;
    assert!(
        told.iter().any(|text| text.contains("defaults")),
        "{told:?}"
    );
}

#[test]
fn the_question_says_why_it_cannot_reload_and_esc_goes_back_to_the_list() {
    let home = scratch_home("plugins-no-reload");
    let (mut app, agent) = app(&home, true);
    app.world_mut().remove_resource::<ReloadStatus>();
    open(&mut app, agent);
    let found = picker(&mut app);
    assert!(found.is_some(), "no list");
    let Some((list, _)) = found else {
        return;
    };
    tick(&mut app, list, &["Budget"], &[]);
    pick(&mut app, list, None);
    let asked = picker(&mut app);
    assert!(asked.is_some(), "no question");
    let Some((question, shown)) = asked else {
        return;
    };
    assert!(shown.body.contains("Reload plugin"), "{}", shown.body);

    // Esc with a filter typed that matches no answer, so nothing is
    // highlighted: back to the list with the ticks, nothing written.
    pick(&mut app, question, None);
    let again = picker(&mut app);
    assert_eq!(
        again.as_ref().map(|(_, shown)| shown.title.as_str()),
        Some("Plugins")
    );
    let ticked = (again.iter().flat_map(|(_, shown)| shown.items.iter()))
        .find(|item| item.label == "Budget")
        .map(|item| item.checked);
    assert_eq!(ticked, Some(true));
    assert_eq!(settings(&home), "");

    // Later says when the changes apply, naming no /reload that cannot run.
    let Some((list, _)) = again else {
        return;
    };
    // The reopened list keeps Budget ticked.
    tick(&mut app, list, &[], &[]);
    pick(&mut app, list, None);
    let Some((question, shown)) = picker(&mut app) else {
        return;
    };
    let later = at(&shown, "Later, at the next start");
    pick(&mut app, question, later.map(Chosen::item));
    let told = &app.world().resource::<Told>().0;
    let said = told.iter().find(|text| text.contains("turned on Budget"));
    assert!(
        said.is_some_and(|text| text.contains("starts again") && !text.contains("/reload")),
        "{told:?}"
    );
}

#[test]
fn a_plugin_of_an_off_branch_keeps_its_own_tick_and_says_its_branch_is_off() {
    let home = scratch_home("plugins-branch-off");
    fs::create_dir_all(home.root()).ok();
    let off = "[plugins]\noff = [\"rig-basics/sessions\"]\n";
    fs::write(home.settings(), off).ok();
    let (mut app, agent) = app(&home, true);
    open(&mut app, agent);
    let found = picker(&mut app);
    assert!(found.is_some(), "{:?}", app.world().resource::<Told>().0);
    let Some((list, shown)) = found else {
        return;
    };
    let checked = |label: &str| {
        (shown.items.iter().find(|item| item.label == label)).map(|item| item.checked)
    };
    assert_eq!(checked("Sessions"), Some(true), "its own choice is on");
    assert!(
        detail(&shown, "Sessions").starts_with("off with its branch"),
        "{shown:#?}"
    );
    assert_eq!(detail(&shown, "Skills"), "on");

    // Enter changes only that plugin's own choice; the branch stays off.
    tick(&mut app, list, &[], &["Forks and rewind"]);
    pick(&mut app, list, None);
    let found = picker(&mut app);
    assert!(found.is_some(), "no question");
    let Some((question, asked)) = found else {
        return;
    };
    let later = at(&asked, "Later, at the next start");
    pick(&mut app, question, later.map(Chosen::item));
    let chosen = config::chosen(&home).unwrap_or_default();
    assert_eq!(chosen.off, ["rig-basics/sessions"]);
    assert!(
        !(chosen.own.iter()).any(|plugin| plugin == "rig_basics::sessions::agents::AgentsPlugin")
    );
    assert!((chosen.own.iter()).any(|plugin| plugin == "rig_basics::sessions::SessionsPlugin"));
}

#[test]
fn plugins_defaults_turns_off_branches_back_on() {
    let home = scratch_home("plugins-defaults-branch");
    fs::create_dir_all(home.root()).ok();
    fs::write(
        home.settings(),
        "[plugins]\noff = [\"rig-basics/sessions\"]\n",
    )
    .ok();
    let (mut app, agent) = app(&home, true);
    run(&mut app, agent, "plugins defaults");
    let found = picker(&mut app);
    assert!(found.is_some(), "{:?}", app.world().resource::<Told>().0);
    let Some((question, asked)) = found else {
        return;
    };
    assert!(asked.body.contains("rig-basics/sessions"), "{}", asked.body);
    let later = at(&asked, "Later, at the next start");
    pick(&mut app, question, later.map(Chosen::item));
    let back = config::chosen(&home).unwrap_or_default();
    assert!(back.off.is_empty(), "{}", settings(&home));
    let defaults: Vec<&str> = (BUILT_IN.iter())
        .filter(|built_in| built_in.on && built_in.available())
        .map(|built_in| built_in.plugin)
        .collect();
    assert_eq!(back.on, defaults);
}

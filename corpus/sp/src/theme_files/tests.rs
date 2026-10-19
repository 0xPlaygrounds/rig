use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::commands::CommandPlugin;
use rig_harness::pick::PickPlugin;
use rig_harness::watch::WatchPlugin;
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, calls, log_ended, press, reply, run_until, scratch_home,
    scripted_models, tui_view,
};
use rig_tui::ratatui::buffer::Cell;
use rig_tui::{ShowAgent, TuiFrame, TuiTestScreen};

use super::*;

/// The `fail` tool: every call fails.
fn fail(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("fail"),
            Outcome::refused("boom went the tool"),
        );
    }
}

/// An app at the terminal view with the plugin, its themes in `dir`, the
/// user's plugin data in `user`, and its agent, shown.
fn start(user: &MemoryStore, dir: &Path, no_color: bool) -> (App, Receiver<()>, Entity) {
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "fail", serde_json::json!({})),
        reply("All done."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let settings = format!("[themes]\ndir = '{}'", dir.display());
    let data = PluginData {
        session: Arc::new(MemoryStore::default()),
        project: Arc::new(MemoryStore::default()),
        user: Arc::new(user.clone()),
    };
    app.insert_resource(Settings::from_toml(&settings))
        .insert_resource(NoColor(no_color))
        .insert_resource(data)
        .insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            CommandPlugin,
            PickPlugin,
            WatchPlugin,
        ))
        .add_open_tool("fail", "Fail.", ToolOptions::default(), fail)
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .insert_resource(TuiTestScreen::new(100, 30));
    tui_view(&mut app);
    app.add_plugins(ThemeFilesPlugin);
    let agent = app
        .world_mut()
        .spawn((Agent, ModelChoice("test/model".to_owned())))
        .id();
    app.finish();
    app.cleanup();
    app.update();
    app.world_mut().trigger(ShowAgent { entity: agent });
    // The watch reports the files there as it is armed.
    run_until(&mut app, &wakes, |world| {
        !world.resource::<ThemeFiles>().palettes.is_empty()
    });
    app.update();
    (app, wakes, agent)
}

/// Runs `line` as a command typed for `agent`, and draws the result.
fn command(app: &mut App, agent: Entity, line: &str) {
    app.world_mut().trigger(RunCommand::new(agent, line));
    app.update();
    app.update();
}

/// The cells of the first `text` on the screen.
fn cells_of(app: &App, text: &str) -> Vec<Cell> {
    let screen = app.world().resource::<TuiTestScreen>();
    let rows = screen.rows();
    let Some((y, at)) = rows
        .iter()
        .enumerate()
        .find_map(|(y, row)| Some((y, row.find(text)?)))
    else {
        return Vec::new();
    };
    let x = rows
        .get(y)
        .and_then(|row| row.get(..at))
        .map_or(0, |before| before.chars().count());
    let (x, y) = (u16::try_from(x).unwrap_or(0), u16::try_from(y).unwrap_or(0));
    let width = u16::try_from(text.chars().count()).unwrap_or(0);
    (x..x + width)
        .filter_map(|x| screen.buffer().cell((x, y)).cloned())
        .collect()
}

/// The backgrounds of the status line's cells no overlay covers: the
/// bottom row of an editor that draws it.
fn status_backgrounds(app: &mut App) -> Vec<Color> {
    let world = app.world_mut();
    let mut lines = world.query_filtered::<(&PanelCanvas, Has<Editor>), With<StatusLine>>();
    let areas: Vec<_> = lines
        .iter(world)
        .map(|(canvas, editor)| match editor {
            true => canvas.area().rows().next_back().unwrap_or_default(),
            false => canvas.area(),
        })
        .collect();
    let frame = world.resource::<TuiFrame>().clone();
    let screen = world.resource::<TuiTestScreen>();
    areas
        .iter()
        .flat_map(|area| area.positions())
        .filter(|at| !frame.overlays.iter().any(|over| over.contains(*at)))
        .filter_map(|at| screen.buffer().cell(at).map(|cell| cell.bg))
        .collect()
}

/// Whether every uncovered status line cell has `color` behind it.
fn status_on(app: &mut App, color: Color) -> bool {
    let backgrounds = status_backgrounds(app);
    !backgrounds.is_empty() && backgrounds.iter().all(|bg| *bg == color)
}

/// Runs frames until `done` holds of the screen, or a few seconds passed.
fn wait_for(app: &mut App, done: impl Fn(&mut App) -> bool) -> bool {
    for _ in 0..200 {
        if done(app) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
        app.update();
    }
    done(app)
}

/// How many screen rows say `text`.
fn rows_saying(app: &App, text: &str) -> usize {
    let rows = app.world().resource::<TuiTestScreen>().rows();
    rows.iter().filter(|row| row.contains(text)).count()
}

#[test]
fn a_theme_file_colours_the_view_follows_edits_and_is_kept() {
    let home = scratch_home("sp-theme-files");
    let dir = home.root().join("themes");
    fs::create_dir_all(&dir).ok();
    let paper = dir.join("paper.toml");
    fs::write(&paper, "status_bg = \"#eee8d5\"\nuser = \"ansi:4\"\n").ok();
    let user = MemoryStore::default();
    let (mut app, wakes, agent) = start(&user, &dir, false);

    // One reply after one failing call, then `/theme paper`.
    app.world_mut()
        .trigger(Deliver::new(agent, "hi there", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    command(&mut app, agent, "theme paper");
    let paper_bg = Color::Rgb(0xee, 0xe8, 0xd5);
    assert!(
        status_on(&mut app, paper_bg),
        "{:?}",
        status_backgrounds(&mut app)
    );
    let typed = cells_of(&app, "hi there");
    assert!(
        !typed.is_empty() && typed.iter().all(|cell| cell.fg == Color::Blue),
        "{typed:?}"
    );
    // No `error` key: `dark`'s.
    let failed = cells_of(&app, "boom went the tool");
    assert!(
        !failed.is_empty() && failed.iter().all(|cell| cell.fg == Color::Red),
        "{failed:?}"
    );

    // Saved again: applied as the watch reports it.
    fs::write(&paper, "status_bg = \"#fdf6e3\"\nuser = \"ansi:4\"\n").ok();
    let edited_bg = Color::Rgb(0xfd, 0xf6, 0xe3);
    assert!(wait_for(&mut app, |app| status_on(app, edited_bg)));

    // A bad colour keeps the colours and says so once, naming the key.
    fs::write(&paper, "status_bg = \"#zz\"\n").ok();
    assert!(wait_for(&mut app, |app| rows_saying(app, "status_bg") > 0));
    std::thread::sleep(Duration::from_millis(300));
    app.update();
    app.update();
    assert_eq!(rows_saying(&app, "status_bg"), 1);
    assert!(status_on(&mut app, edited_bg));

    // The picker previews `light` and puts `paper` back on Esc.
    command(&mut app, agent, "theme");
    press(&mut app, "up");
    app.update();
    assert_eq!(app.world().resource::<Theme>().name(), "light");
    assert!(
        status_on(&mut app, Color::Gray),
        "{:?}",
        status_backgrounds(&mut app)
    );
    press(&mut app, "esc");
    app.update();
    assert_eq!(app.world().resource::<Theme>().name(), "paper");
    assert!(status_on(&mut app, edited_bg));
    drop(app);

    // A restart keeps `paper`, with its file's last good colours.
    fs::write(&paper, "status_bg = \"#fdf6e3\"\nuser = \"ansi:4\"\n").ok();
    let (mut app, _, _) = start(&user, &dir, false);
    assert_eq!(app.world().resource::<Theme>().name(), "paper");
    assert!(status_on(&mut app, edited_bg));
    drop(app);

    // With `NO_COLOR`, no cell has a colour, an error notice's neither.
    let (mut app, _, agent) = start(&user, &dir, true);
    command(&mut app, agent, "theme nothing-like-it");
    assert!(rows_saying(&app, "nothing-like-it") > 0);
    let screen = app.world().resource::<TuiTestScreen>();
    let area = screen.buffer().area;
    let coloured = area.positions().filter(|at| {
        let cell = screen.buffer().cell(*at);
        cell.is_some_and(|cell| cell.fg != Color::Reset || cell.bg != Color::Reset)
    });
    assert_eq!(coloured.count(), 0);
}

#[test]
fn a_theme_files_tints_set_the_views_and_the_rest_stay_its_own() {
    let home = scratch_home("sp-theme-tints");
    let dir = home.root().join("themes");
    fs::create_dir_all(&dir).ok();
    fs::write(
        dir.join("ink.toml"),
        "input = \"#102030\"\ncall_failed = \"#400000\"\n",
    )
    .ok();
    let user = MemoryStore::default();
    let (mut app, _wakes, agent) = start(&user, &dir, false);
    let own = *app.world().resource::<rig_tui::Theme>();

    command(&mut app, agent, "theme ink");
    let tinted = |app: &App| *app.world().resource::<rig_tui::Theme>();
    assert!(wait_for(&mut app, |app| tinted(app).input
        == Color::Rgb(0x10, 0x20, 0x30)));
    let view = tinted(&app);
    assert_eq!(view.call_failed, Color::Rgb(0x40, 0, 0));
    assert_eq!(view.raised_surface, own.raised_surface);

    // Back to `dark`, which sets no tints: the view's own again.
    command(&mut app, agent, "theme dark");
    assert!(wait_for(&mut app, |app| tinted(app) == own));
}

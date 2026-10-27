use bevy_ecs::system::RunSystemOnce;
use crossterm::event::{Event, KeyEvent, KeyEventKind};
use rig_harness::ask::AskPlugin;
use rig_harness::pick::PickPlugin;
use rig_harness::prelude::*;

use super::*;

/// The frames a panel drew.
#[derive(Resource, Default)]
struct Drawn(usize);

/// The texts agents accepted, in order.
#[derive(Resource, Default)]
pub(crate) struct Delivered(pub(crate) Vec<String>);

/// A directory of its own for test `name`, beside the test binary, so
/// nothing lands in the source tree; empty.
pub(crate) fn scratch(name: &str) -> std::path::PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe
        .parent()
        .map(|deps| deps.join(format!("rig-tui-{name}")))
        .unwrap_or_default();
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// An agent with the terminal view's input and choices, but no terminal:
/// its keys are the [`PendingInput`] [`press`] pushes. Also returns the
/// agent.
pub(crate) fn view_app() -> (App, Entity) {
    let mut app = App::new();
    app.set_error_handler(rig_harness::error::panic)
        .insert_resource(Invoked::default())
        .add_plugins((AgentPlugin, AskPlugin, PickPlugin, TuiPlugin))
        .init_resource::<Delivered>()
        .add_observer(
            |accepted: On<Add<Started>>,
             messages: Query<&Incoming>,
             mut delivered: ResMut<Delivered>| {
                if let Ok(message) = messages.get(accepted.entity) {
                    delivered.0.push(message.text().to_owned());
                }
            },
        );
    add_view(&mut app);
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .single(world)
        .unwrap_or(Entity::PLACEHOLDER);
    (app, agent)
}

/// Writes `kind` events of `keys` (such as `ctrl+x ctrl+e`), then runs a
/// frame.
pub(crate) fn key_events(app: &mut App, keys: &str, kind: KeyEventKind) {
    let parsed: Result<Keys, String> = keys.parse();
    assert!(parsed.is_ok(), "{parsed:?}");
    for press in parsed.iter().flat_map(Keys::presses) {
        let key = KeyEvent::new_with_kind(press.code, press.modifiers, kind);
        push(app, KeyInput::Event(Event::Key(key)));
    }
    app.update();
}

/// Puts `item` among the frame's [`PendingInput`].
pub(crate) fn push(app: &mut App, item: KeyInput) {
    app.world_mut().resource_mut::<PendingInput>().0.push(item);
}

/// Presses `keys`, as [`key_events`] writes them.
pub(crate) fn press(app: &mut App, keys: &str) {
    key_events(app, keys, KeyEventKind::Press);
}

/// Types `text`, a key for each character, then runs a frame.
pub(crate) fn type_text(app: &mut App, text: &str) {
    let keys: Vec<String> = text
        .chars()
        .map(|character| match character {
            ' ' => "space".to_owned(),
            other => other.to_string(),
        })
        .collect();
    press(app, &keys.join(" "));
}

/// The main transcript pane.
pub(crate) fn main_pane(world: &World) -> Entity {
    let pane = world.try_query_filtered::<Entity, With<MainPane>>();
    let pane = pane.and_then(|mut panes| panes.single(world).ok());
    pane.unwrap_or(Entity::PLACEHOLDER)
}

/// The main pane's view; a default one without the pane.
pub(crate) fn main_view(app: &App) -> &TranscriptView {
    static NONE: std::sync::LazyLock<TranscriptView> = std::sync::LazyLock::new(Default::default);
    let world = app.world();
    world
        .get::<TranscriptView>(main_pane(world))
        .unwrap_or(&NONE)
}

/// Changes the main pane's view with `change`.
pub(crate) fn steer(app: &mut App, change: impl FnOnce(&mut TranscriptView)) {
    let pane = main_pane(app.world());
    if let Some(mut view) = app.world_mut().get_mut::<TranscriptView>(pane) {
        change(&mut view);
    }
}

/// Has the main pane keep its [`DrawnTranscript`].
pub(crate) fn keep_drawn(app: &mut App) {
    let pane = main_pane(app.world());
    if let Ok(mut pane) = app.world_mut().get_entity_mut(pane) {
        pane.insert(DrawnTranscript::default());
    }
}

/// The main pane's [`DrawnTranscript`]; an empty one without it.
pub(crate) fn drawn(app: &App) -> &DrawnTranscript {
    static NONE: std::sync::LazyLock<DrawnTranscript> = std::sync::LazyLock::new(Default::default);
    let world = app.world();
    world
        .get::<DrawnTranscript>(main_pane(world))
        .unwrap_or(&NONE)
}

/// The rows of `agent`'s transcript in the main pane, `height` rows high
/// and 80 wide, with `streaming` tool calls after the last message, as a
/// frame lays them out where the pane is scrolled, selection and
/// highlights included; notes where the pane is and what it drew, as a
/// frame does.
pub(crate) fn drawn_rows(
    app: &mut App,
    agent: Entity,
    height: usize,
    streaming: Vec<rig_core::message::ToolCall>,
) -> Vec<ratatui::text::Line<'static>> {
    use crate::pane::{PaneState, lay_out_panes};
    assert!(streaming.is_empty());
    let pane = main_pane(app.world());
    let area = ratatui::layout::Rect::new(0, 0, 80, u16::try_from(height).unwrap_or(u16::MAX));
    if let Some(mut canvas) = app.world_mut().get_mut::<crate::PanelCanvas>(pane) {
        canvas.reset(area);
    }
    if let Some(mut shown) = app
        .world_mut()
        .get_resource_mut::<rig_harness::main_agent::ShownAgent>()
    {
        shown.0 = Some(agent);
    }
    app.world_mut().run_system_once(lay_out_panes).ok();
    app.world()
        .get::<PaneState>(pane)
        .map(|state| state.rows.clone())
        .unwrap_or_default()
}

/// [`drawn_rows`] as text, a line per row.
pub(crate) fn on_screen(app: &mut App, agent: Entity, height: usize) -> Vec<String> {
    let rows = drawn_rows(app, agent, height, Vec::new());
    rows.iter().map(ToString::to_string).collect()
}

/// All of `agent`'s transcript, as text, a line per row.
pub(crate) fn transcript(app: &mut App, agent: Entity) -> Vec<String> {
    on_screen(app, agent, 1000)
}

/// The main editor; an empty one without it.
pub(crate) fn input(app: &App) -> &Editor {
    static NONE: std::sync::LazyLock<Editor> = std::sync::LazyLock::new(Default::default);
    let world = app.world();
    let main = world.try_query_filtered::<&Editor, With<MainEditor>>();
    let main = main.and_then(|mut main| main.single(world).ok());
    main.unwrap_or(&NONE)
}

/// Edits the main editor with `edit`.
pub(crate) fn edit(app: &mut App, edit: impl FnOnce(&mut Editor)) {
    let world = app.world_mut();
    let mut main = world.query_filtered::<&mut Editor, With<MainEditor>>();
    if let Ok(mut input) = main.single_mut(world) {
        edit(&mut input);
    }
}

/// The input's text.
pub(crate) fn typed(app: &App) -> String {
    input(app).text().to_owned()
}

#[test]
fn a_panel_plugin_is_inert_in_a_print_run() {
    let mut app = App::new();
    // A system that cannot run fails the test.
    app.set_error_handler(rig_harness::error::panic)
        .insert_resource(Invoked::default())
        .add_plugins(TuiPlugin)
        .init_resource::<Drawn>()
        .add_systems(Update, |mut redraw: MessageWriter<RequestRedraw>| {
            redraw.write(RequestRedraw);
        })
        .add_systems(
            PostUpdate,
            (|mut drawn: ResMut<Drawn>| drawn.0 += 1).in_set(TuiSystems::Draw),
        );
    app.update();
    app.update();
    let world = app.world();
    assert_eq!(world.resource::<Drawn>().0, 0);
    assert!(!world.contains_resource::<view::TuiView>());
}

#[test]
fn a_plugins_test_lays_out_a_transcript_without_a_terminal() {
    let mut app = App::new();
    app.set_error_handler(rig_harness::error::panic)
        .insert_resource(Invoked::default())
        .add_plugins((AgentPlugin, TuiPlugin));
    app.update();
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    let Ok(agent) = agent else { return };
    let note = Deliver::new(agent, "a note", DeliveryMode::Inform);
    app.world_mut().trigger(note);
    app.update();
    let lay_out = move |world: &mut World| {
        TranscriptLayout::lay_out(world, &TranscriptView::default(), agent, 40)
    };
    let laid_out = app.world_mut().run_system_once(lay_out).ok().flatten();
    assert!(laid_out.is_some());
    let Some(drawn) = laid_out else { return };
    assert_eq!(drawn.agent(), Some(agent));
    assert!(
        drawn.rows().iter().any(|row| row.text == "› a note"),
        "{drawn:#?}"
    );
    // As the frame that never comes would hold it.
    let pane = main_pane(app.world());
    app.world_mut().entity_mut(pane).insert(drawn);
}

#[test]
fn typed_keys_edit_the_input_and_enter_sends_it() {
    let (mut app, _) = view_app();
    type_text(&mut app, "helo");
    // Back over the `o`, then the missing `l`.
    press(&mut app, "left l");
    assert_eq!(typed(&app), "hello");
    press(&mut app, "ctrl+a");
    type_text(&mut app, "> ");
    press(&mut app, "enter");
    assert_eq!(app.world().resource::<Delivered>().0, ["> hello"]);
    // No model answers it, so it comes back to the input.
    assert_eq!(typed(&app), "> hello");
}

#[test]
fn a_commands_argument_completes_to_a_contributed_candidate() {
    let (mut app, _) = view_app();
    app.add_command("model", "Choose the model", |_: In<CommandArgs>| {});
    let models = CompletionSource::arg("/model")
        .candidate("openai/gpt-5", "")
        .candidate("acme/fast", "no key");
    app.world_mut().spawn(models);
    app.world_mut()
        .spawn(CompletionSource::arg("other").candidate("acme/other", ""));
    type_text(&mut app, "/model acme/");
    press(&mut app, "tab");
    assert_eq!(typed(&app), "/model acme/fast ");
}

#[test]
fn the_terminal_says_when_it_has_the_users_focus() {
    let (mut app, _) = view_app();
    assert_eq!(
        *app.world().resource::<TerminalFocus>(),
        TerminalFocus(None)
    );
    for (event, focused) in [(Event::FocusLost, false), (Event::FocusGained, true)] {
        push(&mut app, KeyInput::Event(event));
        app.update();
        let focus = *app.world().resource::<TerminalFocus>();
        assert_eq!(focus, TerminalFocus(Some(focused)));
    }
}

/// Whether the view, the input or the transcript counted as changed in the
/// last frame, as the redraw check reads them.
#[derive(Resource, Default)]
struct ChangedLastFrame(bool);

/// A frame with no terminal event and no action leaves the view, the input
/// and the transcript unchanged, so the redraw check can pace a streamed
/// reply instead of drawing every frame.
#[test]
fn a_frame_with_nothing_typed_changes_nothing_the_redraw_check_reads() {
    let (mut app, _) = view_app();
    app.init_resource::<ChangedLastFrame>().add_systems(
        bevy_app::Last,
        |view: Res<crate::view::TuiView>,
         input: Query<Ref<Editor>>,
         transcript: Query<Ref<TranscriptView>>,
         mut changed: ResMut<ChangedLastFrame>| {
            let transcript = transcript.iter().any(|view| view.is_changed());
            let input = input.iter().any(|input| input.is_changed());
            changed.0 = view.is_changed() || input || transcript;
        },
    );
    app.update();
    app.update();
    assert!(!app.world().resource::<ChangedLastFrame>().0);
}

/// Whether the transcript view counted as changed in the last frame, as a
/// plugin that redoes work when it changes reads it.
#[derive(Resource, Default)]
struct TranscriptChanged(bool);

/// A key that leaves the transcript view as it is, such as one typed into
/// the input, does not change it, so a plugin watching it does no work.
#[test]
fn a_key_typed_leaves_the_transcript_view_unchanged() {
    let (mut app, _) = view_app();
    app.init_resource::<TranscriptChanged>().add_systems(
        bevy_app::Last,
        |transcript: Query<Ref<TranscriptView>>, mut changed: ResMut<TranscriptChanged>| {
            changed.0 = transcript.iter().any(|view| view.is_changed());
        },
    );
    app.update();
    type_text(&mut app, "a");
    assert_eq!(input(&app).text(), "a");
    assert!(!app.world().resource::<TranscriptChanged>().0);
}

/// An app that builds its own screen from the public widgets, without the
/// default screen: its transcript draws and its input sends.
#[test]
fn a_screen_built_from_the_public_widgets_draws_and_sends() {
    use ratatui::layout::Constraint;

    let mut app = App::new();
    app.set_error_handler(rig_harness::error::panic)
        .insert_resource(Invoked::default())
        .add_plugins((AgentPlugin, AskPlugin, PickPlugin, TuiViewPlugin))
        .init_resource::<Delivered>()
        .add_observer(
            |accepted: On<Add<Started>>,
             messages: Query<&Incoming>,
             mut delivered: ResMut<Delivered>| {
                if let Ok(message) = messages.get(accepted.entity) {
                    delivered.0.push(message.text().to_owned());
                }
            },
        );
    add_view(&mut app);
    let world = app.world_mut();
    let pane = TranscriptPane::new(TranscriptSource::Shown).titled("chat");
    let pane = world.spawn((pane, MainPane)).id();
    let input = TuiPanel::new(Placement::Top(Constraint::Length(3)));
    world.spawn((Editor::default(), MainEditor, input));
    app.insert_resource(TuiTestScreen::new(60, 20));
    app.update();
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    let Ok(agent) = agent else { return };
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.push(rig_core::completion::Message::user(
            "where is the config read?",
        ));
    }
    app.update();
    let rows = app.world().resource::<TuiTestScreen>().rows();
    let area = app.world().resource::<TuiFrame>().area(pane);
    assert!(area.is_some_and(|area| area.y == 3), "{area:?}");
    assert!(
        rows.iter()
            .any(|row| row.contains("› where is the config read?")),
        "{rows:#?}"
    );
    assert!(rows.iter().any(|row| row.contains("chat")), "{rows:#?}");
    type_text(&mut app, "hello");
    press(&mut app, "enter");
    assert_eq!(app.world().resource::<Delivered>().0, ["hello"]);
    // No model answers either, so both come back to the input: the one
    // put in the conversation by hand as it is there.
    assert_eq!(typed(&app), "where is the config read?\n\nhello");
}

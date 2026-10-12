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
/// its keys are the [`TerminalEvent`]s [`press`] writes. Also returns the
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
                    delivered.0.push(message.text.clone());
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
        app.world_mut()
            .write_message(TerminalEvent(Event::Key(key)));
    }
    app.update();
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

/// The rows of `agent`'s transcript on a screen `height` rows high and 80
/// wide, with `streaming` tool calls after the last message, as a frame
/// draws them where the view is scrolled, selection and highlights
/// included; notes where the view is, as a frame does.
pub(crate) fn drawn_rows(
    app: &mut App,
    agent: Entity,
    height: usize,
    streaming: Vec<rig_core::message::ToolCall>,
) -> Vec<ratatui::text::Line<'static>> {
    use crate::render::{ShownAgent, Sources, decorate, lay_out_transcript};
    use crate::transcript::{CallFacts, Transcript};
    use rig_harness::transcript::{TranscriptEntries, TranscriptEntry};

    let draw =
        move |agents: Query<(&AgentId, &Conversation)>,
              (mut view, mut transcript_view): (ResMut<view::TuiView>, ResMut<TranscriptView>),
              (tools, origins): (Query<&ToolRenderer>, Query<&EntryRenderer>),
              (lists, entries): (Query<&TranscriptEntries>, Query<&TranscriptEntry>)| {
            let Ok((id, conversation)) = agents.get(agent) else {
                return Vec::new();
            };
            let shown = ShownAgent {
                entity: agent,
                id,
                conversation,
                changed: true,
                inbox: &[],
            };
            let renderers = tools.iter().map(|r| (r.tool.as_str(), &r.render)).collect();
            let entry_renderers = origins
                .iter()
                .map(|r| (r.origin.as_str(), &r.render))
                .collect();
            let shown_entries: Vec<_> = lists
                .get(agent)
                .into_iter()
                .flat_map(TranscriptEntries::iter)
                .filter_map(|entry| Some((entry, entries.get(entry).ok()?)))
                .collect();
            let (calls, facts) = (Default::default(), CallFacts::default());
            let sources = Sources {
                view: &transcript_view,
                renderers: &renderers,
                entry_renderers: &entry_renderers,
                calls: &calls,
                facts: &facts,
                partial: None,
                streaming: &streaming,
                entries: &shown_entries,
            };
            let mut rows = Transcript::default();
            let parts = lay_out_transcript(&mut rows, &sources, Some(&shown), 80);
            let visible = rows.visible(&parts, (80, height), &mut view.scroll);
            let selected = (
                transcript_view.selected.as_ref(),
                transcript_view.highlights.as_slice(),
            );
            let lines = decorate(visible.rows, visible.first, &visible.marks, selected);
            view.items = visible.marks.into_iter().map(|(item, _)| item).collect();
            transcript_view.set_top(visible.top);
            lines
        };
    app.world_mut().run_system_once(draw).unwrap_or_default()
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

/// The input's text.
pub(crate) fn typed(app: &App) -> String {
    app.world().resource::<TuiInput>().text().to_owned()
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
fn typed_keys_edit_the_input_and_enter_sends_it() {
    let (mut app, _) = view_app();
    type_text(&mut app, "helo");
    // Back over the `o`, then the missing `l`.
    press(&mut app, "left l");
    assert_eq!(typed(&app), "hello");
    press(&mut app, "ctrl+a");
    type_text(&mut app, "> ");
    press(&mut app, "enter");
    assert_eq!(typed(&app), "");
    assert_eq!(app.world().resource::<Delivered>().0, ["> hello"]);
}

#[test]
fn a_commands_argument_completes_to_a_contributed_candidate() {
    let (mut app, _) = view_app();
    app.add_command("model", "Choose the model", |_: In<CommandArgs>| {});
    let models = ArgCompletion::new("/model")
        .candidate("openai/gpt-5", "")
        .candidate("acme/fast", "no key");
    app.world_mut().spawn(models);
    app.world_mut()
        .spawn(ArgCompletion::new("other").candidate("acme/other", ""));
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
        app.world_mut().write_message(TerminalEvent(event));
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
         input: Res<crate::editor::TuiInput>,
         transcript: Res<crate::view::TranscriptView>,
         mut changed: ResMut<ChangedLastFrame>| {
            changed.0 = view.is_changed() || input.is_changed() || transcript.is_changed();
        },
    );
    app.update();
    app.update();
    assert!(!app.world().resource::<ChangedLastFrame>().0);
}

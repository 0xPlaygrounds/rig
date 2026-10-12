//! The terminal view. It reads agent components, keeps its own view state
//! in a resource, and sends the same requests any other view would. It
//! never owns the app's loop: a thread reads the terminal and wakes the
//! loop, `PreUpdate` handles what it read, and drawing is a system in
//! `PostUpdate` that runs when something drawn changed.
//!
//! The input is a multiline editor with the prompt history
//! ([`PromptHistory`]) and `/` and `@` completion. Answers are drawn as markdown and
//! edits as diffs; a plugin draws its own tools' calls with
//! [`AppToolRenderersExt::add_tool_renderer`], also while their arguments
//! stream, and its transcript entries
//! ([`TranscriptEntry`](rig_harness::transcript::TranscriptEntry)) and
//! delivered texts with [`AppToolRenderersExt::add_entry_renderer`]. The
//! [`TranscriptView`] says, and lets a plugin set, what is expanded and
//! selected (Alt+Up and Alt+Down select, Ctrl+O expands) and the text lit;
//! [`ScrollTo`] scrolls to an item or to a row of it a highlight lights,
//! and [`TranscriptView::top`] reads where the view is.
//!
//! It is the front ([`FRONT`]) of a run where someone sits at the
//! terminal ([`Invoked::interactive`](rig_harness::prelude::Invoked::interactive)):
//! no `--print` and stdin a terminal, unless another front's option
//! chose that one.
//!
//! A plugin adds to the view without touching it: a [`TuiPanel`] beside
//! the transcript or over the screen, drawn by the plugin's own system in
//! [`TuiSystems::Draw`] (see [`panel`]); a [`RequestRedraw`] for a frame;
//! the agent shown, [`Focused`] (the harness's
//! [`Shown`]), and showing another, [`Focus`]; the
//! status line's items ([`StatusItems`], [`AppStatus`]); and the
//! [`TuiScreen`]'s size. It takes part in the input too: a key binding
//! ([`keys`]: the view's own keys are bindings as well, each doing an
//! [`InputAction`], which a plugin may also send), the keys for a
//! panel ([`KeyFocus`]), the draft in the input ([`TuiInput`]), the
//! candidates for a command's argument ([`ArgCompletion`]), the terminal's
//! events ([`TerminalEvent`]) and focus ([`TerminalFocus`]), its own escape
//! codes ([`TerminalWrite`], [`CopyToClipboard`]), a program run with the
//! terminal ([`RunInTerminal`]), the clipboard's image
//! ([`read_clipboard_image`]), and a choice for the user, a
//! [`Picker`](rig_harness::pick::Picker). The view shows every plugin's
//! pickers and answers every agent's questions ([`Ask`](rig_harness::ask::Ask)),
//! one at a time: in front of the transcript, in the input for one with
//! initial text or a multi-line answer, or in a strip above the input for
//! one shown beside the work. These are there in every run, also
//! one with another front such as `--print`, which never draws a frame.
//! [`ratatui`] and [`crossterm`] are re-exported so a plugin uses the same
//! versions.

mod clipboard;
mod complete;
pub mod diff;
mod editor;
mod history;
mod input;
mod jsonl;
pub mod keys;
pub mod markdown;
pub mod panel;
mod pick;
mod render;
mod renderers;
mod status;
mod terminal;
mod transcript;
mod view;
mod wrap;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::common_conditions::any_match_filter;
use rig_ecs::stage::AgentLoop;
use rig_harness::prelude::launcher::BUILD_ORIGIN;
use rig_harness::prelude::{Answerer, AppCliExt, Front, RigHome, Shown};

pub use clipboard::read_clipboard_image;
pub use complete::{ArgCompletion, Candidate};
pub use crossterm;
pub use editor::TuiInput;
pub use history::{HistorySystems, PromptHistory};
pub use input::TerminalEvent;
pub use keys::{
    AppKeyBindingsExt, InputAction, KeyBinding, KeyFocus, KeyPress, KeyPressed, KeyScope, Keys,
    run_command,
};
pub use panel::{Focused, PanelCanvas, Placement, RequestRedraw, TuiPanel, TuiScreen, TuiSystems};
pub use ratatui;
pub use renderers::{
    AppToolRenderersExt, CallState, EntryRenderer, EntryView, RESULT_LINES, RenderEntry,
    RenderToolCall, ToolCallView, ToolRenderer, excerpt,
};
pub use status::{AppStatus, Side, StatusItem, StatusItems, StatusSystems, Tone};
pub use terminal::{CopyToClipboard, RunInTerminal, TerminalExit, TerminalFocus, TerminalWrite};
pub use view::{Focus, Highlight, ItemRow, ScrollTo, ShownNotice, TranscriptItem, TranscriptView};

/// The terminal view's name as a front.
pub const FRONT: &str = "tui";

/// Owns the terminal and draws the focused agent, when it is the run's
/// front: on an interactive terminal, unless another front's option was
/// given.
#[derive(Default)]
pub struct TuiPlugin;

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        // What other plugins use is there in any run; without the
        // terminal, no frame is ever due.
        status::add(app);
        // Spawned in `build`, so a plugin's `finish` may disable one.
        keys::add_built_in(app);
        app.add_entry_renderer(BUILD_ORIGIN, build_failed);
        app.init_resource::<render::FrameLayout>()
            .init_resource::<TranscriptView>()
            .init_resource::<TuiScreen>()
            .init_resource::<TuiInput>()
            .init_resource::<PromptHistory>()
            .init_resource::<TerminalFocus>()
            .add_message::<RequestRedraw>()
            .add_message::<TerminalEvent>()
            .add_message::<TerminalWrite>()
            .add_message::<CopyToClipboard>()
            .add_message::<InputAction>()
            .configure_sets(
                PostUpdate,
                (
                    TuiSystems::Prepare,
                    TuiSystems::Layout,
                    TuiSystems::Draw.run_if(render::frame_due),
                    TuiSystems::Render.run_if(render::frame_due),
                )
                    .chain()
                    .after(StatusSystems),
            )
            .configure_sets(
                Update,
                (HistorySystems::Record, HistorySystems::Save).chain(),
            );
        app.add_front(Front::new(FRONT).on_terminal());
    }

    /// Owns the terminal when the run is the terminal view's: any other
    /// front, such as a print run, has stdout for its own output.
    fn finish(&self, app: &mut App) {
        if !app.runs_front(FRONT) {
            return;
        }
        add_view(app);
        if let Some(home) = app.world().get_resource::<RigHome>() {
            let history = PromptHistory::load(home.0.history());
            app.insert_resource(history);
        }
        app.add_systems(Startup, terminal::open_terminal)
            .add_systems(
                Last,
                (
                    terminal::keep_screen_on_reload,
                    view::keep_draft.run_if(on_message::<AppExit>),
                )
                    .in_set(bevy_app::OnAppExitSystems),
            );
    }
}

/// Everything of the view but the terminal itself, which
/// [`terminal::open_terminal`] opens: the view's tests run it on the
/// events they write.
fn add_view(app: &mut App) {
    app.world_mut()
        .spawn((Name::new("answerer:tui"), Answerer::default()));
    app.add_message::<history::Sent>()
        .init_resource::<RigHome>()
        .init_resource::<Shown>()
        .init_resource::<view::TuiView>()
        .init_resource::<complete::FileIndex>()
        .init_resource::<clipboard::Clipboard>()
        .init_resource::<transcript::CallFacts>()
        .init_resource::<transcript::StreamingCalls>()
        .add_systems(Startup, (view::load_notices, view::load_draft))
        .add_systems(
            PreUpdate,
            (
                input::receive_events.run_if(resource_exists::<input::TerminalInput>),
                input::read_input,
                terminal::track_focus,
                complete::receive_paths,
                clipboard::receive_images,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                view::mark_focused,
                // A command typed this frame runs in the agent loop, so
                // its picker, its notices and a refusal come back in this
                // frame, and so does a question asked there.
                (
                    pick::queue_choices,
                    keys::arm_inline_keys,
                    view::collect_notices,
                )
                    .chain()
                    .after(AgentLoop),
                view::recall_messages.after(AgentLoop),
                history::record.in_set(HistorySystems::Record),
                history::save.in_set(HistorySystems::Save),
            ),
        )
        .add_systems(
            PostUpdate,
            (
                terminal::run_in_terminal
                    .run_if(any_match_filter::<(With<RunInTerminal>, Without<TerminalExit>)>)
                    .before(TuiSystems::Layout),
                (
                    complete::follow_input,
                    transcript::follow_streams,
                    render::layout
                        .run_if(resource_exists::<terminal::Tui>.and_then(render::needs_redraw)),
                )
                    .chain()
                    .in_set(TuiSystems::Layout),
                (terminal::report_releases, render::render)
                    .chain()
                    .in_set(TuiSystems::Render)
                    .run_if(resource_exists::<terminal::Tui>),
            ),
        )
        // After the frame, and after what runs on the way out, which may
        // write to the terminal before the view lets it go.
        .add_systems(
            Last,
            terminal::write_to_terminal
                .after(bevy_app::OnAppExitSystems)
                .run_if(resource_exists::<terminal::Tui>),
        )
        .add_observer(view::on_focus)
        .add_observer(view::on_scroll_to)
        .add_observer(transcript::note_call_facts);
}

/// The look of a failed build's note, which the model reads too.
fn build_failed(entry: &EntryView<'_>) -> Vec<ratatui::text::Line<'static>> {
    use ratatui::style::Style;
    let mut lines = vec![ratatui::text::Line::styled(
        "✗ build failed (noted for the agent)",
        Style::new().red().bold(),
    )];
    let limit = if entry.expanded {
        usize::MAX
    } else {
        RESULT_LINES
    };
    lines.extend(excerpt(entry.text, limit, Style::new().red().dim()));
    lines
}

#[cfg(test)]
mod tests;

//! The terminal view. It reads agent components, keeps its own view state
//! in a resource, and sends the same requests any other view would. It
//! never owns the app's loop: a thread reads the terminal and wakes the
//! loop, `PreUpdate` handles what it read, a key at a time ([`TuiKeys`]),
//! and drawing is a system in
//! `PostUpdate` that runs when something drawn changed.
//!
//! The input is a multiline editor with the prompt history
//! ([`PromptHistory`]) and completion (`/`, `@` and plugins' markers). Answers are drawn as markdown and
//! edits as diffs; a plugin draws its own tools' calls with
//! [`AppToolRenderersExt::add_tool_renderer`], also while their arguments
//! stream, its transcript notes
//! ([`Note`](rig_harness::note::Note)) and
//! delivered texts with [`AppToolRenderersExt::add_entry_renderer`], a
//! kind of content block (images, audio, video, documents) with
//! [`AppToolRenderersExt::add_content_renderer`], a kind of aside (a kept
//! branch, a split) with [`AppToolRenderersExt::add_aside_renderer`], the
//! user's, the model's and its reasoning's text with
//! [`AppToolRenderersExt::add_message_renderer`], and rows under every
//! call or message with [`AppToolRenderersExt::add_call_decorator`] and
//! [`AppToolRenderersExt::add_message_decorator`]. The line producers the
//! transcript draws with are public ([`markdown`], [`wrap`], [`diff`],
//! [`excerpt`]), so a plugin's own text looks the same. A transcript is a
//! [`TranscriptPane`] ([`pane`]) over the agent shown, another agent or a
//! conversation value, which a plugin places in a panel of its own; each
//! pane's [`TranscriptView`] says, and lets a plugin set, what is expanded
//! and selected (Alt+Up and Alt+Down select, Ctrl+O expands) and the text
//! lit; [`ScrollTo`] scrolls a pane to an item or to a row of it a
//! highlight lights, and [`TranscriptView::top`] reads where it is.
//! [`ShownNotice::slot`] says where a notice is drawn in an agent's
//! transcript ([`TranscriptSlot`]), and [`DrawnTranscript`], which a
//! plugin adds to a pane, holds the rows it laid out, by item, with those
//! drawn cut and the matches it lit. [`TranscriptLayout`] lays out any
//! agent's transcript the same way without drawing it, cut as drawn or
//! whole ([`Expanded`]), for an export, a copy or a search.
//! [`TuiFrame`] says where the last frame drew the transcript, the input,
//! the status line and what covers them, and a test draws the view to a
//! [`TuiTestScreen`] in memory.
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
//! [`InputAction`], which a plugin may also put among the keys), the keys for a
//! panel ([`KeyFocus`]), the draft in the input ([`MainInput`], an
//! [`Editor`] a panel may hold too), the
//! candidates completion lists ([`CompletionSource`]), the frame's
//! keys before any binding sees them ([`TerminalEvents`], filtered one at
//! a time in [`InputSystems::Filter`] of [`TuiKeys`], with
//! [`TuiInputState`] saying what the input does with them), the terminal's focus ([`TerminalFocus`]), its own escape
//! codes ([`TerminalWrite`], [`CopyToClipboard`], which also goes to this
//! machine's clipboard), a program run with the terminal
//! ([`RunInTerminal`]), and a choice for the user, a
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
pub mod pane;
pub mod panel;
mod pick;
mod render;
mod renderers;
mod screen;
pub mod status;
mod terminal;
mod transcript;
mod view;
pub mod wrap;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::SingleThreadedExecutor;
use bevy_ecs::schedule::common_conditions::any_match_filter;
use rig_core::conversation::BRANCH;
use rig_ecs::journal::SessionRewritten;
use rig_ecs::stage::AgentLoop;
use rig_harness::prelude::launcher::BUILD_ORIGIN;
use rig_harness::prelude::{Answerer, AppCliExt, AppKeepsExt, Front, RigHome, Shown};
use rig_harness::rig_memory::SPLIT;

pub use complete::{Candidate, Completion, CompletionKind, CompletionSource, CompletionTrigger};
pub use crossterm;
pub use editor::{Editor, MainEditor, MainInput};
pub use history::{HistorySystems, PromptHistory, Sent};
pub use input::{InputSystems, KeyInput, TerminalEvents, TuiInputState, TuiKeys};
pub use keys::{
    AppKeyBindingsExt, InputAction, KeyBinding, KeyFocus, KeyPress, KeyPressed, KeyScope, Keys,
    run_command,
};
pub use pane::{MainPane, MainTranscript, PaneKeys, TranscriptPane, TranscriptSource};
pub use panel::{
    Focused, PanelCanvas, Placement, Repaint, RequestRedraw, TuiPanel, TuiScreen, TuiSystems,
};
pub use pick::{ChoiceAction, ChoiceKeys, InlineStrip, PickerState};
pub use ratatui;
pub use render::{Expanded, TranscriptLayout, TuiFrame};
pub use renderers::{
    AppToolRenderersExt, AsideRenderer, AsideView, CallDecorator, CallState, ChildCall,
    ContentBlock, ContentKind, ContentRenderer, ContentView, DecorateMessage, DecoratedMessage,
    EntryRenderer, EntryView, MessageDecorator, MessageRenderer, MessageView, RESULT_LINES,
    RenderAside, RenderContent, RenderEntry, RenderMessage, RenderToolCall, Role, ToolCallView,
    ToolRenderer, branch_lines, excerpt, split_lines,
};
pub use screen::DefaultScreenPlugin;
pub use status::{AppStatus, Side, StatusItem, StatusItems, StatusLine, StatusSystems, Tone};
pub use terminal::{
    CopyToClipboard, RunInTerminal, TerminalExit, TerminalFocus, TerminalWrite, TuiTestScreen,
};
pub use transcript::PastEffects;
pub use view::{
    Continuation, DrawnLine, DrawnRow, DrawnTranscript, Focus, Highlight, ItemRow, LineSpan,
    MarkIn, Notices, ScrollTo, ShownNotice, TextMark, TranscriptItem, TranscriptSlot,
    TranscriptView,
};

/// The terminal view's name as a front.
pub const FRONT: &str = "tui";

/// Owns the terminal and draws the focused agent, when it is the run's
/// front: on an interactive terminal, unless another front's option was
/// given. It is the [`TuiViewPlugin`] with the [`DefaultScreenPlugin`].
#[derive(Default)]
pub struct TuiPlugin;

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((TuiViewPlugin, DefaultScreenPlugin));
    }
}

/// The terminal view without a screen: the terminal, the keys, the frame
/// and the widgets' systems, which draw whatever widgets are spawned. The
/// [`TuiPlugin`] adds it with the [`DefaultScreenPlugin`]; an app that
/// builds its own screen adds it alone.
#[derive(Default)]
pub struct TuiViewPlugin;

impl Plugin for TuiViewPlugin {
    fn build(&self, app: &mut App) {
        // What other plugins use is there in any run; without the
        // terminal, no frame is ever due.
        status::add(app);
        // Spawned in `build`, so a plugin's `finish` may disable one.
        keys::add_built_in(app);
        app.add_entry_renderer(BUILD_ORIGIN, build_failed)
            .add_aside_renderer(BRANCH, branch_lines)
            .add_aside_renderer(SPLIT, split_lines);
        // The notices shown and the unsent draft, in the session's files.
        app.keeps_files("tui");
        app.init_resource::<render::FrameLayout>()
            .init_resource::<TuiFrame>()
            .init_resource::<Notices>()
            .init_resource::<transcript::CallFacts>()
            .init_resource::<transcript::StreamingCalls>()
            .init_resource::<TuiScreen>()
            .init_resource::<PromptHistory>()
            .init_resource::<TerminalFocus>()
            .init_resource::<TerminalEvents>()
            .init_resource::<TuiInputState>()
            .add_message::<RequestRedraw>()
            .add_message::<Repaint>()
            .add_message::<Sent>()
            .add_message::<TerminalWrite>()
            .add_message::<CopyToClipboard>()
            .add_message::<PastEffects>()
            // One thread, as a key's work is small; a plugin built before
            // this one may have added to it already.
            .edit_schedule(TuiKeys, |keys| {
                keys.set_executor(SingleThreadedExecutor::new());
            })
            .configure_sets(
                TuiKeys,
                (InputSystems::Filter, InputSystems::Dispatch).chain(),
            )
            .add_systems(PreUpdate, input::run_keys)
            .add_systems(
                TuiKeys,
                input::clear_events
                    .in_set(InputSystems::Dispatch)
                    .run_if(not(resource_exists::<view::TuiView>))
                    .ambiguous_with(input::read_input),
            )
            .configure_sets(
                PostUpdate,
                (
                    TuiSystems::Prepare,
                    TuiSystems::Layout,
                    TuiSystems::Fit,
                    TuiSystems::Draw.run_if(render::frame_due),
                    TuiSystems::Render.run_if(render::frame_due),
                )
                    .chain()
                    .after(StatusSystems::Edit),
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
        app.add_systems(PreStartup, history::load)
            .add_systems(Startup, terminal::open_terminal)
            .add_systems(
                Last,
                (
                    terminal::keep_screen_on_reload,
                    (view::keep_draft, view::keep_notices).run_if(on_message::<AppExit>),
                )
                    .in_set(bevy_app::OnAppExitSystems),
            );
    }
}

/// Everything of the view but the terminal itself, which
/// [`terminal::open_terminal`] opens: tests run it on the keys they put
/// in, as `rig_harness_test_support::tui_view` does.
#[doc(hidden)]
pub fn add_view(app: &mut App) {
    app.world_mut()
        .spawn((Name::new("answerer:tui"), Answerer::default()));
    app.add_message::<rig_harness::watch::FileChanges>()
        .add_message::<SessionRewritten>()
        .init_resource::<RigHome>()
        .init_resource::<Shown>()
        .init_resource::<view::TuiView>()
        .init_resource::<complete::FileIndex>()
        .init_resource::<clipboard::Clipboard>()
        .add_systems(Startup, (view::load_notices, view::load_draft))
        .add_systems(
            PreUpdate,
            (
                (
                    input::receive_events.run_if(resource_exists::<input::TerminalInput>),
                    complete::receive_paths,
                    clipboard::receive_images,
                )
                    .chain()
                    .before(input::run_keys),
                (pick::mark_highlighted, input::note_state)
                    .chain()
                    .after(input::run_keys),
            ),
        )
        .add_systems(
            TuiKeys,
            (
                input::note_state.before(InputSystems::Filter),
                input::read_input.in_set(InputSystems::Dispatch),
            ),
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
                    pick::mark_highlighted,
                    pick::follow_previews,
                    keys::arm_inline_keys,
                    view::collect_notices,
                )
                    .chain()
                    .after(AgentLoop),
                view::recall_messages.after(AgentLoop),
                follow_rewrites
                    .before(HistorySystems::Record)
                    .run_if(on_message::<SessionRewritten>),
                history::record.in_set(HistorySystems::Record),
                history::save.in_set(HistorySystems::Save),
            ),
        )
        // One thread: the panels, then the panes in what they leave.
        .edit_schedule(render::LayOut, |layout| {
            layout.set_executor(SingleThreadedExecutor::new());
        })
        .add_systems(
            render::LayOut,
            (
                render::layout.run_if(terminal::has_screen.and_then(render::needs_redraw)),
                pane::lay_out_panes.run_if(render::frame_due),
            )
                .chain(),
        )
        .add_systems(
            PostUpdate,
            (
                terminal::run_in_terminal
                    .run_if(any_match_filter::<(With<RunInTerminal>, Without<TerminalExit>)>)
                    .before(TuiSystems::Layout),
                transcript::note_effects.in_set(TuiSystems::Prepare),
                input::note_state.in_set(TuiSystems::Prepare),
                (
                    pane::draw_panes,
                    editor::draw_editors,
                    status::draw_status_lines,
                    pick::draw_inline_strips,
                )
                    .in_set(TuiSystems::Draw),
                (
                    complete::follow_input,
                    transcript::follow_streams,
                    render::lay_out_frame,
                )
                    .chain()
                    .in_set(TuiSystems::Layout),
                render::lay_out_fitted
                    .after(TuiSystems::Fit)
                    .before(TuiSystems::Draw),
                (
                    terminal::report_releases.run_if(resource_exists::<terminal::Tui>),
                    render::render,
                )
                    .chain()
                    .in_set(TuiSystems::Render)
                    .run_if(terminal::has_screen),
            ),
        )
        // After the frame, and after what runs on the way out, which may
        // write to the terminal before the view lets it go.
        .add_systems(
            Last,
            (
                terminal::write_to_terminal.run_if(terminal::has_screen),
                clipboard::copy_to_system.run_if(resource_exists::<terminal::Tui>),
            )
                .after(bevy_app::OnAppExitSystems),
        )
        .add_observer(keys::release_focus)
        .add_observer(view::on_focus)
        .add_observer(view::on_scroll_to)
        .add_observer(transcript::note_agent_name);
}

/// Maps what the view keeps of the session through each rewrite of it,
/// with the same map: the prompt history, in memory and in its file, the
/// calls it knows of (with what they published, such as a diff) and the
/// notices shown. Before this frame's prompts are recorded.
fn follow_rewrites(
    mut rewritten: MessageReader<SessionRewritten>,
    mut history: ResMut<PromptHistory>,
    mut facts: ResMut<transcript::CallFacts>,
    mut notices: ResMut<Notices>,
) {
    for done in rewritten.read().filter(|done| done.result.is_ok()) {
        history.rewrite(&*done.with);
        facts.rewrite(&*done.with);
        notices.rewrite(&*done.with);
    }
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

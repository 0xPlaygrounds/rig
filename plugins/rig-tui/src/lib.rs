//! The terminal view: the front ([`FRONT`]) of a run where someone sits at
//! the terminal ([`Invoked::interactive`](rig_harness::prelude::Invoked::interactive)).
//! It reads agent components, keeps its own state in resources and
//! components, and sends the same requests any other view would. It never
//! owns the app's loop: a thread reads the terminal and wakes the loop,
//! `PreUpdate` handles input one key at a time ([`TuiKeys`]), and
//! `PostUpdate` lays out and draws a frame only when something drawn
//! changed ([`TuiSystems`]).
//!
//! [`TuiPlugin`] is [`TuiViewPlugin`] plus [`DefaultScreenPlugin`]. Where a
//! plugin hooks in:
//!
//! - **Panels:** [`TuiPanel`], drawn in [`TuiSystems::Draw`] ([`panel`]);
//!   [`RequestRedraw`] for a frame, [`TerminalSize`] for the screen.
//! - **Transcripts:** [`TranscriptPane`] over the [`ShownAgent`], another
//!   agent or a conversation, with its [`TranscriptView`], [`ScrollTo`],
//!   [`DrawnTranscript`] and [`TranscriptLayout`] ([`pane`]); [`ShowAgent`]
//!   shows another agent.
//! - **Looks:** [`TranscriptRender`] for tool calls, notes, content
//!   blocks, asides, message text and decorations; [`BuildFailedLooks`].
//! - **Status line:** [`StatusItems`], [`AppStatus`] ([`status`]).
//! - **Keys:** [`KeyBinding`], [`KeyFocus`] ([`keys`]); [`PendingInput`]
//!   in [`InputSystems::Filter`] for keys before any binding, with the
//!   [`InputContext`].
//! - **Input:** [`Editor`], [`MainInput`], [`PromptHistory`],
//!   [`CompletionSource`].
//! - **The terminal:** [`RunInTerminal`], [`TerminalWrite`],
//!   [`CopyToClipboard`], [`TuiFrame`], [`TerminalFocus`]; a test draws to a
//!   [`TuiTestScreen`].
//! - **Same-looking text:** [`markdown`], [`wrap`], [`diff`], [`highlight`],
//!   [`text_rows`], and
//!   the colors of what each thing is and the flat surfaces of panels,
//!   [`Theme`].
//!
//! The view shows every plugin's pickers and every agent's questions, one
//! at a time ([`PickerInput`]). All of this exists in every run, also under
//! another front such as `--print`, which never draws a frame. [`ratatui`]
//! and [`crossterm`] are re-exported so a plugin uses the same versions.

mod clipboard;
mod complete;
pub mod diff;
mod editor;
pub mod highlight;
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
mod theme;
mod transcript;
mod turns;
mod view;
pub mod wrap;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::common_conditions::any_match_filter;
use rig_ecs::stage::AgentLoop;
use rig_harness::prelude::launcher::BUILD_ORIGIN;
use rig_harness::prelude::{
    Answerer, AppCliExt, AppKeepsExt, AppSettingsExt, Front, RigHome, ShownAgent,
};

pub use complete::{Candidate, Completion, CompletionKind, CompletionSource, CompletionTrigger};
pub use crossterm;
pub use editor::{Editor, MainEditor, MainInput};
pub use history::{HistorySystems, PromptHistory, Sent};
pub use input::{InputContext, InputSystems, KeyInput, PendingInput, TuiKeys};
pub use keys::{
    AppKeyBindingsExt, InputAction, KeyBinding, KeyFocus, KeyPress, KeyPressed, KeyScope, Keys,
    run_command,
};
pub use pane::{MainPane, MainTranscript, PaneKeys, TranscriptPane, TranscriptSource};
pub use panel::{
    PanelCanvas, Placement, RedrawArea, RequestRedraw, TerminalSize, TuiPanel, TuiSystems,
};
pub use pick::{ChoiceAction, ChoiceKeys, InlineStrip, PickerInput};
pub use ratatui;
pub use render::{TranscriptLayout, TuiFrame};
pub use renderers::{
    AsideRender, AsideView, CallState, ChildCall, ContentBlock, ContentKind, ContentRender,
    ContentView, DecoratedMessage, EntryRender, EntryView, MessageRender, MessageView,
    OUTPUT_INDENT, RenderBatch, RenderContent, RenderInput, RenderKey, RenderOutput, RenderRequest,
    RenderRevision, RenderSystems, Role, ToolCallView, ToolRender, TranscriptRender, text_rows,
};
pub use screen::DefaultScreenPlugin;
pub use status::{AppStatus, Side, StatusItem, StatusItems, StatusLine, StatusSystems, Tone};
pub use terminal::{
    CopyToClipboard, RunInTerminal, TerminalExit, TerminalFocus, TerminalWrite, TuiTestScreen,
};
pub use theme::Theme;
pub use transcript::PastEffects;
pub use view::{
    DrawnLine, DrawnRow, DrawnTranscript, Highlight, ItemRow, LineSpan, MarkIn, Notices, RowWrap,
    ScrollTo, ShowAgent, ShownNotice, TextMark, TranscriptItem, TranscriptSlot, TranscriptView,
};

/// The terminal view's name as a front.
pub const FRONT: &str = "tui";

/// The terminal view with its default screen ([`TuiViewPlugin`] plus
/// [`DefaultScreenPlugin`]). It owns the terminal when it is the run's
/// front: on an interactive terminal, unless another front's option was given.
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
        renderers::add(app);
        app.add_systems(
            TranscriptRender,
            render_build_failed
                .in_set(BuildFailedLooks)
                .in_set(RenderSystems::Render),
        );
        // The notices shown and the unsent draft, in the session's files.
        app.keeps_files("tui");
        app.add_setting::<turns::TuiSettings>("tui");
        app.init_resource::<render::FrameLayout>()
            .init_resource::<TuiFrame>()
            .init_resource::<Notices>()
            .init_resource::<transcript::CallFacts>()
            .init_resource::<TerminalSize>()
            .init_resource::<Theme>()
            .init_resource::<PromptHistory>()
            .init_resource::<TerminalFocus>()
            .init_resource::<PendingInput>()
            .init_resource::<InputContext>()
            .add_message::<RequestRedraw>()
            .add_message::<RedrawArea>()
            .add_message::<Sent>()
            .add_message::<TerminalWrite>()
            .add_message::<CopyToClipboard>()
            .add_message::<PastEffects>()
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
                terminal::keep_screen_on_reload.in_set(bevy_app::OnAppExitSystems),
            );
    }
}

/// Adds the view without a terminal: its input, keys, pickers, questions
/// and the transcript it lays out, driven by the keys put in
/// [`PendingInput`] and drawn only to a [`TuiTestScreen`]. The
/// [`TuiViewPlugin`] adds it in `finish` when the run's front is the
/// terminal view, then opens the terminal; a test adds it once, after the
/// plugin, and inserts a [`TuiTestScreen`] instead of a terminal.
pub fn add_view(app: &mut App) {
    app.world_mut()
        .spawn((Name::new("answerer:tui"), Answerer::default()));
    app.add_message::<rig_harness::watch::FileChanges>()
        .init_resource::<RigHome>()
        .init_resource::<ShownAgent>()
        .init_resource::<view::TuiView>()
        .init_resource::<complete::FileIndex>()
        .init_resource::<turns::Told>()
        .register_required_components::<rig_ecs::agent::Partial, transcript::StreamingCalls>()
        .add_observer(transcript::forget_streams)
        .add_observer(clipboard::type_pasted)
        .add_observer(render::redraw_on_remove)
        .add_systems(Startup, (view::load_notices, view::load_draft))
        .add_systems(
            PreUpdate,
            (
                (
                    input::receive_events.run_if(resource_exists::<input::TerminalInput>),
                    complete::receive_paths,
                )
                    .chain()
                    .before(input::run_keys),
                (pick::mark_highlighted, input::note_input_context)
                    .chain()
                    .after(input::run_keys),
            ),
        )
        .add_systems(
            TuiKeys,
            (
                input::note_input_context.before(InputSystems::Filter),
                input::read_input.in_set(InputSystems::Dispatch),
            ),
        )
        .add_systems(
            Update,
            (
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
                view::recall_messages
                    .after(AgentLoop)
                    .after(view::collect_notices),
                history::record.in_set(HistorySystems::Record),
                history::save.in_set(HistorySystems::Save),
            ),
        )
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
                turns::tell_state,
                input::note_input_context.in_set(TuiSystems::Prepare),
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
        .add_systems(
            Last,
            (
                view::keep_draft,
                // With the notices sent in `Last`, such as a failed last save.
                (view::collect_notices, view::keep_notices)
                    .chain()
                    .after(rig_ecs::WriteLogs),
            )
                .in_set(bevy_app::OnAppExitSystems)
                .run_if(on_message::<AppExit>),
        )
        // After the frame, and after what runs on the way out, which may
        // write to the terminal before the view lets it go.
        .add_systems(
            Last,
            turns::restore_title
                .run_if(on_message::<AppExit>)
                .in_set(bevy_app::OnAppExitSystems),
        )
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
        .add_observer(transcript::note_agent_name)
        .add_observer(turns::turn_started)
        .add_observer(turns::turn_ended);
}

/// The system that draws a failed build's note, in
/// [`RenderSystems::Render`]. A plugin that draws it its own way runs after
/// it.
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BuildFailedLooks;

/// The look of a failed build's note, which the model reads too.
fn build_failed(entry: &EntryView<'_>) -> Vec<ratatui::text::Line<'static>> {
    use ratatui::style::Style;
    let mut lines = vec![ratatui::text::Line::styled(
        "✗ build failed (noted for the agent)",
        Style::new().red().bold(),
    )];
    lines.extend(text_rows(entry.text, Style::new().red().dim()));
    lines
}

/// Draws the notes of failed builds.
fn render_build_failed(mut batch: ResMut<RenderBatch>) {
    for (entry, output) in batch.entries_mut(BUILD_ORIGIN) {
        output.lines = build_failed(&entry.view());
    }
}

#[cfg(test)]
mod tests;

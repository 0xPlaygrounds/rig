//! A read-only view of a conversation no agent of this app holds, such as
//! a saved session read with `rig_memory::Session::read` or a fork's
//! preview: a [`ConversationPanel`] drawn by the transcript's own layout
//! and renderers, in a box over the screen.

use bevy_ecs::prelude::*;
use crossterm::event::KeyCode;
use ratatui::layout::Constraint;
use ratatui::widgets::{Block, Paragraph};
use rig_core::conversation::Conversation;

use super::keys::{KeyBinding, KeyPress, KeyPressed, KeyScope};
use super::panel::{PanelCanvas, Placement, RequestRedraw, TuiPanel};
use super::render::{ShownAgent, Sources, lay_out_transcript};
use super::renderers::Looks;
use super::transcript::{CallFacts, Scroll, Transcript};
use super::view::{ItemRow, TranscriptItem, TranscriptView};

/// Rows a page key moves.
const PAGE: usize = 10;

/// A read-only view of a conversation no agent of this app holds, drawn in
/// its panel by the transcript's own layout and renderers. With the
/// [`KeyFocus`](crate::KeyFocus), Up and Down, PageUp and PageDown, and
/// Home and End scroll it, and Esc closes it (the view despawns the
/// entity). Its messages are laid out again only when it changes or the
/// panel's width does.
///
/// ```no_run
/// use rig_harness::prelude::*;
/// use rig_harness::rig_core::conversation::Conversation;
/// use rig_tui::{ConversationPanel, KeyFocus, TranscriptItem};
///
/// /// Shows `conversation` from message `turn` and gives it the keys.
/// fn show(conversation: Conversation, turn: Option<MessageId>, mut focus: ResMut<KeyFocus>, mut commands: Commands) {
///     let at = turn.map(TranscriptItem::Message);
///     let title = "main · turn 3".to_owned();
///     focus.0 = Some(commands.spawn(ConversationPanel { title, conversation, at }).id());
/// }
/// ```
#[derive(Component, Clone, Debug)]
#[require(
    TuiPanel = TuiPanel::new(Placement::Over {
        width: Constraint::Percentage(90),
        height: Constraint::Percentage(90),
    }),
    Showing
)]
#[component(on_add = bind_keys)]
pub struct ConversationPanel {
    /// Its border's title, such as a session's name and a turn.
    pub title: String,
    /// The conversation.
    pub conversation: Conversation,
    /// The item put at the top when it opens, such as a turn's first
    /// message.
    pub at: Option<TranscriptItem>,
}

/// How a [`ConversationPanel`] is drawn: its rows and where it is
/// scrolled.
#[derive(Component, Default)]
pub(crate) struct Showing {
    transcript: Transcript,
    scroll: Scroll,
    /// Whether it went to its `at` yet.
    opened: bool,
    /// The rows shown at the last frame.
    height: usize,
}

/// A key of a [`ConversationPanel`], on its binding's entity.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
enum Move {
    Up(usize),
    Down(usize),
    Top,
    Bottom,
    Close,
}

/// Binds the panel's keys in its own scope, on entities it owns, so they
/// go with it.
fn bind_keys(mut world: bevy_ecs::world::DeferredWorld, context: bevy_ecs::lifecycle::HookContext) {
    let panel = context.entity;
    let keys = [
        (KeyCode::Up, Move::Up(1), "Scroll up"),
        (KeyCode::Down, Move::Down(1), "Scroll down"),
        (KeyCode::PageUp, Move::Up(PAGE), "Page up"),
        (KeyCode::PageDown, Move::Down(PAGE), "Page down"),
        (KeyCode::Home, Move::Top, "To the top"),
        (KeyCode::End, Move::Bottom, "To the bottom"),
        (KeyCode::Esc, Move::Close, "Close"),
    ];
    let mut commands = world.commands();
    for (code, step, label) in keys {
        let binding = KeyBinding::new(KeyPress::key(code), label).in_scope(KeyScope::Panel(panel));
        let name = Name::new(format!("key:{}", binding.keys));
        commands
            .spawn((name, binding, step, ChildOf(panel)))
            .observe(move_panel);
    }
}

/// Scrolls or closes the panel whose key was pressed.
fn move_panel(
    pressed: On<KeyPressed>,
    keys: Query<(&Move, &ChildOf)>,
    mut panels: Query<&mut Showing>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let Ok((step, of)) = keys.get(pressed.entity) else {
        return;
    };
    let panel = of.parent();
    let Ok(mut shown) = panels.get_mut(panel) else {
        return;
    };
    let page = shown.height.saturating_sub(1).max(1);
    match *step {
        Move::Up(rows) => shown.scroll.up(rows.min(page)),
        Move::Down(rows) => shown.scroll.down(rows.min(page)),
        Move::Top => shown.scroll.up(usize::MAX),
        Move::Bottom => shown.scroll.follow(),
        Move::Close => {
            commands.entity(panel).despawn();
        }
    }
    redraw.write(RequestRedraw);
}

/// Draws each conversation panel with room into its canvas: its border
/// and title, and the rows where it is scrolled, laid out as the
/// transcript lays out an agent's.
pub(crate) fn draw_panels(
    mut panels: Query<(
        Entity,
        Ref<ConversationPanel>,
        &mut Showing,
        &mut PanelCanvas,
    )>,
    looks: Looks,
) {
    let renderers = looks.by_kind();
    // A view of its own: the shown agent's marks and expanded items name
    // messages by id, which this conversation's may share.
    let view = TranscriptView::default();
    let facts = CallFacts::default();
    let sources = Sources {
        view: &view,
        renderers: &renderers,
        facts: &facts,
        whole: false,
        note_cut: false,
    };
    for (entity, panel, mut shown, mut canvas) in &mut panels {
        let block = Block::bordered().title(format!(" {} · Esc closes ", panel.title));
        let inner = block.inner(canvas.area());
        if inner.is_empty() {
            continue;
        }
        let shown = &mut *shown;
        let mut value = ShownAgent::value(&panel.conversation, entity);
        value.changed = panel.is_changed();
        let parts = lay_out_transcript(&mut shown.transcript, &sources, Some(&value), inner.width);
        if !shown.opened {
            shown.opened = true;
            if let Some(at) = panel.at.clone() {
                shown.scroll.to(at, ItemRow::Top(0));
            }
        }
        shown.height = usize::from(inner.height);
        let size = (inner.width, shown.height);
        let visible = shown
            .transcript
            .visible(&parts, size, &mut shown.scroll, &[]);
        canvas.render(Paragraph::new(visible.rows).block(block));
    }
}

#[cfg(test)]
mod tests;

//! The dialog in the terminal view: a box over the screen that takes the
//! keys ([`KeyFocus`]) while an [`ApprovalBatch`] is open. Each agent's
//! rows under its name, each with a checkbox, the tool and its arguments
//! in a line; the note below them.
//!
//! Keys: Up and Down select, Space ticks or unticks, `a` ticks all, `n`
//! none, `e` (or Right) shows the selected call's full arguments, Tab
//! moves to the note and back, Enter confirms, Esc denies all.

use std::collections::HashSet;

use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::Line;
use rig_tui::ratatui::widgets::{Block, Paragraph, Wrap};
use rig_tui::{
    InputSystems, KeyFocus, KeyInput, PanelCanvas, Placement, RequestRedraw, TerminalEvents,
    TuiKeys, TuiPanel, TuiPlugin, TuiSystems,
};

use super::*;

/// The dialog's box, showing the dialog `batch`: the row selected, the
/// rows shown in full, and whether the keys type the note.
#[derive(Component, Debug)]
pub struct BatchPanel {
    /// The [`ApprovalBatch`] shown.
    pub batch: Entity,
    /// The row selected, an index of [`ApprovalBatch::rows`].
    pub selected: usize,
    /// The calls whose arguments are shown in full.
    pub expanded: HashSet<Entity>,
    /// Whether the keys type the note.
    pub typing_note: bool,
}

pub(super) fn add(app: &mut App) {
    if !app.is_plugin_added::<TuiPlugin>() {
        app.add_plugins(TuiPlugin);
    }
    app.add_observer(open)
        .add_observer(close)
        .add_systems(TuiKeys, keys.in_set(InputSystems::Filter))
        .add_systems(Update, keep_focus)
        .add_systems(PostUpdate, draw.in_set(TuiSystems::Draw));
}

/// A dialog opened: its box opens over the screen with the keys.
fn open(opened: On<Add<ApprovalBatch>>, mut focus: ResMut<KeyFocus>, mut commands: Commands) {
    let over = Placement::Over {
        width: Constraint::Percentage(80),
        height: Constraint::Percentage(70),
    };
    let panel = BatchPanel {
        batch: opened.entity,
        selected: 0,
        expanded: HashSet::new(),
        typing_note: false,
    };
    focus.0 = Some(commands.spawn((panel, TuiPanel::new(over))).id());
}

/// A dialog closed: its box goes, and the keys go back.
fn close(
    closed: On<Remove<ApprovalBatch>>,
    panels: Query<(Entity, &BatchPanel)>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    for (panel, shown) in &panels {
        if shown.batch == closed.entity {
            commands.entity(panel).despawn();
            redraw.write(RequestRedraw);
        }
    }
}

/// The open dialog keeps the keys while nothing else took them.
fn keep_focus(panels: Query<Entity, With<BatchPanel>>, mut focus: ResMut<KeyFocus>) {
    if focus.0.is_none()
        && let Some(panel) = panels.iter().next()
    {
        focus.0 = Some(panel);
    }
}

/// Takes the key being done while the dialog has the keys, and does it.
fn keys(
    mut events: ResMut<TerminalEvents>,
    focus: Res<KeyFocus>,
    mut panels: Query<&mut BatchPanel>,
    mut batches: Query<&mut ApprovalBatch>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let Some(Ok(mut panel)) = focus.0.map(|focused| panels.get_mut(focused)) else {
        return;
    };
    let Ok(mut batch) = batches.get_mut(panel.batch) else {
        return;
    };
    for item in std::mem::take(&mut events.0) {
        let KeyInput::Event(event) = item else {
            continue;
        };
        let key = match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => key,
            Event::Paste(text) if panel.typing_note => {
                batch.note.push_str(text.lines().next().unwrap_or_default());
                redraw.write(RequestRedraw);
                continue;
            }
            _ => continue,
        };
        let plain = !key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
        let last = batch.rows().count().saturating_sub(1);
        match key.code {
            KeyCode::Enter => {
                commands.trigger(ConfirmBatch {
                    entity: panel.batch,
                });
            }
            KeyCode::Esc => {
                commands.trigger(DenyBatch {
                    entity: panel.batch,
                });
            }
            KeyCode::Tab | KeyCode::BackTab => panel.typing_note = !panel.typing_note,
            KeyCode::Backspace if panel.typing_note => {
                batch.note.pop();
            }
            KeyCode::Char(typed) if panel.typing_note && plain => batch.note.push(typed),
            KeyCode::Up => panel.selected = panel.selected.saturating_sub(1),
            KeyCode::Down => panel.selected = panel.selected.saturating_add(1).min(last),
            KeyCode::Char(' ') => {
                if let Some(row) = batch.row_mut(panel.selected) {
                    row.checked = !row.checked;
                }
            }
            KeyCode::Char('a') => batch.tick_all(true),
            KeyCode::Char('n') => batch.tick_all(false),
            KeyCode::Char('e') | KeyCode::Right => {
                if let Some(call) = batch.rows().nth(panel.selected).map(|row| row.call)
                    && !panel.expanded.remove(&call)
                {
                    panel.expanded.insert(call);
                }
            }
            _ => continue,
        }
        redraw.write(RequestRedraw);
    }
}

/// Draws each open dialog: a group per agent, a row per call, the note,
/// and the keys.
fn draw(mut panels: Query<(&BatchPanel, &mut PanelCanvas)>, batches: Query<&ApprovalBatch>) {
    for (panel, mut canvas) in &mut panels {
        let Ok(batch) = batches.get(panel.batch) else {
            continue;
        };
        let mut lines = Vec::new();
        let mut index = 0;
        for group in &batch.groups {
            lines.push(Line::from(group.agent_name.clone()).bold());
            for row in &group.rows {
                let tick = if row.checked { "[x]" } else { "[ ]" };
                let line = Line::from(format!("  {tick} {}  {}", row.tool, row.summary));
                let selected = index == panel.selected && !panel.typing_note;
                lines.push(if selected { line.reversed() } else { line });
                if panel.expanded.contains(&row.call) {
                    let full = row.arguments.lines();
                    lines.extend(full.map(|text| Line::from(format!("        {text}")).dim()));
                }
                index += 1;
            }
        }
        lines.push(Line::default());
        let note = Line::from(format!("Note for the denied: {}", batch.note));
        lines.push(if panel.typing_note {
            note.reversed()
        } else {
            note
        });
        lines.push(
            Line::from(
                "Space tick  a all  n none  e arguments  Tab note  Enter confirm  Esc deny all",
            )
            .style(Style::new().dim()),
        );
        let title = format!(" Approve {index} tool calls ");
        canvas.render(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(Block::bordered().title(title)),
        );
    }
}

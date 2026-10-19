//! The terminals panel of the terminal view: the open terminals by name,
//! and the live screen of the one shown. `/term [name]` shows one and gives
//! the panel the keys, which then go to the program until Esc.

use rig_harness::prelude::*;
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::{Style, Stylize as _};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::ratatui::widgets::{Padding, Paragraph};
use rig_tui::{
    InputSystems, KeyFocus, KeyInput, PanelCanvas, PendingInput, Placement, RequestRedraw, Theme,
    TuiPanel, TuiSystems,
};

use crate::terminal_sessions::{Output, Terminal, keys};

/// The panel's entity, and the terminal it shows.
#[derive(Component, Default)]
pub struct TerminalsPanel {
    /// The terminal shown; the newest when it is gone.
    pub shown: Option<Entity>,
}

pub(crate) fn add(app: &mut App) {
    app.add_message::<RequestRedraw>()
        .init_resource::<PendingInput>()
        .init_resource::<Theme>()
        .add_systems(Startup, spawn)
        .add_systems(rig_tui::TuiKeys, type_keys.in_set(InputSystems::Filter))
        .add_systems(
            PostUpdate,
            (place.in_set(TuiSystems::Prepare), draw.in_set(TuiSystems::Draw)),
        )
        .add_observer(redraw)
        .add_command(
            "term",
            "Show a terminal the agent opened and type in it (Esc gives the keys back): /term [name]",
            show,
        );
}

fn spawn(mut commands: Commands) {
    commands.spawn((Name::new("terminals panel"), TerminalsPanel::default()));
}

/// A frame for each change of a terminal's screen.
fn redraw(_: On<Insert<Done<Output>>>, mut redraws: MessageWriter<RequestRedraw>) {
    redraws.write(RequestRedraw);
}

/// The panel is there while a terminal is, and shows one that is.
fn place(
    mut panels: Query<(Entity, &mut TerminalsPanel, Has<TuiPanel>)>,
    terminals: Query<Entity, With<Terminal>>,
    mut commands: Commands,
) {
    for (panel, mut state, placed) in &mut panels {
        let live = state.shown.is_some_and(|shown| terminals.contains(shown));
        if !live {
            let newest = terminals.iter().max_by_key(|entity| entity.index());
            if state.shown != newest {
                state.shown = newest;
            }
        }
        match (state.shown.is_some(), placed) {
            (true, false) => {
                let placement = Placement::Bottom(Constraint::Percentage(45));
                commands.entity(panel).insert(TuiPanel::new(placement));
            }
            (false, true) => {
                commands.entity(panel).remove::<TuiPanel>();
            }
            _ => {}
        }
    }
}

/// The names, the shown one lit, over the shown terminal's screen, the
/// rows about its cursor when the panel is shorter.
fn draw(
    mut panels: Query<(&mut PanelCanvas, &TerminalsPanel, Entity)>,
    terminals: Query<(Entity, &Terminal)>,
    focus: Option<Res<KeyFocus>>,
    theme: Res<Theme>,
) {
    for (mut canvas, state, panel) in &mut panels {
        let typing = focus.as_ref().is_some_and(|focus| focus.0 == Some(panel));
        let mut names = Vec::new();
        let mut screen = None;
        for (entity, terminal) in &terminals {
            let shown = state.shown == Some(entity);
            let style = if shown {
                theme.selection()
            } else {
                Style::new()
            };
            names.push(Span::styled(format!(" {} ", terminal.name), style));
            if shown {
                screen = Some(terminal);
            }
        }
        let Some(terminal) = screen else { continue };
        let (lines, cursor, exit) = {
            let state = terminal.pty.lock();
            (state.screen.lines(), state.screen.cursor_row(), state.exit)
        };
        let height = usize::from(canvas.area().height.saturating_sub(2));
        let from = (cursor + 1).saturating_sub(height).min(lines.len());
        let body: Vec<Line> = lines
            .into_iter()
            .skip(from)
            .take(height)
            .map(Line::from)
            .collect();
        let how = match (exit, typing) {
            (Some(code), _) => format!(" exited with code {code} "),
            (None, true) => " typing: Esc gives the keys back ".to_owned(),
            (None, false) => " /term to type ".to_owned(),
        };
        let block = theme
            .panel(Line::from(names))
            .padding(Padding::horizontal(1))
            .title_bottom(Line::from(how).dim());
        canvas.render(Paragraph::new(body).block(block));
    }
}

/// While the panel has the keys, what the user types goes to the terminal
/// shown, and its agent is told the next time it looks.
fn type_keys(
    events: Res<PendingInput>,
    focus: Option<Res<KeyFocus>>,
    panels: Query<&TerminalsPanel>,
    terminals: Query<&Terminal>,
) {
    let shown = focus
        .and_then(|focus| focus.0)
        .and_then(|panel| panels.get(panel).ok())
        .and_then(|panel| panel.shown)
        .and_then(|shown| terminals.get(shown).ok());
    let Some(terminal) = shown else {
        return;
    };
    for item in &events.0 {
        let KeyInput::Event(event) = item else {
            continue;
        };
        let bytes = match event {
            // Esc gives the keys back to the input.
            Event::Key(key) if key.code == KeyCode::Esc => continue,
            Event::Key(key) if key.kind != KeyEventKind::Release => keys::of_key(key),
            Event::Paste(text) => text.clone().into_bytes(),
            _ => continue,
        };
        if !bytes.is_empty() {
            terminal.pty.send(&bytes, true).ok();
        }
    }
}

/// `/term [name]`: shows that terminal, or the one shown, and gives it the
/// keys.
fn show(
    In(args): In<CommandArgs>,
    mut panels: Query<(Entity, &mut TerminalsPanel)>,
    terminals: Query<(Entity, &Terminal)>,
    focus: Option<ResMut<KeyFocus>>,
    mut notices: MessageWriter<Notice>,
) {
    let name = args.args.trim();
    let found = terminals
        .iter()
        .filter(|(_, terminal)| name.is_empty() || terminal.name == name)
        .max_by_key(|(entity, _)| entity.index());
    let (Some((terminal, _)), Ok((panel, mut state))) = (found, panels.single_mut()) else {
        let names: Vec<&str> = terminals.iter().map(|(_, t)| t.name.as_str()).collect();
        let text = if names.is_empty() {
            "No terminal is open.".to_owned()
        } else {
            format!("No terminal `{name}`; open: {}.", names.join(", "))
        };
        notices.write(Notice::error(args.agent, text));
        return;
    };
    if name.is_empty() {
        state.shown = state.shown.or(Some(terminal));
    } else {
        state.shown = Some(terminal);
    }
    if let Some(mut focus) = focus {
        focus.0 = Some(panel);
    }
}

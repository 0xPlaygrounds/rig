//! The palette in the terminal view. Ctrl-P opens a box over the screen
//! that takes the keys ([`KeyFocus`]), leaving the draft in the input as
//! it is. What is typed is its filter, read in order from the
//! [`PendingInput`], and the rows are ranked afresh from what exists
//! each time it changes. Up and Down (or Tab) select, Enter picks, and Esc
//! (or Ctrl-P again) closes it.

use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::ratatui::widgets::{List, ListItem, ListState, StatefulWidget};
use rig_tui::{
    AppKeyBindingsExt, InputSystems, KeyBinding, KeyFocus, KeyInput, KeyPress, KeyPressed,
    MainInput, PanelCanvas, PendingInput, Placement, RequestRedraw, ShowAgent, Theme, TuiPanel,
    TuiPlugin, TuiSystems,
};

use super::*;

/// Pick `entry` for `agent`: what Enter in the palette does with the
/// selected entry, which it then keeps among the recent picks.
#[derive(Event, Clone, Debug)]
pub struct Activate {
    /// The agent it is picked for: the one shown.
    pub agent: Entity,
    /// The entry.
    pub entry: Entry,
}

/// The open palette, a [`TuiPanel`] over the screen: its filter, the rows
/// ranked for it, and the row selected.
#[derive(Component)]
pub(crate) struct Palette {
    pub(crate) filter: String,
    pub(crate) rows: Vec<Entry>,
    selected: usize,
}

impl Palette {
    /// Ranks the rows for the filter now, and selects the first.
    fn refilter(&mut self, sources: &PaletteSources) {
        self.rows = sources.rank(&self.filter);
        self.selected = 0;
    }
}

/// The palette's systems, among the terminal view's input filters
/// ([`InputSystems::Filter`]), so a command it picks runs in the same
/// frame.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PaletteSystems;

pub(super) fn add(app: &mut App) {
    if !app.is_plugin_added::<TuiPlugin>() {
        app.add_plugins(TuiPlugin);
    }
    // Shadows the input's Ctrl-P (the line above, which Up does too).
    app.add_key_binding(
        KeyBinding::new(KeyPress::ctrl('p'), "Command palette"),
        toggle,
    );
    app.add_observer(activate)
        .configure_sets(
            rig_tui::TuiKeys,
            PaletteSystems.in_set(InputSystems::Filter),
        )
        .add_systems(rig_tui::TuiKeys, type_in_palette.in_set(PaletteSystems))
        .add_systems(PostUpdate, draw.in_set(TuiSystems::Draw));
}

/// Ctrl-P: opens the palette with an empty filter, giving it the keys;
/// closes it when open.
fn toggle(
    pressed: On<KeyPressed>,
    open: Query<Entity, With<Palette>>,
    sources: PaletteSources,
    mut focus: ResMut<KeyFocus>,
    mut commands: Commands,
) {
    if pressed.released {
        return;
    }
    if let Ok(palette) = open.single() {
        commands.entity(palette).despawn();
        return;
    }
    let mut palette = Palette {
        filter: String::new(),
        rows: Vec::new(),
        selected: 0,
    };
    palette.refilter(&sources);
    let over = Placement::Over {
        width: Constraint::Percentage(70),
        height: Constraint::Percentage(60),
    };
    focus.0 = Some(commands.spawn((palette, TuiPanel::new(over))).id());
}

/// Types into the open palette, in order: characters and Backspace edit
/// the filter, Up, Down and Tab select, Enter picks the selected entry
/// for the agent shown. It closes when it loses the keys, as on an Esc it
/// does not bind. The keys of the frame it opened in were the input's.
fn type_in_palette(
    events: Res<PendingInput>,
    mut open: Query<(Entity, &mut Palette), With<TuiPanel>>,
    (focus, shown): (Res<KeyFocus>, Res<ShownAgent>),
    sources: PaletteSources,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let Ok((panel, mut palette)) = open.single_mut() else {
        return;
    };
    if focus.0 != Some(panel) {
        commands.entity(panel).despawn();
        return;
    }
    let last = |palette: &Palette| palette.rows.len().saturating_sub(1);
    for item in &events.0 {
        let KeyInput::Event(event) = item else {
            continue;
        };
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => match key.code {
                KeyCode::Enter => {
                    commands.entity(panel).despawn();
                    if let (Some(agent), Some(entry)) =
                        (shown.0, palette.rows.get(palette.selected).cloned())
                    {
                        commands.trigger(Activate { agent, entry });
                    }
                    return;
                }
                KeyCode::Up | KeyCode::BackTab => {
                    palette.selected = palette.selected.saturating_sub(1);
                }
                KeyCode::Down | KeyCode::Tab => {
                    palette.selected = palette.selected.saturating_add(1).min(last(&palette));
                }
                KeyCode::Backspace => {
                    palette.filter.pop();
                    palette.refilter(&sources);
                }
                KeyCode::Char(typed)
                    if !key
                        .modifiers
                        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
                {
                    palette.filter.push(typed);
                    palette.refilter(&sources);
                }
                _ => continue,
            },
            Event::Paste(text) => {
                palette
                    .filter
                    .push_str(text.lines().next().unwrap_or_default());
                palette.refilter(&sources);
            }
            _ => continue,
        }
        redraw.write(RequestRedraw);
    }
}

/// Does what an entry is for, and keeps it among the recent picks.
fn activate(
    activated: On<Activate>,
    mut recent: ResMut<RecentPicks>,
    home: Res<RigHome>,
    mut input: Option<MainInput>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Activate { agent, entry } = &*activated;
    recent.record(entry, &home.0);
    match &entry.action {
        Action::Run(line) => {
            commands.trigger(RunCommand::new(*agent, line.clone()));
        }
        Action::Prefill(name) => {
            if let Some(input) = input.as_deref_mut() {
                input.clear();
                input.set(format!("{name} "));
            }
        }
        Action::Focus(shown) => commands.trigger(ShowAgent { entity: *shown }),
        Action::Describe => {
            let off = if entry.off { " (off)" } else { "" };
            let text = format!("Tool {}{off}: {}", entry.label, entry.detail);
            notices.write(Notice::info(*agent, text));
        }
    }
}

fn draw(mut panels: Query<(&Palette, &mut PanelCanvas)>, theme: Res<Theme>) {
    for (palette, mut canvas) in &mut panels {
        let items: Vec<ListItem> = palette
            .rows
            .iter()
            .map(|entry| {
                let style = if entry.off {
                    Style::new().dim()
                } else {
                    Style::new()
                };
                let off = if entry.off { " (off)" } else { "" };
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{:<8}", entry.kind.tag()), Style::new().dim()),
                    Span::styled(entry.label.clone(), style.bold()),
                    Span::styled(format!("  {}{off}", entry.detail), style.dim()),
                ]))
            })
            .collect();
        let title = match palette.filter.as_str() {
            "" => "Command palette: type to filter".to_owned(),
            filter => format!("Command palette: {filter}"),
        };
        let list = List::new(items)
            .block(theme.raised(title))
            .highlight_style(theme.selection());
        let area = canvas.area();
        let mut state = ListState::default().with_selected(Some(palette.selected));
        StatefulWidget::render(list, area, canvas.buffer_mut(), &mut state);
    }
}

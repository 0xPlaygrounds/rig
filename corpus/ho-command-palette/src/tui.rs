//! The palette in the terminal view. Ctrl-P keeps the draft aside and
//! opens a box over the screen; what is typed in the input is the filter,
//! ranked afresh from what exists each time it changes. Up and Down (or
//! Tab) select, Enter picks, and Esc (or Ctrl-P again) closes it and gives
//! the draft back. While it is open, its Enter, Esc, Up, Down and Tab are
//! global bindings, which shadow the input's own; closed, they are
//! disabled.

use rig_tui::crossterm::event::KeyCode;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::ratatui::widgets::{Block, List, ListItem, ListState, StatefulWidget};
use rig_tui::{
    AppKeyBindingsExt, Focus, KeyBinding, KeyPress, KeyPressed, PanelCanvas, Placement,
    RequestRedraw, TuiInput, TuiPanel, TuiPlugin, TuiSystems,
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

/// On the palette's entity, which is a [`TuiPanel`] while it is open.
#[derive(Component)]
struct PalettePanel;

/// The palette's state.
#[derive(Resource)]
pub(crate) struct Palette {
    /// Its panel's entity.
    panel: Entity,
    /// Its keys while open, disabled while closed.
    keys: Vec<Entity>,
    /// The draft and its cursor, kept aside while it is open.
    draft: Option<(String, usize)>,
    /// The filter the rows were ranked for.
    filter: String,
    /// The rows shown.
    pub(crate) rows: Vec<Entry>,
    /// The selected row.
    selected: usize,
}

impl Palette {
    /// Ranks the rows for `filter` and selects the first.
    fn refilter(&mut self, sources: &PaletteSources, filter: &str) {
        self.rows = sources.rank(filter);
        filter.clone_into(&mut self.filter);
        self.selected = 0;
    }

    /// Closes it, disabling its keys; the draft it kept aside.
    fn close(&mut self, commands: &mut Commands) -> Option<(String, usize)> {
        commands.entity(self.panel).remove::<TuiPanel>();
        for &key in &self.keys {
            commands.entity(key).insert(Disabled);
        }
        self.draft.take()
    }
}

pub(super) fn add(app: &mut App) {
    if !app.is_plugin_added::<TuiPlugin>() {
        app.add_plugins(TuiPlugin);
    }
    let panel = app
        .world_mut()
        .spawn((Name::new("command-palette"), PalettePanel))
        .id();
    let key = |code, label: &str| KeyBinding::new(KeyPress::key(code), label);
    // Shadows the input's Ctrl-P (the line above, which Up does too).
    app.add_key_binding(
        KeyBinding::new(KeyPress::ctrl('p'), "Command palette"),
        toggle,
    );
    let keys: Vec<Entity> = [
        app.add_key_binding(key(KeyCode::Enter, "Pick the palette's entry"), pick),
        app.add_key_binding(key(KeyCode::Esc, "Close the palette"), close),
        app.add_key_binding(key(KeyCode::Up, "The palette's entry above"), select(-1)),
        app.add_key_binding(key(KeyCode::Down, "The palette's entry below"), select(1)),
        app.add_key_binding(key(KeyCode::Tab, "The palette's next entry"), select(1)),
    ]
    .into_iter()
    .flatten()
    .collect();
    for &key in &keys {
        app.world_mut().entity_mut(key).insert(Disabled);
    }
    app.insert_resource(Palette {
        panel,
        keys,
        draft: None,
        filter: String::new(),
        rows: Vec::new(),
        selected: 0,
    })
    .add_observer(activate)
    .add_systems(Update, refilter.run_if(is_open))
    .add_systems(PostUpdate, draw.in_set(TuiSystems::Draw));
}

fn is_open(palette: Res<Palette>) -> bool {
    palette.draft.is_some()
}

/// Ctrl-P: opens the palette with an empty filter, keeping the draft
/// aside; closes it when open.
fn toggle(
    pressed: On<KeyPressed>,
    mut palette: ResMut<Palette>,
    mut input: ResMut<TuiInput>,
    sources: PaletteSources,
    mut commands: Commands,
) {
    if pressed.released {
        return;
    }
    if palette.draft.is_some() {
        restore(palette.close(&mut commands), &mut input);
        return;
    }
    let cursor = input.cursor();
    palette.draft = Some((input.take(), cursor));
    palette.refilter(&sources, "");
    let over = Placement::Over {
        width: Constraint::Percentage(70),
        height: Constraint::Percentage(60),
    };
    commands.entity(palette.panel).insert(TuiPanel::new(over));
    for &key in &palette.keys {
        commands.entity(key).remove::<Disabled>();
    }
}

/// Puts the draft kept aside back in the input.
fn restore(draft: Option<(String, usize)>, input: &mut TuiInput) {
    if let Some((text, cursor)) = draft {
        input.set(text);
        input.set_cursor(cursor);
    }
}

/// Esc: closes the palette and gives the draft back.
fn close(
    pressed: On<KeyPressed>,
    mut palette: ResMut<Palette>,
    mut input: ResMut<TuiInput>,
    mut commands: Commands,
) {
    if !pressed.released {
        restore(palette.close(&mut commands), &mut input);
    }
}

/// Up, Down and Tab: the entry `by` rows away.
fn select(by: isize) -> impl FnMut(On<KeyPressed>, ResMut<Palette>, MessageWriter<RequestRedraw>) {
    move |pressed, mut palette, mut redraw| {
        if pressed.released {
            return;
        }
        let last = palette.rows.len().saturating_sub(1);
        palette.selected = palette.selected.saturating_add_signed(by).min(last);
        redraw.write(RequestRedraw);
    }
}

/// Enter: closes the palette, gives the draft back, and picks the
/// selected entry for the agent shown.
fn pick(
    pressed: On<KeyPressed>,
    mut palette: ResMut<Palette>,
    mut input: ResMut<TuiInput>,
    sources: PaletteSources,
    mut commands: Commands,
) {
    if pressed.released {
        return;
    }
    // Keys typed in the frame of the Enter changed the filter after the
    // rows were ranked.
    if palette.filter != input.text() {
        palette.refilter(&sources, input.text());
    }
    let Some(entry) = palette.rows.get(palette.selected).cloned() else {
        return;
    };
    restore(palette.close(&mut commands), &mut input);
    if let Some(agent) = pressed.agent {
        commands.trigger(Activate { agent, entry });
    }
}

/// Does what an entry is for, and keeps it among the recent picks.
fn activate(
    activated: On<Activate>,
    mut recent: ResMut<RecentPicks>,
    home: Res<RigHome>,
    mut input: ResMut<TuiInput>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Activate { agent, entry } = &*activated;
    recent.record(entry, &home.0);
    match &entry.action {
        Action::Run(line) => {
            commands.trigger(RunCommand::new(*agent, line.clone()));
        }
        Action::Prefill(name) => input.set(format!("{name} ")),
        Action::Focus(shown) => commands.trigger(Focus { entity: *shown }),
        Action::Describe => {
            let off = if entry.off { " (off)" } else { "" };
            let text = format!("Tool {}{off}: {}", entry.label, entry.detail);
            notices.write(Notice::info(*agent, text));
        }
    }
}

/// Ranks the rows afresh, from what exists now, when the filter changed.
fn refilter(
    input: Res<TuiInput>,
    mut palette: ResMut<Palette>,
    sources: PaletteSources,
    mut redraw: MessageWriter<RequestRedraw>,
) {
    if input.is_changed() && palette.filter != input.text() {
        palette.refilter(&sources, input.text());
        redraw.write(RequestRedraw);
    }
}

fn draw(
    mut panels: Query<&mut PanelCanvas, (With<PalettePanel>, With<TuiPanel>)>,
    palette: Res<Palette>,
) {
    for mut canvas in &mut panels {
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
            "" => " Command palette: type to filter ".to_owned(),
            filter => format!(" Command palette: {filter} "),
        };
        let list = List::new(items)
            .block(Block::bordered().title(title))
            .highlight_style(Style::new().reversed());
        let area = canvas.area();
        let mut state = ListState::default().with_selected(Some(palette.selected));
        StatefulWidget::render(list, area, canvas.buffer_mut(), &mut state);
    }
}

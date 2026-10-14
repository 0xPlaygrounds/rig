//! Choices over the transcript: the pickers of [`rig_harness::pick`] and
//! the questions ([`Ask`]) of any plugin, which the view answers for
//! every agent. One is shown at a time, in the order they came; the next
//! waits until it is closed. A new [`Picker`] on the same entity starts it
//! over. The user's choice is a [`Pick`](rig_harness::pick::Pick), which the harness
//! carries out.
//!
//! A picker or question that takes several items shows a tick box on each:
//! Tab ticks the selected item and moves on. Enter answers the selected
//! one until ticks are in play (an item started ticked, or the user ticked
//! or unticked one); from then on it answers exactly the ticked ones, even
//! none. A question that takes text shows them too, so a choice goes with
//! a note: Tab ticks a choice (one only, unless it takes several), and
//! Enter on the typed answer answers the ticked choices with it as their
//! note. Right shows what a choice says beyond its label
//! ([`Choice::more`](rig_harness::ask::Choice::more)) in place of the
//! body, and Left hides it. A long body above the list scrolls with
//! Shift+Up and Shift+Down; PageUp and PageDown page the list. A picker or
//! question may bind keys of its own ([`ChoiceKeys`]).
//!
//! A question superseded by another (`SupersededBy`, how a plugin changes
//! an open [`Ask`]) is not shown; when it was the one shown, its successor
//! takes its place with what the user did carried over by choice id: the
//! ticks, what was typed and the choice selected.
//!
//! A question with initial text or a multi-line answer is answered in the
//! input instead: its text replaces the draft until Enter answers or Esc
//! refuses, and the draft comes back. A question shown beside the work
//! ([`Ask::inline`]) waits in a strip above the input, which keeps the
//! keys; Alt+1 to Alt+9 pick its choices, or answer it in the input, and
//! Alt+0 dismisses it.

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::Stylize;
use ratatui::text::{Line, Span};
use rig_ecs::stage::{Effect, Outcome, Settled, SupersededBy};
use rig_harness::ask::Ask;
use rig_harness::pick::{Chosen, Picker};
use rig_tools::shorten;
use unicode_width::UnicodeWidthStr;

use ratatui::layout::Constraint;

use super::editor::{Editor, MainEditor};
use super::panel::{PanelCanvas, Placement, TuiPanel};
use super::view::TuiView;

/// Lines a page key moves.
const PAGE: usize = 10;

/// On a [`Picker`] or an [`Ask`]: keys of its own in the view's choice
/// list, found before the view's own. A plain character's binding works
/// while a choice row is selected and nothing is typed; once the user
/// types, or selects the typed-answer row, characters type.
///
/// ```
/// use rig_tui::crossterm::event::KeyCode;
/// use rig_tui::{ChoiceAction, ChoiceKeys};
///
/// let keys = ChoiceKeys(vec![
///     (KeyCode::Char(' '), ChoiceAction::Tick),
///     (KeyCode::Char('a'), ChoiceAction::TickAll),
///     (KeyCode::Char('n'), ChoiceAction::TickNone),
///     (KeyCode::Char('e'), ChoiceAction::More),
/// ]);
/// # drop(keys);
/// ```
#[derive(Component, Clone, Debug, Default, PartialEq, Eq)]
pub struct ChoiceKeys(pub Vec<(KeyCode, ChoiceAction)>);

/// What a key of [`ChoiceKeys`] does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChoiceAction {
    /// Ticks or unticks the selected item, staying on it.
    Tick,
    /// Ticks every item shown, in a list that takes several.
    TickAll,
    /// Unticks every item.
    TickNone,
    /// Shows or hides what the selected choice says beyond its label
    /// ([`Choice::more`](rig_harness::ask::Choice::more)); a [`Picker`]'s
    /// items say nothing more.
    More,
    /// What Enter does.
    Answer,
    /// What Esc does.
    Close,
}

/// The choice shown: a [`Picker`] or an [`Ask`], what is typed to filter
/// or answer it, and the selected row.
pub(crate) struct Choosing {
    /// The picker or question.
    pub(crate) entity: Entity,
    /// What is typed.
    pub(crate) filter: String,
    /// The selected position among the visible rows.
    pub(crate) selected: usize,
    /// The positions of the items ticked, in a choice of several, in list
    /// order.
    pub(crate) ticked: Vec<usize>,
    /// The body's first row shown, which drawing keeps within the body.
    pub(crate) scroll: usize,
    /// For a question answered in the input, what was typed there before,
    /// which comes back once it is answered.
    pub(crate) draft: Option<String>,
    /// Whether what the selected choice says beyond its label is shown in
    /// place of the body.
    pub(crate) more: bool,
    /// Whether ticks are in play: an item started ticked, or the user
    /// ticked or unticked one. Enter then answers exactly the ticked.
    pub(crate) touched: bool,
}

/// A picker or a question, as the view lists it.
pub(crate) struct Listing<'a> {
    pub(crate) title: &'a str,
    pub(crate) body: &'a str,
    pub(crate) items: Vec<Item<'a>>,
    pub(crate) free_text: bool,
    /// Whether items show tick boxes, and Tab ticks them.
    pub(crate) ticks: bool,
    /// Whether several items may be ticked.
    pub(crate) multiple: bool,
    pub(crate) selected: usize,
    /// For a question answered in the input, the text it starts from.
    pub(crate) edit: Option<&'a str>,
    /// Its own keys ([`ChoiceKeys`]).
    pub(crate) keys: &'a [(KeyCode, ChoiceAction)],
}

/// An item or choice of a [`Listing`].
pub(crate) struct Item<'a> {
    pub(crate) label: &'a str,
    /// What is shown beside the label, such as a session's age.
    pub(crate) detail: Option<&'a str>,
    /// Whether it is ticked at first, in a choice of several.
    pub(crate) checked: bool,
    /// What it says beyond its label, shown when the user asks.
    pub(crate) more: Option<&'a str>,
}

impl Item<'_> {
    /// Its row, `width` wide: the label, shortened with `…` to fit, and
    /// the detail whole and dimmed at the right edge.
    pub(crate) fn line(&self, width: usize) -> Line<'static> {
        let Some(detail) = self.detail else {
            return Line::from(shorten(self.label, width));
        };
        let label = shorten(self.label, width.saturating_sub(detail.width() + 2));
        let gap = width.saturating_sub(label.width() + detail.width());
        Line::from(vec![
            Span::from(label),
            Span::from(" ".repeat(gap)),
            Span::from(detail.to_owned()).dim(),
        ])
    }
}

impl Listing<'_> {
    /// What the selected row says beyond its label, when the user asked
    /// for it.
    pub(crate) fn more(&self, choosing: &Choosing) -> Option<&str> {
        let rows = self.rows(&choosing.filter);
        match rows.get(choosing.selected) {
            Some(Row::Item(index)) if choosing.more => self.items.get(*index)?.more,
            _ => None,
        }
    }

    /// The rows that `filter` leaves: the items whose label holds every
    /// word of it, ignoring case, then, for free text, the typed answer.
    pub(crate) fn rows(&self, filter: &str) -> Vec<Row> {
        let lower = filter.to_lowercase();
        let mut rows: Vec<Row> = (self.items.iter().enumerate())
            .filter(|(_, item)| {
                let label = item.label.to_lowercase();
                lower.split_whitespace().all(|word| label.contains(word))
            })
            .map(|(index, _)| Row::Item(index))
            .collect();
        if self.free_text {
            rows.push(Row::Typed);
        }
        rows
    }
}

/// A row of a shown choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Row {
    /// The item or choice at this position.
    Item(usize),
    /// What is typed, as the answer.
    Typed,
}

/// The pickers and open questions the view can show.
#[derive(SystemParam)]
pub(crate) struct Choices<'w, 's> {
    pickers: Query<'w, 's, &'static Picker>,
    asks: Query<'w, 's, &'static Ask, (Without<Outcome>, Without<Settled>, Without<SupersededBy>)>,
    keys: Query<'w, 's, &'static ChoiceKeys>,
}

impl Choices<'_, '_> {
    /// How `entity` is shown, while it is an open picker or question.
    pub(crate) fn shown(&self, entity: Entity) -> Option<Listing<'_>> {
        let keys = self.keys.get(entity).map_or(&[][..], |keys| &keys.0);
        if let Ok(picker) = self.pickers.get(entity) {
            return Some(Listing {
                title: &picker.title,
                body: &picker.body,
                items: picker
                    .items
                    .iter()
                    .map(|item| Item {
                        label: &item.label,
                        detail: item.detail.as_deref(),
                        checked: item.checked,
                        more: None,
                    })
                    .collect(),
                free_text: picker.free_text,
                ticks: picker.multiple,
                multiple: picker.multiple,
                selected: picker.selected,
                edit: None,
                keys,
            });
        }
        Some(listing(self.asks.get(entity).ok()?, keys))
    }
}

/// How `ask` is listed, with `keys` of its own.
fn listing<'a>(ask: &'a Ask, keys: &'a [(KeyCode, ChoiceAction)]) -> Listing<'a> {
    Listing {
        title: &ask.title,
        body: &ask.body,
        items: ask
            .choices
            .iter()
            .map(|choice| Item {
                label: &choice.label,
                detail: None,
                checked: choice.checked,
                more: choice.more.as_deref(),
            })
            .collect(),
        free_text: ask.free_text,
        // A choice may go with a note.
        ticks: ask.multiple || (ask.free_text && !ask.choices.is_empty()),
        multiple: ask.multiple,
        selected: 0,
        // One beside the work is answered in the input once the user
        // chooses to type the answer.
        edit: ask
            .initial
            .as_deref()
            .or((ask.multiline || ask.inline).then_some("")),
        keys,
    }
}

/// What the user did on the question `old` shows, carried over to its
/// successor `new` on `entity` by choice id: ticks (a new choice ticked as
/// it starts), what was typed, the row selected and the draft.
fn carried(choosing: Choosing, old: &Ask, new: &Ask, entity: Entity) -> Choosing {
    let id_at = |index: &usize| old.choices.get(*index).map(|choice| choice.id.as_str());
    let ticked: Vec<&str> = choosing.ticked.iter().filter_map(id_at).collect();
    let was = |id: &str| old.choices.iter().any(|choice| choice.id == id);
    let choices = new.choices.iter().enumerate();
    let ticked = choices.filter(|(_, choice)| match was(&choice.id) {
        true => ticked.contains(&choice.id.as_str()),
        false => new.multiple && choice.checked,
    });
    let (old_rows, new_rows) = (
        listing(old, &[]).rows(&choosing.filter),
        listing(new, &[]).rows(&choosing.filter),
    );
    let row = match old_rows.get(choosing.selected) {
        Some(Row::Item(index)) => id_at(index)
            .and_then(|id| new.choices.iter().position(|choice| choice.id == id))
            .map(Row::Item),
        other => other.copied(),
    };
    let selected = row.and_then(|row| new_rows.iter().position(|known| *known == row));
    Choosing {
        entity,
        ticked: ticked.map(|(index, _)| index).collect(),
        selected: selected.unwrap_or_default(),
        ..choosing
    }
}

/// Keeps the queue of pickers and questions in the order they came, and
/// the first one shown: a new one waits, one closed or answered elsewhere
/// leaves, and a replaced picker starts over. A question shown beside the
/// work waits in its own list.
pub(crate) fn queue_choices(
    mut view: ResMut<TuiView>,
    mut input: Query<&mut Editor, With<MainEditor>>,
    pickers: Query<Entity, Added<Picker>>,
    replaced: Query<(), Changed<Picker>>,
    asks: Query<(Entity, &Effect, &Ask), Added<Ask>>,
    superseded: Query<(&Ask, &SupersededBy)>,
    every_ask: Query<&Ask>,
    choices: Choices,
) {
    let state = view.bypass_change_detection();
    // A changed question takes its place in the queue, and in front.
    let successor = |entity: &Entity| superseded.get(*entity).ok().map(|(_, by)| by.0);
    for slot in state.queue.iter_mut().chain(state.inline.iter_mut()) {
        *slot = successor(slot).unwrap_or(*slot);
    }
    let mut seen = Vec::new();
    state.queue.retain(|entity| {
        let first = !seen.contains(entity);
        seen.push(*entity);
        first
    });
    if let Some(choosing) = state.choosing.take() {
        let old = superseded.get(choosing.entity).ok();
        let new = old.and_then(|(_, by)| Some((by.0, every_ask.get(by.0).ok()?)));
        state.choosing = Some(match (old, new) {
            (Some((old, _)), Some((entity, new))) if choices.shown(entity).is_some() => {
                carried(choosing, old, new, entity)
            }
            _ => choosing,
        });
    }
    let before = state.choosing.as_ref().map(|choosing| choosing.entity);
    let inline_before = state.inline.clone();
    let mut asked: Vec<(Entity, &Effect, &Ask)> = asks.iter().collect();
    asked.sort_by_key(|(_, effect, _)| effect.id());
    let open = |entity: &Entity| choices.shown(*entity).is_some();
    state.queue.retain(open);
    state.inline.retain(open);
    let (inline, asked): (Vec<_>, Vec<_>) = asked.into_iter().partition(|(.., ask)| ask.inline);
    for (entity, ..) in inline {
        if !state.inline.contains(&entity) && open(&entity) {
            state.inline.push(entity);
        }
    }
    for entity in pickers.iter().chain(asked.into_iter().map(|(ask, ..)| ask)) {
        if !state.queue.contains(&entity) && open(&entity) {
            state.queue.push(entity);
        }
    }
    if state.inline != inline_before {
        view.set_changed();
    }
    let front = view.queue.first().copied();
    let restart = front.is_some_and(|front| replaced.contains(front));
    if front == before && !restart {
        return;
    }
    let mut main = input.single_mut().ok();
    let mut none = Editor::default();
    let input = main.as_deref_mut().unwrap_or(&mut none);
    show_first(&mut view, &choices, input);
}

/// Shows the first of the queue still open, from the start. A question
/// answered in the input takes the input, keeping the draft, which comes
/// back when it closes.
pub(crate) fn show_first(view: &mut TuiView, choices: &Choices, input: &mut Editor) {
    if let Some(Choosing {
        draft: Some(draft), ..
    }) = view.choosing.take()
    {
        input.set(draft);
    }
    view.choosing = view.queue.iter().find_map(|&entity| {
        let shown = choices.shown(entity)?;
        let draft = shown.edit.map(|initial| {
            let draft = input.take();
            input.set(initial);
            draft
        });
        let items = shown.items.iter().enumerate();
        let ticked: Vec<usize> = items
            .filter(|(_, item)| shown.multiple && item.checked)
            .map(|(index, _)| index)
            .collect();
        Some(Choosing {
            entity,
            filter: String::new(),
            selected: shown.selected.min(shown.items.len().saturating_sub(1)),
            touched: !ticked.is_empty(),
            ticked,
            scroll: 0,
            draft,
            more: false,
        })
    });
}

/// Handles a key in the shown choice; returns what closes it: `Some(None)`
/// for closing without a choice.
pub(crate) fn choice_key(
    key: KeyEvent,
    choosing: &mut Choosing,
    shown: &Listing<'_>,
) -> Option<Option<Chosen>> {
    let control = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    let rows = shown.rows(&choosing.filter);
    let last = rows.len().saturating_sub(1);
    let typing =
        !choosing.filter.is_empty() || matches!(rows.get(choosing.selected), Some(Row::Typed));
    let bound = shown.keys.iter().find(|(code, _)| {
        *code == key.code && !control && !alt && !(typing && matches!(code, KeyCode::Char(_)))
    });
    if let Some((_, action)) = bound {
        return choice_action(*action, choosing, shown, &rows);
    }
    match key.code {
        KeyCode::Up if shift => choosing.scroll = choosing.scroll.saturating_sub(1),
        KeyCode::Down if shift => choosing.scroll = choosing.scroll.saturating_add(1),
        KeyCode::Esc => return Some(None),
        KeyCode::Char('c') if control => return Some(None),
        KeyCode::Char(_) if control || key.modifiers.contains(KeyModifiers::ALT) => {}
        KeyCode::Char(character) => {
            choosing.filter.push(character);
            choosing.selected = 0;
        }
        KeyCode::Backspace => {
            choosing.filter.pop();
            choosing.selected = 0;
        }
        KeyCode::Up => choosing.selected = choosing.selected.saturating_sub(1),
        KeyCode::Down => choosing.selected = (choosing.selected + 1).min(last),
        KeyCode::PageUp => choosing.selected = choosing.selected.saturating_sub(PAGE),
        KeyCode::PageDown => choosing.selected = (choosing.selected + PAGE).min(last),
        KeyCode::Right => choosing.more = true,
        KeyCode::Left => choosing.more = false,
        KeyCode::Tab if shown.ticks => {
            if tick(choosing, shown, &rows) {
                choosing.selected = (choosing.selected + 1).min(last);
            }
        }
        KeyCode::Enter => return answer(choosing, shown, &rows),
        _ => {}
    }
    None
}

/// Does `action` in the shown choice; returns what closes it, as
/// [`choice_key`] does.
fn choice_action(
    action: ChoiceAction,
    choosing: &mut Choosing,
    shown: &Listing<'_>,
    rows: &[Row],
) -> Option<Option<Chosen>> {
    match action {
        ChoiceAction::Tick if shown.ticks => {
            tick(choosing, shown, rows);
        }
        ChoiceAction::TickAll if shown.multiple => {
            let items = rows.iter().filter_map(|row| match row {
                Row::Item(index) => Some(*index),
                Row::Typed => None,
            });
            choosing.ticked = items.collect();
            choosing.ticked.sort_unstable();
            choosing.touched = true;
        }
        ChoiceAction::TickNone if shown.ticks => {
            choosing.ticked.clear();
            choosing.touched = true;
        }
        ChoiceAction::More => choosing.more = !choosing.more,
        ChoiceAction::Answer => return answer(choosing, shown, rows),
        ChoiceAction::Close => return Some(None),
        ChoiceAction::Tick | ChoiceAction::TickAll | ChoiceAction::TickNone => {}
    }
    None
}

/// Ticks or unticks the selected item; whether one is selected.
fn tick(choosing: &mut Choosing, shown: &Listing<'_>, rows: &[Row]) -> bool {
    let Some(Row::Item(index)) = rows.get(choosing.selected) else {
        return false;
    };
    match choosing.ticked.binary_search(index) {
        Ok(at) => {
            choosing.ticked.remove(at);
        }
        Err(_) if !shown.multiple => choosing.ticked = vec![*index],
        Err(at) => choosing.ticked.insert(at, *index),
    }
    choosing.touched = true;
    true
}

/// What Enter answers: the selected row, or once ticks are in play in a
/// choice of several, exactly the ticked items; the typed answer with the
/// ticked items as its note's.
fn answer(choosing: &Choosing, shown: &Listing<'_>, rows: &[Row]) -> Option<Option<Chosen>> {
    match rows.get(choosing.selected)? {
        Row::Item(_) if choosing.touched && shown.multiple => {
            Some(Some(Chosen::items(choosing.ticked.clone())))
        }
        Row::Item(index) => Some(Some(Chosen::item(*index))),
        Row::Typed if choosing.filter.trim().is_empty() => None,
        Row::Typed => {
            let chosen = Chosen::items(choosing.ticked.clone());
            Some(Some(chosen.with_text(choosing.filter.trim())))
        }
    }
}

/// The strip that shows the question asked beside the work, if any
/// ([`Ask::inline`]), with the keys that answer it: one row while one
/// shows, none otherwise, in a panel at the top or the bottom. The default
/// screen's is above the input.
#[derive(Component, Clone, Copy, Debug, Default)]
#[require(TuiPanel = TuiPanel::new(Placement::Bottom(Constraint::Length(1))))]
pub struct InlineStrip;

/// Draws the question shown beside the work in each strip with room.
pub(crate) fn draw_inline_strips(
    mut strips: Query<&mut PanelCanvas, With<InlineStrip>>,
    view: Res<TuiView>,
    choices: Choices,
) {
    let Some(ask) = view.inline.first().and_then(|&ask| choices.shown(ask)) else {
        return;
    };
    for mut canvas in &mut strips {
        canvas.render(inline_line(&ask));
    }
}

/// The strip of a question shown beside the work: its title and the keys
/// that answer it.
fn inline_line(ask: &Listing<'_>) -> Line<'static> {
    let mut spans = vec![
        Span::from("? ").yellow().bold(),
        Span::from(ask.title.to_owned()).bold(),
        Span::from(" "),
    ];
    let typed = ask.free_text.then_some("Answer…");
    let labels = ask.items.iter().map(|item| item.label).chain(typed);
    for (index, label) in labels.take(9).enumerate() {
        spans.push(Span::from(format!(" [Alt+{} {label}]", index + 1)).cyan());
    }
    spans.push(Span::from(" [Alt+0 Dismiss]").dim());
    Line::from(spans)
}

#[cfg(test)]
mod tests;

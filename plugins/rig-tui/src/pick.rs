//! Pickers and questions over the transcript.

use std::io;

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::Stylize;
use ratatui::text::{Line, Span};
use rig_core::conversation::pairing::safe_start;
use rig_core::conversation::{self, Aside, Author, MessageId, TURN, TurnEnd, TurnRecord};
use rig_core::message::{AssistantContent, Message, ToolResultContent, UserContent};
use rig_ecs::stage::{Effect, Outcome, Settled, SupersededBy};
use rig_harness::ask::Ask;
use rig_harness::pick::{Chosen, Highlighted, PickPreview, Picker};
use rig_tools::shorten;
use unicode_width::UnicodeWidthStr;

use ratatui::layout::Constraint;

use super::editor::{Editor, MainEditor};
use super::pane::{TranscriptPane, TranscriptSource};
use super::panel::{PanelCanvas, Placement, TuiPanel};
use super::view::TuiView;
use super::view::{ItemRow, ScrollTo, TranscriptItem, TranscriptSlot};

/// Lines a page key moves.
const PAGE: usize = 10;

/// About how much of a conversation a preview lays out, in bytes of what
/// its messages draw: a few screens, laid out in tens of milliseconds.
const PREVIEW_BYTES: usize = 64 * 1024;

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
    /// ([`Choice::more`](rig_harness::ask::Choice::more), or a [`Picker`]
    /// item's [`more`](rig_harness::pick::PickItem::more), else its
    /// detail).
    More,
    /// What Enter does.
    Answer,
    /// What Esc does.
    Close,
}

/// What the user did in the [`Picker`] or [`Ask`] the view shows, on its
/// entity while shown: the filter or answer typed, the row selected and
/// what is ticked. The view shows every agent's and plugin's, one at a
/// time, in the order they came. A new `Picker` on the entity refills the
/// list and keeps all three (the selected row found again by its command,
/// else its label). A question superseded by another (`SupersededBy`)
/// hands them to its successor, by choice id. The item highlighted is
/// [`Highlighted`] on the entity, and a [`PickPreview`] there is drawn
/// beside the list.
///
/// Keys:
/// - **Tab** ticks the selected row in one that takes several (or the
///   choice a typed note goes with). Once anything is ticked, Enter
///   answers exactly the ticked rows.
/// - **Right** and **Left** show and hide a choice's or an item's
///   [`more`](rig_harness::ask::Choice::more).
/// - **Shift+Up** and **Shift+Down** scroll a long body; **PageUp** and
///   **PageDown** page the list. A picker adds keys with [`ChoiceKeys`].
/// - A question with initial text or a multi-line answer is answered in
///   the input; Esc refuses it and brings the draft back.
/// - An [`Ask::inline`] question waits in a strip above the input:
///   Alt+1 to Alt+9 pick, and Alt+0 dismisses.
#[derive(Component, Clone, Debug, Default, PartialEq, Eq)]
pub struct PickerInput {
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
    /// The item highlighted last, by its command (a choice's id) and its
    /// label, which a picker filled again selects again.
    pub(crate) picked: Option<(String, String)>,
}

impl PickerInput {
    /// What is typed, which filters the items or answers.
    pub fn filter(&self) -> &str {
        &self.filter
    }

    /// The items ticked, by their position in the list, in a choice of
    /// several.
    pub fn ticked(&self) -> &[usize] {
        &self.ticked
    }

    /// Whether it is a question answered in the input rather than over the
    /// transcript.
    pub fn in_input(&self) -> bool {
        self.draft.is_some()
    }
}

/// A picker or a question, as the view lists it.
pub(crate) struct Listing<'a> {
    pub(crate) title: &'a str,
    pub(crate) body: &'a str,
    pub(crate) items: Vec<Item<'a>>,
    pub(crate) free_text: bool,
    /// Whether what is typed is a secret, drawn hidden.
    pub(crate) secret: bool,
    /// Whether items show tick boxes, and Tab ticks them.
    pub(crate) ticks: bool,
    /// Whether several items may be ticked.
    pub(crate) multiple: bool,
    pub(crate) selected: usize,
    /// For a question answered in the input, the text it starts from.
    pub(crate) edit: Option<&'a str>,
    /// Its own keys ([`ChoiceKeys`]).
    pub(crate) keys: &'a [(KeyCode, ChoiceAction)],
    /// Whether typing finds items by their more too: a picker's, such as a
    /// session's whole folder, but not a question's, where it may be a
    /// call's arguments and typing writes an answer.
    pub(crate) finds_more: bool,
}

/// An item or choice of a [`Listing`].
pub(crate) struct Item<'a> {
    /// What names it: a picker item's command, a choice's id.
    pub(crate) id: &'a str,
    pub(crate) label: &'a str,
    /// What is shown beside the label, such as a session's age.
    pub(crate) detail: Option<&'a str>,
    /// Whether it is ticked at first, in a choice of several.
    pub(crate) checked: bool,
    /// What it says beyond its label, shown when the user asks.
    pub(crate) more: Option<&'a str>,
}

impl Item<'_> {
    /// Its row, `width` wide: the label, shortened with `…` to end before
    /// `column`, then from `column` the detail dimmed, shortened to fit.
    pub(crate) fn line(&self, width: usize, column: usize) -> Line<'static> {
        let Some(detail) = self.detail else {
            return Line::from(shorten(self.label, width));
        };
        let label = shorten(self.label, column.saturating_sub(2));
        let gap = column.saturating_sub(label.width());
        let detail = shorten(detail, width.saturating_sub(label.width() + gap));
        Line::from(vec![
            Span::from(label),
            Span::from(" ".repeat(gap)),
            Span::from(detail).dim(),
        ])
    }
}

/// The column the details of `items` start at in rows `width` wide: the
/// one that shows the most labels whole, each counted twice, and details
/// whole, the rightmost of those. It keeps a quarter of the row for the
/// labels and a third for the details, and is never past two after the
/// widest label.
pub(crate) fn detail_column(items: &[Item<'_>], width: usize) -> usize {
    let sizes: Vec<(usize, usize)> = (items.iter())
        .filter_map(|item| Some((item.label.width(), item.detail?.width())))
        .collect();
    let widest = sizes
        .iter()
        .map(|(label, _)| label + 2)
        .max()
        .unwrap_or_default();
    let last = widest.min(width.saturating_sub(width / 3));
    let whole = |column: usize| -> usize {
        let shown = |(label, detail): &(usize, usize)| {
            2 * usize::from(label + 2 <= column)
                + usize::from(*detail <= width.saturating_sub(column))
        };
        sizes.iter().map(shown).sum()
    };
    (widest.min(width / 4)..=last)
        .max_by_key(|column| whole(*column))
        .unwrap_or(last)
}

impl Listing<'_> {
    /// What is typed, as drawn: a secret as one dot per character.
    pub(crate) fn typed(&self, filter: &str) -> String {
        match self.secret {
            true => "•".repeat(filter.chars().count()),
            false => filter.to_owned(),
        }
    }

    /// What the selected row says beyond its label, when the user asked
    /// for it.
    pub(crate) fn more(&self, choosing: &PickerInput) -> Option<&str> {
        let rows = self.rows(&choosing.filter);
        match rows.get(choosing.selected) {
            Some(Row::Item(index)) if choosing.more => self.items.get(*index)?.more,
            _ => None,
        }
    }

    /// The rows that `filter` leaves: the items whose label, detail or, as
    /// [`finds_more`](Self::finds_more) says, more holds every word of it,
    /// ignoring case, those whose label does first; then, for free text, the
    /// typed answer.
    pub(crate) fn rows(&self, filter: &str) -> Vec<Row> {
        let lower = filter.to_lowercase();
        let holds = |said: &[Option<&str>]| {
            let said = said.iter().flatten().copied().collect::<Vec<_>>();
            let said = said.join("\n").to_lowercase();
            lower.split_whitespace().all(|word| said.contains(word))
        };
        let found = (self.items.iter().enumerate()).filter(|(_, item)| {
            let more = item.more.filter(|_| self.finds_more);
            holds(&[Some(item.label), item.detail, more])
        });
        let (named, beside): (Vec<_>, Vec<_>) =
            found.partition(|(_, item)| holds(&[Some(item.label)]));
        let mut rows: Vec<Row> = (named.into_iter().chain(beside))
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
                        id: &item.command,
                        label: &item.label,
                        detail: item.detail.as_deref(),
                        checked: item.checked,
                        more: item.more.as_deref().or(item.detail.as_deref()),
                    })
                    .collect(),
                free_text: picker.free_text,
                secret: picker.secret,
                ticks: picker.multiple,
                multiple: picker.multiple,
                selected: picker.selected,
                edit: None,
                keys,
                finds_more: true,
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
                id: &choice.id,
                label: &choice.label,
                detail: None,
                checked: choice.checked,
                more: choice.more.as_deref(),
            })
            .collect(),
        free_text: ask.free_text,
        secret: false,
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
        finds_more: false,
    }
}

/// What the user did on the question `old` shows, carried over to its
/// successor `new` by choice id: ticks (a new choice ticked as it starts),
/// what was typed, the row selected and the draft.
fn carried(choosing: PickerInput, old: &Ask, new: &Ask) -> PickerInput {
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
    PickerInput {
        ticked: ticked.map(|(index, _)| index).collect(),
        selected: selected.unwrap_or_default(),
        ..choosing
    }
}

/// The pickers and questions shown, with what the user did in them: one at
/// most.
pub(crate) type Showing<'w, 's> = Query<'w, 's, (Entity, &'static mut PickerInput)>;

/// Keeps the queue of pickers and questions in the order they came, and
/// the first one shown: a new one waits, one closed or answered elsewhere
/// leaves, a changed question's successor takes its place with what the
/// user did, and a picker filled again keeps it. A question shown beside
/// the work waits in its own list.
pub(crate) fn queue_choices(
    mut view: ResMut<TuiView>,
    mut input: Query<&mut Editor, With<MainEditor>>,
    (pickers, filled): (Query<Entity, Added<Picker>>, Query<(), Changed<Picker>>),
    asks: Query<(Entity, &Effect, &Ask), Added<Ask>>,
    (superseded, every_ask): (Query<(&Ask, &SupersededBy)>, Query<&Ask>),
    mut showing: Showing,
    choices: Choices,
    mut commands: Commands,
) {
    // A question taken away while in the input, never closed, takes its
    // answer with it and gives back the draft, as one refused does.
    if showing.iter().all(|(_, shown)| shown.draft.is_none())
        && let Ok(mut main) = input.single_mut()
        && main.answering()
    {
        main.drop_question();
    }
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
    let mut before = None;
    for (entity, shown) in &showing {
        before = Some(entity);
        let old = superseded.get(entity).ok();
        let new = old.and_then(|(_, by)| Some((by.0, every_ask.get(by.0).ok()?)));
        if let (Some((old, _)), Some((next, new))) = (old, new)
            && choices.shown(next).is_some()
        {
            commands
                .entity(entity)
                .try_remove::<(PickerInput, Highlighted)>();
            commands
                .entity(next)
                .try_insert(carried(shown.clone(), old, new));
            before = Some(next);
        }
    }
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
    if front == before {
        let refilled = front.filter(|front| filled.contains(*front));
        let shown = refilled.and_then(|front| Some((showing.get_mut(front).ok()?, front)));
        if let Some(((_, mut state), front)) = shown
            && let Some(listing) = choices.shown(front)
        {
            refill(&mut state, &listing);
        }
        return;
    }
    let mut main = input.single_mut().ok();
    let mut none = Editor::default();
    let input = main.as_deref_mut().unwrap_or(&mut none);
    show_first(&view, &choices, (&showing, input), &mut commands);
}

/// Where the terminal view takes in this frame's pickers and questions, in
/// `Update` after the agent loop, and finds a picker's selection again
/// when it was filled again. A plugin that fills a picker again in `Update`
/// does so before it, so the next key acts on the rows shown.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChoiceSystems;

/// Selects again, in `shown` filled again, the item `state` had selected,
/// by its command, else its label, among the rows what is typed leaves;
/// else the row it was on, as far as there are rows.
fn refill(state: &mut PickerInput, shown: &Listing<'_>) {
    let rows = shown.rows(&state.filter);
    let item = |row: &Row| match row {
        Row::Item(index) => shown.items.get(*index),
        Row::Typed => None,
    };
    let picked = state.picked.as_ref();
    let by = |same: &dyn Fn(&Item<'_>, &(String, String)) -> bool| {
        picked.and_then(|picked| {
            rows.iter()
                .position(|row| item(row).is_some_and(|item| same(item, picked)))
        })
    };
    let found = by(&|item, (id, _)| !id.is_empty() && item.id == id)
        .or_else(|| by(&|item, (_, label)| item.label == label));
    let selected = found.unwrap_or(state.selected);
    state.selected = selected.min(rows.len().saturating_sub(1));
    state.ticked.retain(|index| *index < shown.items.len());
}

/// Shows the first of the queue still open, from the start, in place of
/// the one `showing`. A question answered in the input takes the input,
/// keeping the draft, which comes back when it closes.
pub(crate) fn show_first(
    view: &TuiView,
    choices: &Choices,
    (showing, input): (&Showing, &mut Editor),
    commands: &mut Commands,
) {
    for (entity, shown) in showing {
        if let Some(draft) = &shown.draft {
            input.take_answer();
            input.set(draft.clone());
        }
        commands
            .entity(entity)
            .try_remove::<(PickerInput, Highlighted)>();
    }
    let first = view.queue.iter().find_map(|&entity| {
        let shown = choices.shown(entity)?;
        let draft = shown.edit.map(|initial| input.answer(initial));
        let items = shown.items.iter().enumerate();
        let ticked: Vec<usize> = items
            .filter(|(_, item)| shown.multiple && item.checked)
            .map(|(index, _)| index)
            .collect();
        let state = PickerInput {
            filter: String::new(),
            selected: shown.selected.min(shown.items.len().saturating_sub(1)),
            touched: !ticked.is_empty(),
            ticked,
            scroll: 0,
            draft,
            more: false,
            picked: None,
        };
        Some((entity, state))
    });
    if let Some((entity, state)) = first {
        commands.entity(entity).try_insert(state);
    }
}

/// Keeps [`Highlighted`] on the picker or question shown over the
/// transcript at the item its selected row is, and none on a typed answer,
/// and notes that item for a picker filled again, unless it was filled
/// since: its selection is found again first.
pub(crate) fn mark_highlighted(
    mut showing: Query<(Entity, &mut PickerInput, Option<&Highlighted>)>,
    filled: Query<(), Changed<Picker>>,
    choices: Choices,
    mut commands: Commands,
) {
    for (entity, mut state, highlighted) in &mut showing {
        let Some(shown) = choices.shown(entity).filter(|_| state.draft.is_none()) else {
            continue;
        };
        let rows = shown.rows(&state.filter);
        let index = match rows.get(state.selected) {
            Some(Row::Item(index)) => Some(*index),
            _ => None,
        };
        let item = index.and_then(|index| shown.items.get(index));
        let picked = item.map(|item| (item.id.to_owned(), item.label.to_owned()));
        if state.picked != picked && !filled.contains(entity) {
            state.bypass_change_detection().picked = picked;
        }
        match index {
            Some(index) if highlighted != Some(&Highlighted(index)) => {
                commands.entity(entity).try_insert(Highlighted(index));
            }
            None if highlighted.is_some() => {
                commands.entity(entity).try_remove::<Highlighted>();
            }
            _ => {}
        }
    }
}

/// Handles a key in the shown choice; returns what closes it: `Some(None)`
/// for closing without a choice.
pub(crate) fn choice_key(
    key: KeyEvent,
    choosing: &mut PickerInput,
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
    choosing: &mut PickerInput,
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
fn tick(choosing: &mut PickerInput, shown: &Listing<'_>, rows: &[Row]) -> bool {
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
fn answer(choosing: &PickerInput, shown: &Listing<'_>, rows: &[Row]) -> Option<Option<Chosen>> {
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

/// Marks the transcript pane that draws a picker's conversation preview.
#[derive(Component)]
pub(crate) struct PreviewPane;

/// Keeps a transcript pane over the conversation a picker shown previews
/// ([`PickPreview::Conversation`]) in the right half of the picker, above
/// its keys, scrolled to its entry, and none once the picker is closed or
/// previews something else.
pub(crate) fn follow_previews(
    showing: Query<(Entity, &PickerInput, Ref<PickPreview>)>,
    panes: Query<(Entity, &ChildOf), With<PreviewPane>>,
    mut commands: Commands,
) {
    let previewing = |picker: Entity| {
        let shown = showing.get(picker).ok();
        let shown = shown.filter(|(_, state, _)| !state.in_input());
        shown.filter(|(.., preview)| matches!(**preview, PickPreview::Conversation { .. }))
    };
    for (pane, of) in &panes {
        if previewing(of.parent()).is_none() {
            commands.entity(pane).despawn();
        }
    }
    for (picker, _, preview) in &showing {
        let PickPreview::Conversation { conversation, at } = &*preview else {
            continue;
        };
        let pane = panes.iter().find(|(_, of)| of.parent() == picker);
        if previewing(picker).is_none() || pane.is_some() && !preview.is_changed() {
            continue;
        }
        let source = TranscriptSource::Value(window(conversation, *at));
        let shown = TranscriptPane::new(source).titled("preview");
        let pane = match pane {
            Some((pane, _)) => commands.entity(pane).insert(shown).id(),
            None => {
                let within = Placement::Within {
                    panel: picker,
                    part: 1,
                    of: 2,
                };
                let spawned = (shown, TuiPanel::new(within), PreviewPane, ChildOf(picker));
                commands.spawn(spawned).id()
            }
        };
        let item = at.map(TranscriptItem::Message);
        commands.trigger(ScrollTo {
            pane,
            item,
            row: ItemRow::Top(0),
        });
    }
}

/// The part of `conversation` a preview lays out, its entries keeping
/// their ids: `at`'s entry, else the last, then the entries after it, then
/// those before it, until they pass [`PREVIEW_BYTES`]. Its edges may take in
/// more, so no call is parted from its results, and no stopped call from
/// the entry its stop is drawn under. Turn records stay; other asides stay
/// when drawn within it.
fn window(
    conversation: &conversation::Conversation,
    at: Option<MessageId>,
) -> conversation::Conversation {
    let messages = conversation.messages();
    let at = at.and_then(|at| conversation.index_of(at));
    let first = at.unwrap_or(messages.len().saturating_sub(1));
    let mut size = messages.get(first).map_or(0, |message| drawn_size(message));
    let mut within = |message: &Message| {
        let fits = size < PREVIEW_BYTES;
        if fits {
            size += drawn_size(message);
        }
        fits
    };
    let after = messages.iter().skip(first.saturating_add(1));
    let after = after.take_while(|message| within(message)).count();
    let before = messages.iter().take(first).rev();
    let before = before.take_while(|message| within(message)).count();
    let mut start = first - before;
    let mut end = (first + 1 + after).min(messages.len());
    while start > 0 && safe_start(messages, start) != start {
        start -= 1;
    }
    end = safe_start(messages, end);
    // A stop past the end takes the window to it when the reply whose
    // calls it answered, the last of its turn asking for one, is in it.
    let stopped = |turn: TurnRecord, end: usize| {
        let Some(TurnEnd::Stopped {
            at: Some(stop),
            calls,
            ..
        }) = turn.end
        else {
            return None;
        };
        let stop = conversation.index_of(stop).filter(|stop| *stop >= end)?;
        let from = conversation.start_of(turn.start).min(stop);
        let asked =
            (from..=stop)
                .rev()
                .find(|at| match messages.get(*at).map(|message| &**message) {
                    Some(Message::Assistant(reply)) => {
                        reply.tool_calls().any(|call| calls.contains(&call.id))
                    }
                    _ => false,
                })?;
        (start..end).contains(&asked).then_some(stop)
    };
    while let Some(stop) = conversation
        .turns()
        .find_map(|(_, turn)| stopped(turn, end))
    {
        end = safe_start(messages, stop.saturating_add(1));
    }
    let mut shown = conversation.view();
    shown.splice(end.., [], Author::rig());
    shown.splice(..start, [], Author::rig());
    for (id, aside) in conversation.asides() {
        if aside.kind == TURN {
            continue;
        }
        match TranscriptSlot::of_aside(aside, conversation) {
            TranscriptSlot::Top if start == 0 => {}
            TranscriptSlot::After(index) if index.saturating_add(1) == start => {
                let top = Aside {
                    after: None,
                    ..aside.clone()
                };
                shown.replace_aside(id, top);
            }
            TranscriptSlot::After(index) | TranscriptSlot::Inside(index, _)
                if (start..end).contains(&index) => {}
            _ => {
                shown.remove_aside(id);
            }
        }
    }
    shown
}

/// About how much of `message` is drawn, in bytes: its text, reasoning,
/// calls' arguments and text or JSON results. Media, a provider's own
/// data and system messages draw a row or nothing.
fn drawn_size(message: &Message) -> usize {
    match message {
        Message::System { .. } => 0,
        Message::User { content } => content
            .iter()
            .map(|item| match item {
                UserContent::Text(text) => text.text.len(),
                UserContent::ToolResult(result) => result
                    .content
                    .iter()
                    .map(|part| match part {
                        ToolResultContent::Text(text) => text.text.len(),
                        ToolResultContent::Json { value } => json_size(value),
                        _ => 0,
                    })
                    .sum(),
                _ => 0,
            })
            .sum(),
        Message::Assistant(reply) => reply
            .content
            .iter()
            .map(|item| match item {
                AssistantContent::Text(text) => text.text.len(),
                AssistantContent::Reasoning(reasoning) => reasoning.text.len(),
                AssistantContent::ToolCall(call) => json_size(&call.function.arguments),
                _ => 0,
            })
            .sum(),
    }
}

/// How many bytes `value` takes as JSON, counted without writing it.
fn json_size(value: &impl serde::Serialize) -> usize {
    struct Count(usize);
    impl io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0 += bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut count = Count(0);
    serde_json::to_writer(&mut count, value).map_or(0, |()| count.0)
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

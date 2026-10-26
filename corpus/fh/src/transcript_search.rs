//! FH-20: transcript search, as in `less`.
//!
//! Ctrl+F opens a search bar under the transcript (a [`TuiPanel`] that
//! takes the keys), and so does `/` while the transcript has the keys
//! ([`KeyScope::Transcript`]: an item of it is selected). Otherwise `/`
//! starts a slash command, as it does without the plugin.
//! What is typed searches as it is typed, ignoring case unless the query
//! has a capital letter; `re:` makes the rest a regular expression. It
//! searches the transcript as the view lays it out ([`TranscriptLayout`]),
//! at the width it draws at: each line as its render systems draw it, its
//! wrapped rows rejoined, the notices where they are drawn and the reply
//! streaming in.
//!
//! The matches the view draws are lit in the transcript, each where the
//! search found it ([`Highlight::spans`]), the one shown in a style of its
//! own, across the rows a wrap split one into. Moving to one scrolls the
//! view to its row ([`ScrollTo`] with [`ItemRow::Top`]). Enter takes it; `n` (or Enter again) and `N`
//! (or Shift+Enter) move, wrapping around, and the bar counts (`3/17`).
//! Tab searches every agent, naming the agent of the match and showing it. Esc
//! closes the bar and unlights the matches; unless a match was taken, it
//! also scrolls back to where the view was. The bar takes the keys it does, so none of them reaches the
//! input or its bindings: its Esc does not also stop the agent. The search
//! only reads: it never changes a conversation.

use std::collections::HashMap;
use std::ops::Range;

use regex::{Regex, RegexBuilder};
use rig_harness::prelude::*;
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::{
    AppKeyBindingsExt, DrawnLine, DrawnTranscript, Highlight, InputSystems, ItemRow, KeyBinding,
    KeyFocus, KeyInput, KeyPress, KeyPressed, KeyScope, LineSpan, MainPane, PanelCanvas,
    PendingInput, Placement, RequestRedraw, ScrollTo, ShowAgent, TranscriptItem, TranscriptLayout,
    TranscriptView, TuiPanel, TuiPlugin, TuiSystems,
};

/// Whose transcript is searched.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scope {
    /// The agent shown.
    #[default]
    Shown,
    /// Every agent.
    All,
}

/// A match.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    /// The agent whose transcript it is in.
    pub agent: Entity,
    /// The items it is drawn in, outermost first, such as a message and
    /// a call in it; none for rows of no item, such as the reply
    /// streaming in.
    pub within: Vec<TranscriptItem>,
    /// Which of its item's matches it is, top to bottom.
    pub nth: usize,
    /// The row it starts in, in its item (in the transcript with no item).
    pub row: usize,
    /// The line it is in, among its item's lines (the transcript's with
    /// no item), as [`DrawnTranscript::lines`] gives them.
    pub line: usize,
    /// Its bytes in that line's text.
    pub bytes: Range<usize>,
}

impl Hit {
    /// The innermost item it is drawn in, such as a call rather than its
    /// message.
    pub fn item(&self) -> Option<&TranscriptItem> {
        self.within.last()
    }

    /// What tells it apart across searches.
    fn key(&self) -> MatchKey {
        (self.agent, self.item().cloned(), self.nth)
    }

    /// Where the view lights it.
    fn span(&self) -> LineSpan {
        let (item, line, bytes) = (self.item().cloned(), self.line, self.bytes.clone());
        LineSpan { item, line, bytes }
    }
}

/// The query as a regular expression, or `None` for an empty one: `re:`
/// starts a regular expression, anything else is matched as typed; case is
/// ignored unless the query has a capital letter.
pub fn compile(query: &str) -> Result<Option<Regex>, regex::Error> {
    let (pattern, literal) = match query.strip_prefix("re:") {
        Some(pattern) => (pattern.to_owned(), false),
        None => (regex::escape(query), true),
    };
    if pattern.is_empty() {
        return Ok(None);
    }
    let typed = if literal { query } else { &pattern };
    // A capital after `\` is a class such as `\W`, not a capital letter.
    let mut escaped = false;
    let capital = typed.chars().any(|letter| {
        let capital = !escaped && letter.is_uppercase();
        escaped = !escaped && letter == '\\';
        capital
    });
    RegexBuilder::new(&pattern)
        .case_insensitive(!capital)
        .build()
        .map(Some)
}

/// The non-empty matches of `regex` in `agent`'s transcript `laid_out`,
/// line by line, top to bottom: each line as it was before the view
/// wrapped it, so a match a wrap split is found, at the row it starts in.
pub fn find(regex: &Regex, agent: Entity, laid_out: &DrawnTranscript) -> Vec<Hit> {
    let lines: Vec<DrawnLine> = laid_out.lines().collect();
    let mut counted: HashMap<Option<TranscriptItem>, usize> = HashMap::new();
    let mut hits = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        for found in regex
            .find_iter(&line.text)
            .filter(|found| !found.is_empty())
        {
            let at = line.row_at(found.start());
            let within: Vec<TranscriptItem> = laid_out
                .items()
                .iter()
                .filter(|(_, rows)| rows.contains(&at))
                .map(|(item, _)| item.clone())
                .collect();
            let item = within.last().cloned();
            let rows = item.as_ref().and_then(|item| laid_out.rows_of(item));
            let first = rows.map_or(0, |rows| rows.start);
            let before = lines.partition_point(|line| line.rows.start < first);
            let count = counted.entry(item).or_default();
            hits.push(Hit {
                agent,
                within,
                nth: *count,
                row: at.saturating_sub(first),
                line: index.saturating_sub(before),
                bytes: found.range(),
            });
            *count += 1;
        }
    }
    hits
}

/// The open search bar, on its [`TuiPanel`]'s entity.
#[derive(Component, Debug)]
pub struct TranscriptSearch {
    /// What is typed.
    pub query: String,
    /// Whose transcript is searched.
    pub scope: Scope,
    regex: Option<Regex>,
    error: bool,
    hits: Vec<Hit>,
    current: Option<usize>,
    /// Typing edits the query; after Enter, `n` and `N` move.
    editing: bool,
    /// Whether a match was taken with Enter (or `n`, `N`).
    taken: bool,
    /// The query or scope changed since the last search.
    dirty: bool,
    /// Whether to search again, with the layout, this frame.
    pending: bool,
    /// The agent shown and where the view was scrolled when it opened.
    back: Option<(Entity, Option<(TranscriptItem, usize)>)>,
    /// The match the view was last scrolled to, and its row then.
    went: Option<(MatchKey, usize)>,
    /// The highlights it lit.
    lit: Vec<Highlight>,
}

/// What the bar did with a key.
enum Pressed {
    /// Nothing: it is not one of the bar's.
    Other,
    /// Closed the bar.
    Esc,
    /// Typed, moved or switched the scope.
    Done,
}

/// What tells a match apart across searches.
type MatchKey = (Entity, Option<TranscriptItem>, usize);

impl TranscriptSearch {
    fn new(back: Option<(Entity, Option<(TranscriptItem, usize)>)>) -> Self {
        Self {
            query: String::new(),
            scope: Scope::Shown,
            regex: None,
            error: false,
            hits: Vec::new(),
            current: None,
            editing: true,
            taken: false,
            dirty: true,
            pending: false,
            back,
            went: None,
            lit: Vec::new(),
        }
    }

    /// Every match, in transcript order.
    pub fn hits(&self) -> &[Hit] {
        &self.hits
    }

    /// The match shown.
    pub fn current(&self) -> Option<&Hit> {
        self.hits.get(self.current?)
    }

    /// The count the bar shows, such as `3/17`, or `bad regex`.
    pub fn counter(&self) -> String {
        if self.error {
            return "bad regex".to_owned();
        }
        let at = self.current.map_or(0, |current| current + 1);
        format!("{at}/{}", self.hits.len())
    }

    /// What tells match `at` apart.
    fn key(&self, at: Option<usize>) -> Option<MatchKey> {
        self.hits.get(at?).map(Hit::key)
    }

    /// The regular expression of the query, compiled again when it
    /// changed.
    fn compiled(&mut self) -> Option<Regex> {
        if self.dirty {
            self.dirty = false;
            let compiled = compile(&self.query);
            self.error = compiled.is_err();
            self.regex = compiled.ok().flatten();
        }
        self.regex.clone()
    }

    /// Takes the matches of a new search, keeping the match shown where it
    /// still is; else, while typing, the latest match.
    fn found(&mut self, hits: Vec<Hit>) {
        let was = self.key(self.current);
        self.hits = hits;
        let kept = (0..self.hits.len()).find(|&at| was.is_some() && self.key(Some(at)) == was);
        let last = self.hits.len().checked_sub(1);
        self.current = match kept {
            Some(kept) if !self.editing => Some(kept),
            _ if self.editing => last,
            _ => self.current.zip(last).map(|(at, last)| at.min(last)),
        };
    }

    /// Does `event` in the bar.
    fn press(&mut self, event: &Event) -> Pressed {
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                let shift = key.modifiers.contains(KeyModifiers::SHIFT);
                let plain = !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
                match key.code {
                    KeyCode::Esc => return Pressed::Esc,
                    KeyCode::Enter if shift => self.step(-1),
                    KeyCode::Enter if self.editing => {
                        self.editing = false;
                        self.taken = self.current.is_some();
                    }
                    KeyCode::Enter => self.step(1),
                    KeyCode::Tab => {
                        self.scope = match self.scope {
                            Scope::Shown => Scope::All,
                            Scope::All => Scope::Shown,
                        };
                        self.dirty = true;
                    }
                    KeyCode::Backspace => {
                        self.editing = true;
                        self.query.pop();
                        self.dirty = true;
                    }
                    KeyCode::Char('n') if plain && !self.editing => self.step(1),
                    KeyCode::Char('N') if plain && !self.editing => self.step(-1),
                    KeyCode::Char(typed) if plain => {
                        self.editing = true;
                        self.query.push(typed);
                        self.dirty = true;
                    }
                    _ => return Pressed::Other,
                }
            }
            Event::Paste(text) => {
                self.editing = true;
                self.query.push_str(text.lines().next().unwrap_or_default());
                self.dirty = true;
            }
            _ => return Pressed::Other,
        }
        Pressed::Done
    }

    /// Takes the match `by` matches on, wrapping around.
    fn step(&mut self, by: isize) {
        let count = self.hits.len();
        if count == 0 {
            return;
        }
        self.taken = true;
        self.editing = false;
        let count = isize::try_from(count).unwrap_or(isize::MAX);
        let current = self.current.and_then(|at| isize::try_from(at).ok());
        let next = match current {
            Some(current) => (current + by).rem_euclid(count),
            None if by >= 0 => 0,
            None => count - 1,
        };
        self.current = usize::try_from(next).ok();
    }
}

/// `drawn`, the matches as the view draws them, lit where it draws them:
/// the one `shown` first, in a style of its own, then the rest.
fn highlights<'a>(
    drawn: impl Iterator<Item = &'a Hit>,
    shown: Option<&MatchKey>,
) -> Vec<Highlight> {
    let (current, rest): (Vec<&Hit>, Vec<&Hit>) = drawn.partition(|hit| Some(&hit.key()) == shown);
    let current = (current, Style::new().black().on_cyan());
    let rest = (rest, Style::new().black().on_yellow());
    [current, rest]
        .into_iter()
        .filter(|(hits, _)| !hits.is_empty())
        .map(|(hits, style)| Highlight {
            style,
            spans: hits.into_iter().map(Hit::span).collect(),
            ..Highlight::default()
        })
        .collect()
}

/// Transcript search in the terminal view, on Ctrl+F, or `/` while the
/// transcript has the keys.
#[derive(Default)]
pub struct TranscriptSearchPlugin;

impl Plugin for TranscriptSearchPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TuiPlugin>() {
            app.add_plugins(TuiPlugin);
        }
        // Shadows the input's Ctrl+F (a character forward, as Right does).
        app.add_key_binding(
            KeyBinding::new(KeyPress::ctrl('f'), "Search the transcript"),
            open,
        );
        // Shadows the input's `/` only while an item is selected.
        app.add_key_binding(
            KeyBinding::new(KeyPress::key(KeyCode::Char('/')), "Search the transcript")
                .in_scope(KeyScope::Transcript),
            open,
        );
        app.add_systems(rig_tui::TuiKeys, type_in_bar.in_set(InputSystems::Filter))
            .add_systems(Update, follow)
            .add_systems(
                PostUpdate,
                (
                    search_pending.before(TuiSystems::Prepare),
                    draw.in_set(TuiSystems::Draw),
                ),
            );
    }
}

/// Opens the bar with the keys, unless it is open, noting where the view
/// was.
fn open(
    pressed: On<KeyPressed>,
    bars: Query<(), With<TranscriptSearch>>,
    shown: Res<ShownAgent>,
    view: Single<&TranscriptView, With<MainPane>>,
    mut focus: ResMut<KeyFocus>,
    mut commands: Commands,
) {
    if pressed.released || !bars.is_empty() {
        return;
    }
    let back = shown.0.map(|agent| (agent, view.top().cloned()));
    let bar = (
        TranscriptSearch::new(back),
        TuiPanel::new(Placement::Bottom(Constraint::Length(1))),
    );
    focus.0 = Some(commands.spawn(bar).id());
}

/// The main pane: the one the bar searches, lights and scrolls.
type Main<'w, 's> = Single<'w, 's, (Entity, &'static mut TranscriptView), With<MainPane>>;

/// What a search reads live: whether a conversation (its notes too), a
/// reply streaming in, a running call's output or the agent shown changed
/// since the last frame; the notices and the width are the view's.
#[derive(bevy_ecs::system::SystemParam)]
struct Changes<'w, 's> {
    changed: Query<
        'w,
        's,
        (),
        Or<(
            Changed<Conversation>,
            Changed<Partial>,
            Changed<CallOutput>,
            Changed<ToolProgress>,
        )>,
    >,
    shown: Res<'w, ShownAgent>,
}

impl Changes<'_, '_> {
    fn any(&self) -> bool {
        !self.changed.is_empty() || self.shown.is_changed()
    }
}

/// Searches the open bar again when it asked ([`TranscriptSearch`]'s
/// `pending`), and shows the matches.
fn search_pending(world: &mut World) {
    let mut bars = world.query::<(Entity, &TranscriptSearch)>();
    let Some(bar) = bars
        .iter(world)
        .find(|(_, search)| search.pending)
        .map(|(bar, _)| bar)
    else {
        return;
    };
    search_now(world, bar);
    world.run_system_cached(show_current).ok();
}

/// Searches the bar `bar` again: the matches of its query in the transcript
/// of the agent shown, or of every agent, laid out as the view draws it, at
/// the width it draws at. It needs the world, since the layout runs the
/// render systems.
fn search_now(world: &mut World, bar: Entity) {
    let Some((regex, scope)) = world.get_mut::<TranscriptSearch>(bar).map(|mut search| {
        search.pending = false;
        (search.compiled(), search.scope)
    }) else {
        return;
    };
    let shown = world.get_resource::<ShownAgent>().and_then(|shown| shown.0);
    let mut panes = world.query_filtered::<&TranscriptView, With<MainPane>>();
    let Ok(view) = panes.single(world).cloned() else {
        return;
    };
    let mut agents: Vec<Entity> = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .filter(|agent| scope == Scope::All || Some(*agent) == shown)
        .collect();
    agents.sort();
    let width = view.width().unwrap_or(u16::MAX);
    let hits = regex.map_or_else(Vec::new, |regex| {
        let laid_out = agents.into_iter().filter_map(|agent| {
            let laid_out = TranscriptLayout::lay_out(world, &view, agent, width)?;
            Some(find(&regex, agent, &laid_out))
        });
        laid_out.flatten().collect()
    });
    if let Some(mut search) = world.get_mut::<TranscriptSearch>(bar) {
        search.found(hits);
    }
}

/// Shows the open bar's matches.
fn show_current(
    mut bars: Query<&mut TranscriptSearch>,
    shown: Res<ShownAgent>,
    main: Main,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let Ok(mut search) = bars.single_mut() else {
        return;
    };
    let (pane, mut view) = main.into_inner();
    show(&mut search, (pane, &mut view), shown.0, &mut commands);
    redraw.write(RequestRedraw);
}

/// Closes the open bar.
fn close_bar(mut bars: Query<(Entity, &mut TranscriptSearch)>, main: Main, mut commands: Commands) {
    let Ok((bar, mut search)) = bars.single_mut() else {
        return;
    };
    let (pane, mut view) = main.into_inner();
    close(bar, &mut search, (pane, &mut view), &mut commands);
}

/// Types the key into the open bar, searching again when the query
/// changed, and shows the match in the transcript; Esc closes it. A filter
/// of each key.
fn type_in_bar(world: &mut World) {
    let mut bars = world.query_filtered::<Entity, With<TranscriptSearch>>();
    let Ok(bar) = bars.single(world) else {
        return;
    };
    // The bar takes the keys it does, as the panel with the keys; those
    // after an Esc are the input's again.
    let Some(mut pending) = world.get_resource_mut::<PendingInput>() else {
        return;
    };
    let events = std::mem::take(&mut pending.0);
    let mut kept = Vec::with_capacity(events.len());
    let (mut closing, mut typed) = (false, false);
    for item in events {
        let KeyInput::Event(event) = &item else {
            kept.push(item);
            continue;
        };
        if closing {
            kept.push(item);
            continue;
        }
        let mut search = world.get_mut::<TranscriptSearch>(bar);
        match search.as_mut().map(|search| search.press(event)) {
            Some(Pressed::Other) | None => kept.push(item),
            Some(Pressed::Esc) => closing = true,
            Some(Pressed::Done) => {
                typed = true;
                if search.is_some_and(|search| search.dirty) {
                    search_now(world, bar);
                }
            }
        }
    }
    if let Some(mut pending) = world.get_resource_mut::<PendingInput>() {
        pending.0 = kept;
    }
    if closing {
        world.run_system_cached(close_bar).ok();
    } else if typed {
        world.run_system_cached(show_current).ok();
    }
}

/// Each frame: searches the open bar again when what it searches changed,
/// such as a reply streaming in, and closes it when it lost the keys.
fn follow(
    mut bars: Query<(Entity, &mut TranscriptSearch)>,
    focus: Res<KeyFocus>,
    changes: Changes,
    main: Main,
    mut commands: Commands,
) {
    let Ok((bar, mut search)) = bars.single_mut() else {
        return;
    };
    let (pane, mut view) = main.into_inner();
    if focus.0 != Some(bar) {
        close(bar, &mut search, (pane, &mut view), &mut commands);
    } else if changes.any() || view.is_changed() {
        // The view counts as changed only when it changes, such as on a
        // new width, which the layout wraps at, or a notice.
        search.pending = true;
    }
}

/// Lights the matches where the view draws them, ahead of other plugins'
/// highlights, and, when the match shown moved, scrolls to its row.
fn show(
    search: &mut TranscriptSearch,
    (pane, view): (Entity, &mut Mut<TranscriptView>),
    showing: Option<Entity>,
    commands: &mut Commands,
) {
    let shown = search.key(search.current);
    let went = shown.clone().zip(search.current().map(|hit| hit.row));
    let jump = (went != search.went).then(|| search.current().cloned());
    search.went = went;
    let jump = jump.flatten();
    // The agent the view draws once it jumps.
    let agent = jump.as_ref().map(|hit| hit.agent).or(showing);
    if let Some(hit) = jump {
        let (item, row) = (hit.item().cloned(), ItemRow::Top(hit.row));
        commands.trigger(ShowAgent { entity: hit.agent });
        commands.trigger(ScrollTo { pane, item, row });
    }
    let drawn = search.hits.iter().filter(|hit| Some(hit.agent) == agent);
    let lit = highlights(drawn, shown.as_ref());
    if lit != search.lit {
        view.highlights
            .retain(|highlight| !search.lit.contains(highlight));
        view.highlights.splice(0..0, lit.iter().cloned());
        search.lit = lit;
    }
}

/// Closes the bar and unlights the matches; with no match taken, also
/// scrolls back to where the view was.
fn close(
    bar: Entity,
    search: &mut TranscriptSearch,
    (pane, view): (Entity, &mut Mut<TranscriptView>),
    commands: &mut Commands,
) {
    commands.entity(bar).despawn();
    if !search.lit.is_empty() {
        view.highlights
            .retain(|highlight| !search.lit.contains(highlight));
    }
    if search.taken {
        return;
    }
    if let Some((agent, top)) = search.back.take() {
        let (item, row) = top.map_or((None, 0), |(item, row)| (Some(item), row));
        let row = ItemRow::Top(row);
        commands.trigger(ShowAgent { entity: agent });
        commands.trigger(ScrollTo { pane, item, row });
    }
}

/// Draws the bar: the query, the count, and where the match is.
fn draw(mut bars: Query<(&TranscriptSearch, &mut PanelCanvas)>, names: Query<&Name>) {
    for (search, mut canvas) in &mut bars {
        let mut spans = vec![
            Span::from(format!("/{}", search.query)).bold(),
            Span::from(format!("  {}", search.counter())).yellow(),
        ];
        if search.scope == Scope::All {
            let agent = search.current().and_then(|hit| names.get(hit.agent).ok());
            let agent = agent.map(Name::as_str);
            spans.push(Span::from(format!(
                "  all agents · {}",
                agent.unwrap_or("-")
            )));
        }
        let keys = if search.editing {
            "  Enter take · Tab scope · Esc close"
        } else {
            "  n next · N previous · Esc close"
        };
        spans.push(Span::from(keys).dim());
        canvas.render(Line::from(spans));
    }
}

#[cfg(test)]
mod tests;

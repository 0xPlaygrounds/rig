//! FH-20: transcript search, as in `less`.
//!
//! Ctrl+F opens a search bar under the transcript (a [`TuiPanel`] that
//! takes the keys). What is typed searches as it is typed, ignoring case
//! unless the query has a capital letter; `re:` makes the rest a regular
//! expression. It searches the transcript as the view lays it out
//! ([`TranscriptLayout`]), every item whole, at the width it draws at:
//! each line as its renderer draws it, its wrapped rows rejoined, a
//! collapsed output's cut lines included, the notices where they are drawn
//! and the reply streaming in.
//!
//! The matches the view draws are lit in the transcript, each where the
//! search found it ([`Highlight::spans`]), the one shown in a style of its
//! own, across the rows a wrap split one into. Moving to one scrolls the
//! view to its row ([`ScrollTo`] with [`ItemRow::Top`]), expanding the items it
//! is in that are drawn cut. Enter takes it; `n` (or Enter again) and `N`
//! (or Shift+Enter) move, wrapping around, and the bar counts (`3/17`).
//! Tab searches every agent, naming the agent of the match and showing it. Esc
//! closes the bar and unlights the matches; unless a match was taken, it
//! also collapses what the search expanded and scrolls back to where the
//! view was. The search only reads: it never changes a conversation.

use std::collections::HashMap;
use std::ops::Range;

use regex::{Regex, RegexBuilder};
use rig_harness::prelude::*;
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::{
    AppKeyBindingsExt, DrawnLine, DrawnTranscript, Expanded, Highlight, ItemRow, KeyBinding,
    KeyFocus, KeyPress, KeyPressed, LineSpan, PanelCanvas, Placement, RequestRedraw, ScrollTo,
    TerminalEvent, TranscriptItem, TranscriptLayout, TranscriptView, TuiPanel, TuiPlugin,
    TuiSystems,
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
    /// The agent shown and where the view was scrolled when it opened.
    back: Option<(Entity, Option<(TranscriptItem, usize)>)>,
    /// The match the view was last scrolled to, and its row then.
    went: Option<(MatchKey, usize)>,
    /// The items the search expanded, which were drawn cut before.
    expanded: Vec<TranscriptItem>,
    /// The highlights it lit.
    lit: Vec<Highlight>,
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
            back,
            went: None,
            expanded: Vec::new(),
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

    /// Searches `transcripts` again with the query, keeping the match
    /// shown where it still is; else, while typing, the latest match.
    fn search(&mut self, transcripts: &Transcripts, view: &TranscriptView) {
        let was = self.key(self.current);
        if self.dirty {
            self.dirty = false;
            let compiled = compile(&self.query);
            self.error = compiled.is_err();
            self.regex = compiled.ok().flatten();
        }
        self.hits = self
            .regex
            .as_ref()
            .map(|regex| transcripts.find(regex, self.scope, view))
            .unwrap_or_default();
        let kept = (0..self.hits.len()).find(|&at| was.is_some() && self.key(Some(at)) == was);
        let last = self.hits.len().checked_sub(1);
        self.current = match kept {
            Some(kept) if !self.editing => Some(kept),
            _ if self.editing => last,
            _ => self.current.zip(last).map(|(at, last)| at.min(last)),
        };
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
fn highlights(drawn: &[Hit], shown: Option<&MatchKey>) -> Vec<Highlight> {
    let (current, rest): (Vec<&Hit>, Vec<&Hit>) =
        drawn.iter().partition(|hit| Some(&hit.key()) == shown);
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

/// Transcript search in the terminal view, on Ctrl+F.
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
        app.add_systems(Update, type_in_bar)
            .add_systems(PostUpdate, draw.in_set(TuiSystems::Draw));
    }
}

/// Ctrl+F: opens the bar with the keys, unless it is open.
fn open(
    pressed: On<KeyPressed>,
    bars: Query<(), With<TranscriptSearch>>,
    shown: Res<Shown>,
    view: Res<TranscriptView>,
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

/// What the bar searches, read live.
#[derive(bevy_ecs::system::SystemParam)]
struct Transcripts<'w, 's> {
    layout: TranscriptLayout<'w, 's>,
    agents: Query<'w, 's, Entity, With<Agent>>,
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
    shown: Res<'w, Shown>,
}

impl Transcripts<'_, '_> {
    /// Whether a conversation (its notes too), a reply streaming in, a running
    /// call's output or the agent shown changed since the last frame; the
    /// notices and the width are the view's.
    fn changed(&self) -> bool {
        !self.changed.is_empty() || self.shown.is_changed()
    }

    /// The matches of `regex` in the transcript of the agent shown, or of
    /// every agent, laid out as the view draws it with every item whole,
    /// at the width it draws at.
    fn find(&self, regex: &Regex, scope: Scope, view: &TranscriptView) -> Vec<Hit> {
        let mut agents: Vec<Entity> = self
            .agents
            .iter()
            .filter(|agent| scope == Scope::All || Some(*agent) == self.shown.0)
            .collect();
        agents.sort();
        let width = view.width().unwrap_or(u16::MAX);
        let laid_out = agents.into_iter().filter_map(|agent| {
            let laid_out = self.layout.lay_out(view, agent, width, Expanded::All)?;
            Some(find(regex, agent, &laid_out))
        });
        laid_out.flatten().collect()
    }
}

/// Types into the open bar, in order, then searches again when the query
/// or what it searches changed, and shows the match in the transcript. It
/// closes when it loses the keys, as on an Esc. The keys of the frame it
/// opened in were the input's.
fn type_in_bar(
    mut events: MessageReader<TerminalEvent>,
    mut bars: Query<(Entity, &mut TranscriptSearch, Ref<TuiPanel>)>,
    focus: Res<KeyFocus>,
    transcripts: Transcripts,
    mut view: ResMut<TranscriptView>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let Ok((bar, mut search, opened)) = bars.single_mut() else {
        events.clear();
        return;
    };
    if opened.is_added() {
        events.clear();
    }
    if focus.0 != Some(bar) && !opened.is_added() {
        close(bar, &mut search, &mut view, &mut commands);
        return;
    }
    let (mut closing, mut typed) = (false, false);
    for TerminalEvent(event) in events.read() {
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                let shift = key.modifiers.contains(KeyModifiers::SHIFT);
                let plain = !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
                match key.code {
                    KeyCode::Esc => closing = true,
                    KeyCode::Enter if shift => search.step(-1),
                    KeyCode::Enter if search.editing => {
                        search.editing = false;
                        search.taken = search.current.is_some();
                    }
                    KeyCode::Enter => search.step(1),
                    KeyCode::Tab => {
                        search.scope = match search.scope {
                            Scope::Shown => Scope::All,
                            Scope::All => Scope::Shown,
                        };
                        search.dirty = true;
                    }
                    KeyCode::Backspace => {
                        search.editing = true;
                        search.query.pop();
                        search.dirty = true;
                    }
                    KeyCode::Char('n') if plain && !search.editing => search.step(1),
                    KeyCode::Char('N') if plain && !search.editing => search.step(-1),
                    KeyCode::Char(typed) if plain => {
                        search.editing = true;
                        search.query.push(typed);
                        search.dirty = true;
                    }
                    _ => continue,
                }
            }
            Event::Paste(text) => {
                search.editing = true;
                let line = text.lines().next().unwrap_or_default();
                search.query.push_str(line);
                search.dirty = true;
            }
            _ => continue,
        }
        if closing {
            break;
        }
        typed = true;
        if search.dirty {
            search.search(&transcripts, &view);
        }
    }
    if closing {
        close(bar, &mut search, &mut view, &mut commands);
        return;
    }
    // The view counts as changed only when it changes, such as on a new
    // width, which the layout wraps at, or a notice; not on a key.
    if !search.dirty && (transcripts.changed() || view.is_changed()) {
        search.search(&transcripts, &view);
        typed = true;
    }
    if typed {
        show(&mut search, &mut view, &transcripts, &mut commands);
        redraw.write(RequestRedraw);
    }
}

/// Lights the matches where the view draws them, ahead of other plugins'
/// highlights, and, when the match shown moved, scrolls to its row,
/// expanding the items it is in that are drawn cut, in place of those the
/// search expanded before. The row is the match's as the view draws its
/// item with those expanded, not whole: a call cut before the match in
/// the same message still draws cut.
fn show(
    search: &mut TranscriptSearch,
    view: &mut ResMut<TranscriptView>,
    transcripts: &Transcripts,
    commands: &mut Commands,
) {
    let shown = search.key(search.current);
    let went = shown.clone().zip(search.current().map(|hit| hit.row));
    let jump = (went != search.went).then(|| search.current().cloned());
    search.went = went;
    let jump = jump.flatten();
    let width = view.width().unwrap_or(u16::MAX);
    let as_shown = |view: &TranscriptView, agent| {
        let layout = &transcripts.layout;
        layout.lay_out(view, agent, width, Expanded::AsShown)
    };
    if let Some(hit) = &jump {
        for item in search
            .expanded
            .extract_if(.., |item| !hit.within.contains(item))
        {
            view.expanded.remove(&item);
        }
        let cut: Vec<TranscriptItem> = as_shown(view, hit.agent)
            .map(|drawn| {
                hit.within
                    .iter()
                    .filter(|item| drawn.is_cut(item))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        for item in cut {
            if view.expanded.insert(item.clone()) {
                search.expanded.push(item);
            }
        }
    }
    // The agent the view draws once it jumps.
    let agent = jump.as_ref().map(|hit| hit.agent).or(transcripts.shown.0);
    let laid_out = agent.and_then(|agent| as_shown(view, agent));
    let drawn = search.regex.as_ref().zip(agent).zip(laid_out.as_ref());
    let drawn = drawn.map(|((regex, agent), laid_out)| find(regex, agent, laid_out));
    let drawn = drawn.unwrap_or_default();
    if let Some(hit) = jump {
        let found = drawn.iter().find(|found| found.key() == hit.key());
        let row = ItemRow::Top(found.map_or(hit.row, |found| found.row));
        let (agent, item) = (hit.agent, hit.item().cloned());
        commands.trigger(ScrollTo { agent, item, row });
    }
    // Drawn cut, the match shown counts among other matches of its item.
    let whole = |hit: &Hit| {
        laid_out
            .as_ref()
            .is_some_and(|laid_out| !hit.within.iter().any(|item| laid_out.is_cut(item)))
    };
    let shown = shown.filter(|_| search.current().is_some_and(whole));
    let lit = highlights(&drawn, shown.as_ref());
    if lit != search.lit {
        view.highlights
            .retain(|highlight| !search.lit.contains(highlight));
        view.highlights.splice(0..0, lit.iter().cloned());
        search.lit = lit;
    }
}

/// Closes the bar and unlights the matches; with no match taken, also
/// collapses what it expanded and scrolls back to where the view was.
fn close(
    bar: Entity,
    search: &mut TranscriptSearch,
    view: &mut ResMut<TranscriptView>,
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
    for item in search.expanded.drain(..) {
        view.expanded.remove(&item);
    }
    if let Some((agent, top)) = search.back.take() {
        let (item, row) = top.map_or((None, 0), |(item, row)| (Some(item), row));
        let row = ItemRow::Top(row);
        commands.trigger(ScrollTo { agent, item, row });
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

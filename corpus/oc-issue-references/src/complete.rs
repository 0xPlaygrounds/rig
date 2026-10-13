//! `#` completion in the terminal view's input: the open issues whose
//! title (or number) matches what follows the `#` before the cursor, in a
//! row panel under the input; Tab puts the first in. The view's own
//! completion completes only `/` and `@` ([`rig_tui::CompletionKind`]),
//! so this one is a panel and a key filter of the plugin's.

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_tui::crossterm::event::{Event, KeyCode};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::widgets::Paragraph;
use rig_tui::{
    InputSystems, KeyInput, PanelCanvas, Placement, RequestRedraw, TerminalEvents, TuiInput,
    TuiKeys, TuiPanel, TuiSystems,
};

use crate::{State, load_open_issues};

/// Issues the panel lists at most.
const SHOWN: usize = 5;

/// The tracker's open issues, number and title, loaded in the background
/// the first time the user types `#`, once per session.
#[derive(Resource, Default)]
pub struct OpenIssues {
    /// The issues, once loaded; empty when they could not be.
    pub issues: Option<Vec<(u64, String)>>,
    loading: bool,
}

/// The issues offered for the `#` token before the cursor.
#[derive(Resource, Default, PartialEq, Eq)]
struct Offered(Vec<(u64, String)>);

/// The panel that lists [`Offered`].
#[derive(Component)]
struct IssuePanel;

/// On the entity loading the open issues: what they loaded as.
#[derive(Component)]
struct OpenLoaded(Result<Vec<(u64, String)>, String>);

impl From<Result<Vec<(u64, String)>, String>> for OpenLoaded {
    fn from(loaded: Result<Vec<(u64, String)>, String>) -> Self {
        Self(loaded)
    }
}

pub(crate) fn add(app: &mut App) {
    app.init_resource::<OpenIssues>()
        .init_resource::<Offered>()
        // Inert without the terminal view, which is there in any run.
        .init_resource::<TuiInput>()
        .add_message::<RequestRedraw>()
        .add_systems(
            PostUpdate,
            (
                offer.in_set(TuiSystems::Prepare),
                draw.in_set(TuiSystems::Draw),
            ),
        )
        .add_systems(TuiKeys, accept.in_set(InputSystems::Filter))
        .add_observer(open_loaded);
}

/// The byte offset of the `#` token the cursor ends, and what follows it.
fn token(input: &TuiInput) -> Option<(usize, &str)> {
    let before = input.text().get(..input.cursor())?;
    let start = before.rfind('#')?;
    let query = before.get(start + 1..)?;
    let boundary = before
        .get(..start)?
        .chars()
        .next_back()
        .is_none_or(|c| c.is_whitespace() || c == '(');
    (boundary && !query.contains(char::is_whitespace)).then_some((start, query))
}

/// The issues of `open` that `query` matches by number or title.
fn matching(open: &[(u64, String)], query: &str) -> Vec<(u64, String)> {
    let query = query.to_lowercase();
    let matches = |(number, title): &&(u64, String)| {
        number.to_string().starts_with(&query) || title.to_lowercase().contains(&query)
    };
    open.iter().filter(matches).take(SHOWN).cloned().collect()
}

/// Offers the issues the `#` token before the cursor matches, loading
/// the open issues the first time.
fn offer(
    input: Res<TuiInput>,
    mut open: ResMut<OpenIssues>,
    mut offered: ResMut<Offered>,
    (state, processes, root, wake): (Res<State>, Res<Processes>, Res<ProjectRoot>, Res<Wake>),
    panels: Query<Entity, With<IssuePanel>>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let token = token(&input);
    if token.is_some() && open.issues.is_none() && !open.loading {
        open.loading = true;
        let work = load_open_issues(state.0.clone(), processes.launcher(), root.0.clone());
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        let running = Running::spawn_into::<OpenLoaded, _>(pool, &wake, work);
        commands.spawn((Name::new("loading the open issues"), running));
    }
    let issues = open.issues.as_deref().unwrap_or_default();
    let now = Offered(
        token
            .map(|(_, query)| matching(issues, query))
            .unwrap_or_default(),
    );
    if now == *offered {
        return;
    }
    for panel in &panels {
        commands.entity(panel).despawn();
    }
    if let Ok(rows) = u16::try_from(now.0.len())
        && rows > 0
    {
        let placement = Placement::BelowInput(Constraint::Length(rows));
        commands.spawn((IssuePanel, TuiPanel::new(placement)));
    }
    *offered = now;
    redraw.write(RequestRedraw);
}

/// Keeps the open issues once loaded.
fn open_loaded(
    loaded: On<Insert<OpenLoaded>>,
    results: Query<&OpenLoaded>,
    mut open: ResMut<OpenIssues>,
    mut commands: Commands,
) {
    if let Ok(OpenLoaded(result)) = results.get(loaded.entity) {
        open.issues = Some(result.clone().unwrap_or_default());
        commands.entity(loaded.entity).despawn();
    }
}

fn draw(mut panels: Query<&mut PanelCanvas, With<IssuePanel>>, offered: Res<Offered>) {
    let lines: Vec<String> = offered
        .0
        .iter()
        .enumerate()
        .map(|(at, (number, title))| {
            let mark = if at == 0 { "Tab" } else { "   " };
            format!("{mark} #{number} {title}")
        })
        .collect();
    for mut canvas in &mut panels {
        canvas.render(Paragraph::new(lines.join("\n")));
    }
}

/// Tab while issues are offered puts the first in place of the token.
fn accept(mut events: ResMut<TerminalEvents>, mut input: ResMut<TuiInput>, open: Res<OpenIssues>) {
    let tab = |item: &KeyInput| matches!(item, KeyInput::Event(Event::Key(key)) if key.code == KeyCode::Tab);
    if !events.0.first().is_some_and(tab) {
        return;
    }
    let Some((start, query)) = token(&input) else {
        return;
    };
    let issues = open.issues.as_deref().unwrap_or_default();
    let Some((number, _)) = matching(issues, query).into_iter().next() else {
        return;
    };
    let (text, cursor) = (input.text(), input.cursor());
    let chosen = format!("#{number} ");
    let before = text.get(..start).unwrap_or_default();
    let after = text.get(cursor..).unwrap_or_default();
    let replaced = format!("{before}{chosen}{after}");
    input.set(replaced);
    input.set_cursor(start + chosen.len());
    events.0.remove(0);
}

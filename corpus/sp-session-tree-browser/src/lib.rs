//! A session and branch browser for the terminal view: `/browse` opens a
//! full-screen box of three panes over the screen, the saved sessions of
//! this project with the live one first (searchable with `/`), the
//! selected session's branch tree, one node per turn, and a read-only
//! preview of the conversation from the root to the selected node, drawn
//! by a transcript pane, so with every plugin's looks.
//!
//! Up and Down (or `j` and `k`) move, Tab goes to the next pane, Enter on
//! a node resumes its session there (an inner node becomes the end of the
//! active line, so the next prompt starts a branch), `o` shows the
//! session read-only over the whole screen, and Esc closes. The browser
//! is a box over the screen, so closing it leaves the screen as it was:
//! the agent shown, the transcript's scroll and the input's draft. A
//! saved session is read off the main thread, and only once selected.

use std::collections::{HashMap, HashSet};

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::{Conversation as Line, MessageKey};
use rig_harness::rig_memory::{read_line, title_of};
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::Line as Row;
use rig_tui::ratatui::widgets::{Block, Clear, Paragraph};
use rig_tui::{
    InputAction, InputSystems, KeyBinding, KeyFocus, KeyInput, KeyPress, KeyPressed, KeyScope,
    PaneKeys, PanelCanvas, Placement, RequestRedraw, TerminalEvents, TranscriptPane,
    TranscriptSource, TuiKeys, TuiPanel, TuiSystems,
};

#[cfg(test)]
mod tests;

/// `/browse`: the session and branch browser.
#[derive(Default)]
pub struct SessionTreeBrowserPlugin;

impl Plugin for SessionTreeBrowserPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RigHome>()
            .init_resource::<SessionStores>()
            .add_command(
                "browse",
                "Browse the saved sessions and their branches; Enter resumes at a turn",
                browse,
            )
            .add_observer(listed)
            .add_observer(read)
            .add_systems(TuiKeys, type_search.in_set(InputSystems::Filter))
            .add_systems(
                PostUpdate,
                (draw_sessions, draw_tree, draw_box).in_set(TuiSystems::Draw),
            );
    }
}

/// Who makes the browser's changes to a conversation.
fn by() -> Author {
    Author::plugin("browse")
}

/// A session the browser lists.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Which {
    /// The one running here.
    Live,
    /// A saved one.
    Saved(SessionId),
}

/// A session's row in the sessions pane.
#[derive(Clone, Debug)]
struct SessionRow {
    which: Which,
    title: String,
    detail: String,
}

/// A turn in a session's tree.
#[derive(Clone, Debug)]
struct Node {
    /// Its first entry, the user's prompt.
    first: MessageKey,
    /// Its last entry: where a line through it ends with it.
    last: MessageKey,
    /// The first line of its prompt.
    title: String,
    /// The turns after it, oldest first.
    children: Vec<usize>,
    /// Whether the session's active line ends with it.
    current: bool,
}

/// The turns of a session's lines, merged where the lines share them.
#[derive(Clone, Debug, Default)]
struct Tree {
    nodes: Vec<Node>,
    roots: Vec<usize>,
}

/// A row of the tree pane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TreeRow {
    /// A turn, `depth` branchings in.
    Node { node: usize, depth: usize },
    /// The middle of a long run without branching, folded: the run's first
    /// node, the last node folded and how many are.
    Fold {
        start: usize,
        last: usize,
        hidden: usize,
        depth: usize,
    },
}

/// The browser's state, on its box.
#[derive(Component)]
struct Browser {
    /// The sessions pane, the tree pane and the preview pane.
    panes: [Entity; 3],
    /// The sessions, the live one first; `None` while they are read.
    sessions: Option<Vec<SessionRow>>,
    /// The search typed after `/`, and whether it is being typed.
    search: String,
    searching: bool,
    /// The session selected, among those the search leaves.
    selected: usize,
    /// Each saved session read, or why it could not be.
    read: HashMap<SessionId, Result<Line, String>>,
    /// The saved sessions being read.
    reading: HashSet<SessionId>,
    /// The tree shown, of which session, its rows and the one selected.
    tree: Option<(Which, Tree)>,
    rows: Vec<TreeRow>,
    row: usize,
    /// The runs unfolded, by their first node.
    unfolded: HashSet<usize>,
    /// What the last action said, such as a refusal.
    said: Option<String>,
    /// Whether the preview fills the screen, showing a whole session.
    open: bool,
}

/// The sessions pane.
#[derive(Component)]
struct SessionsPane;

/// The tree pane.
#[derive(Component)]
struct TreePane;

/// What a key of the browser does.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
enum Act {
    Up,
    Down,
    Next,
    Search,
    Enter,
    Open,
    Close,
}

/// The saved sessions of this project, read off the main thread.
struct Listed(Vec<SessionRow>);

/// A saved session's main agent's conversation, read off the main thread.
struct Read(Option<(SessionId, Result<Line, String>)>);

/// `/browse`: the browser's box, its panes and their keys; the sessions
/// pane gets the keys.
fn browse(In(args): In<CommandArgs>, world: &mut World) {
    let mut open = world.query_filtered::<(), With<Browser>>();
    if open.iter(world).next().is_some() || !world.contains_resource::<KeyFocus>() {
        return;
    }
    let all = Constraint::Percentage(100);
    let over = Placement::Over {
        width: all,
        height: all,
    };
    let root = world
        .spawn((Name::new("session browser"), TuiPanel::new(over)))
        .id();
    let within = |part| {
        TuiPanel::new(Placement::Within {
            panel: root,
            part,
            of: 3,
        })
    };
    let sessions = world.spawn((SessionsPane, within(0), ChildOf(root))).id();
    let tree = world.spawn((TreePane, within(1), ChildOf(root))).id();
    let pane = TranscriptPane::new(TranscriptSource::Value(Line::new())).titled("preview");
    let preview = world.spawn((pane, within(2), PaneKeys, ChildOf(root))).id();
    let keys = [
        (KeyCode::Up, Act::Up),
        (KeyCode::Char('k'), Act::Up),
        (KeyCode::Down, Act::Down),
        (KeyCode::Char('j'), Act::Down),
        (KeyCode::Tab, Act::Next),
        (KeyCode::Enter, Act::Enter),
        (KeyCode::Char('o'), Act::Open),
        (KeyCode::Esc, Act::Close),
    ];
    for pane in [sessions, tree] {
        for (code, act) in keys {
            bind(world, pane, code, act);
        }
    }
    bind(world, sessions, KeyCode::Char('/'), Act::Search);
    bind(world, preview, KeyCode::Tab, Act::Next);
    bind(world, preview, KeyCode::Esc, Act::Close);
    for (code, action) in [
        (KeyCode::Char('k'), InputAction::ScrollUp),
        (KeyCode::Char('j'), InputAction::ScrollDown),
    ] {
        let binding = KeyBinding::new(KeyPress::key(code), "Scroll");
        world.spawn((
            binding.in_scope(KeyScope::Panel(preview)),
            action,
            ChildOf(preview),
        ));
    }
    let live = live_row(world, args.agent);
    world.entity_mut(root).insert(Browser {
        panes: [sessions, tree, preview],
        sessions: None,
        search: String::new(),
        searching: false,
        selected: 0,
        read: HashMap::new(),
        reading: HashSet::new(),
        tree: None,
        rows: Vec::new(),
        row: 0,
        unfolded: HashSet::new(),
        said: None,
        open: false,
    });
    list(world, root, live);
    world.insert_resource(KeyFocus(Some(sessions)));
    refresh(world);
}

/// Binds `code` in `pane` to `act`.
fn bind(world: &mut World, pane: Entity, code: KeyCode, act: Act) {
    let binding = KeyBinding::new(KeyPress::key(code), format!("{act:?}"));
    let binding = binding.in_scope(KeyScope::Panel(pane));
    world.spawn((binding, act, ChildOf(pane))).observe(pressed);
}

/// The live session's row: the start of what its agent was first asked.
fn live_row(world: &World, agent: Entity) -> SessionRow {
    let shown = world.get_resource::<Shown>().and_then(|shown| shown.0);
    let conversation = world.get::<Conversation>(shown.unwrap_or(agent));
    let title = conversation.and_then(|conversation| first_typed(conversation));
    SessionRow {
        which: Which::Live,
        title: title.map_or_else(|| "this session".to_owned(), |title| title_of(&title)),
        detail: "current".to_owned(),
    }
}

/// Lists the saved sessions of this project, newest first, after `live`,
/// off the main thread: only their listings are read.
fn list(world: &mut World, root: Entity, live: SessionRow) {
    let home = world.resource::<RigHome>().0.clone();
    let stores = world.resource::<SessionStores>().clone();
    let here = world
        .get_resource::<SessionPaths>()
        .map(|paths| paths.path().to_owned());
    let project = world
        .get_resource::<ProjectRoot>()
        .map(|root| root.0.clone());
    let Some(wake) = world.get_resource::<Wake>().cloned() else {
        return;
    };
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let work = async move {
        let mut saved: Vec<SessionEntry> = home
            .saved_sessions()
            .into_iter()
            .filter(|(_, dir)| Some(dir.path()) != here.as_deref())
            .filter_map(|(_, dir)| SessionEntry::read(&dir, &stores))
            .filter(|entry| project.is_none() || entry.directory == project)
            .collect();
        saved.sort_by_key(|saved| std::cmp::Reverse(saved.updated));
        let rows = saved.into_iter().map(|entry| SessionRow {
            title: match (&entry.name, entry.listing.meta.title.as_str()) {
                (Some(name), _) => name.clone(),
                (None, "") => format!("session {}", entry.id),
                (None, title) => title.to_owned(),
            },
            detail: entry.detail(),
            which: Which::Saved(entry.id),
        });
        Listed(std::iter::once(live).chain(rows).collect())
    };
    world
        .entity_mut(root)
        .insert(Running::spawn(pool, &wake, work));
}

/// Puts the sessions read in the browser.
fn listed(
    done: On<Insert<Done<Listed>>>,
    mut browsers: Query<(&mut Browser, &mut Done<Listed>)>,
    mut commands: Commands,
) {
    if let Ok((mut browser, mut listed)) = browsers.get_mut(done.entity) {
        browser.sessions = Some(std::mem::take(&mut listed.0.0));
        commands.queue(refresh);
    }
}

/// Puts a saved session read in the browser, and lets its task go.
fn read(
    done: On<Insert<Done<Read>>>,
    mut tasks: Query<(&mut Done<Read>, &ChildOf)>,
    mut browsers: Query<&mut Browser>,
    mut commands: Commands,
) {
    let Ok((mut read, &ChildOf(root))) = tasks.get_mut(done.entity) else {
        return;
    };
    let Some((id, conversation)) = read.0.0.take() else {
        return;
    };
    if let Ok(mut browser) = browsers.get_mut(root) {
        browser.reading.remove(&id);
        browser.read.insert(id, conversation);
    }
    commands.entity(done.entity).despawn();
    commands.queue(refresh);
}

/// A key of the browser, pressed.
fn pressed(pressed: On<KeyPressed>, acts: Query<&Act>, mut commands: Commands) {
    if let (Ok(&act), false) = (acts.get(pressed.entity), pressed.released) {
        commands.queue(move |world: &mut World| {
            act_on(world, act);
            refresh(world);
        });
    }
}

/// The browser's box.
fn browser(world: &mut World) -> Option<Entity> {
    let mut browsers = world.query_filtered::<Entity, With<Browser>>();
    browsers.iter(world).next()
}

/// Changes the browser on `root`.
fn edit(world: &mut World, root: Entity, change: impl FnOnce(&mut Browser)) {
    if let Some(mut browser) = world.get_mut::<Browser>(root) {
        change(&mut browser);
    }
}

/// Which of the browser's panes has the keys: 0 sessions, 1 tree, 2
/// preview.
fn focused(world: &World, panes: [Entity; 3]) -> Option<usize> {
    let focus = world.get_resource::<KeyFocus>()?.0?;
    panes.iter().position(|pane| *pane == focus)
}

/// Does what `act` says in the pane with the keys.
fn act_on(world: &mut World, act: Act) {
    let Some(root) = browser(world) else {
        return;
    };
    let Some(browser) = world.get::<Browser>(root) else {
        return;
    };
    let panes = browser.panes;
    let [_, tree, preview] = panes;
    let open = browser.open;
    let pane = focused(world, panes);
    edit(world, root, |browser| browser.said = None);
    let focus = |world: &mut World, pane: Entity| world.insert_resource(KeyFocus(Some(pane)));
    match (act, pane) {
        (Act::Close, _) if open => {
            edit(world, root, |browser| browser.open = false);
            focus(world, tree);
        }
        (Act::Close, _) => {
            world.entity_mut(root).despawn();
        }
        _ if open => {}
        (Act::Next, Some(at)) => {
            if let Some(&next) = panes.get((at + 1) % panes.len()) {
                focus(world, next);
            }
        }
        (Act::Search, _) => edit(world, root, |browser| browser.searching = true),
        (Act::Up | Act::Down, Some(0)) => edit(world, root, |browser| {
            let count = visible(browser).len();
            browser.selected = step(browser.selected, act, count);
        }),
        (Act::Up | Act::Down, Some(1)) => edit(world, root, |browser| {
            browser.row = step(browser.row, act, browser.rows.len());
        }),
        (Act::Enter, Some(0)) => focus(world, tree),
        (Act::Enter, Some(1)) => {
            let row = world.get::<Browser>(root);
            match row.and_then(|browser| browser.rows.get(browser.row).copied()) {
                Some(TreeRow::Fold { start, .. }) => edit(world, root, |browser| {
                    browser.unfolded.insert(start);
                    if let Some((_, tree)) = &browser.tree {
                        browser.rows = rows(tree, &browser.unfolded);
                    }
                }),
                Some(TreeRow::Node { node, .. }) => resume(world, root, node),
                None => {}
            }
        }
        (Act::Open, Some(0 | 1)) => {
            let loaded = world.get::<Browser>(root).is_some_and(|browser| {
                selected(browser).is_some_and(|which| match which {
                    Which::Live => true,
                    Which::Saved(id) => browser.read.get(id).is_some_and(Result::is_ok),
                })
            });
            if loaded {
                edit(world, root, |browser| browser.open = true);
                focus(world, preview);
            }
        }
        _ => {}
    }
}

/// `at` moved one up or down among `count`.
fn step(at: usize, act: Act, count: usize) -> usize {
    match act {
        Act::Up => at.saturating_sub(1),
        _ => (at + 1).min(count.saturating_sub(1)),
    }
}

/// The sessions the search leaves, in order.
fn visible(browser: &Browser) -> Vec<&SessionRow> {
    let search = browser.search.to_lowercase();
    let rows = browser.sessions.iter().flatten();
    rows.filter(|row| {
        search.is_empty()
            || row.title.to_lowercase().contains(&search)
            || row.detail.to_lowercase().contains(&search)
    })
    .collect()
}

/// The session selected, if any.
fn selected(browser: &Browser) -> Option<&Which> {
    let rows = visible(browser);
    rows.get(browser.selected).map(|row| &row.which)
}

/// Types the search while it is being typed: every key goes to it, Enter
/// and Esc end it.
fn type_search(
    mut events: ResMut<TerminalEvents>,
    mut browsers: Query<&mut Browser>,
    mut commands: Commands,
) {
    let Some(mut browser) = browsers
        .iter_mut()
        .next()
        .filter(|browser| browser.searching)
    else {
        return;
    };
    let Some(KeyInput::Event(Event::Key(key))) = events.0.first() else {
        return;
    };
    let key = *key;
    events.0.remove(0);
    if key.kind == KeyEventKind::Release {
        return;
    }
    let plain = !key
        .modifiers
        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
    match key.code {
        KeyCode::Char(character) if plain => browser.search.push(character),
        KeyCode::Backspace => {
            browser.search.pop();
        }
        KeyCode::Enter | KeyCode::Esc => browser.searching = false,
        _ => return,
    }
    browser.selected = 0;
    commands.queue(refresh);
}

/// Brings what the browser shows up to its state: reads the session
/// selected if it is not, builds its tree, and previews the node
/// selected.
fn refresh(world: &mut World) {
    let Some(root) = browser(world) else {
        return;
    };
    let Some(browser) = world.get::<Browser>(root) else {
        return;
    };
    let which = selected(browser).cloned();
    let panes = browser.panes;
    let open = browser.open;
    let unread = match &which {
        Some(Which::Saved(id))
            if !browser.read.contains_key(id) && !browser.reading.contains(id) =>
        {
            Some(id.clone())
        }
        _ => None,
    };
    if let Some(id) = unread {
        start_reading(world, root, id);
    }
    let conversation = which
        .as_ref()
        .and_then(|which| conversation(world, root, which));
    let mut shown = None;
    edit(world, root, |browser| {
        let (Some(which), Some(conversation)) = (&which, &conversation) else {
            browser.tree = None;
            browser.rows.clear();
            return;
        };
        let stale = browser.tree.as_ref().is_none_or(|(of, _)| of != which);
        if stale || *which == Which::Live {
            if stale {
                browser.unfolded.clear();
            }
            let tree = build(conversation);
            let rows = rows(&tree, &browser.unfolded);
            if stale {
                let current = rows.iter().position(|row| match row {
                    TreeRow::Node { node, .. } => tree.nodes.get(*node).is_some_and(|n| n.current),
                    TreeRow::Fold { .. } => false,
                });
                browser.row = current.unwrap_or(0);
            }
            browser.row = browser.row.min(rows.len().saturating_sub(1));
            browser.rows = rows;
            browser.tree = Some((which.clone(), tree));
        }
        let node = browser.rows.get(browser.row).map(|row| match row {
            TreeRow::Node { node, .. } => *node,
            TreeRow::Fold { last, .. } => *last,
        });
        let tree = browser.tree.as_ref().map(|(_, tree)| tree);
        let last = node
            .and_then(|node| tree?.nodes.get(node))
            .map(|node| node.last);
        shown = match (open, last) {
            (true, _) => Some((conversation.view(), "read-only: Esc closes".to_owned())),
            (false, Some(last)) => place(conversation, last)
                .and_then(|(line, end)| conversation.line_conversation(line, end))
                .map(|shown| (shown, "preview".to_owned())),
            (false, None) => None,
        };
    });
    let (shown, title) = shown.unwrap_or_else(|| (Line::new(), "preview".to_owned()));
    // Opened, the preview is the box's only panel: the other two are
    // taken out until it closes.
    let within = |part, of| {
        TuiPanel::new(Placement::Within {
            panel: root,
            part,
            of,
        })
    };
    let [sessions, tree, preview] = panes;
    for (part, pane) in [(0, sessions), (1, tree)] {
        match (open, world.get::<TuiPanel>(pane).is_some()) {
            (true, true) => {
                world.entity_mut(pane).remove::<TuiPanel>();
            }
            (false, false) => {
                world.entity_mut(pane).insert(within(part, 3));
            }
            _ => {}
        }
    }
    let placement = if open { within(0, 1) } else { within(2, 3) };
    if let Some(mut panel) = world.get_mut::<TuiPanel>(preview) {
        panel.set_if_neq(placement);
    }
    if let Some(mut pane) = world.get_mut::<TranscriptPane>(preview) {
        pane.source = TranscriptSource::Value(shown);
        pane.title = Some(title);
    }
    world.write_message(RequestRedraw);
}

/// Reads saved session `id` off the main thread, on a task under `root`.
fn start_reading(world: &mut World, root: Entity, id: SessionId) {
    let dir = world.resource::<RigHome>().0.session(&id);
    let stores = world.resource::<SessionStores>().clone();
    let Some(wake) = world.get_resource::<Wake>().cloned() else {
        return;
    };
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let reading = id.clone();
    let work = async move {
        let store = stores.open(&dir).map_err(|failure| failure.to_string());
        let read = store.and_then(|store| {
            let read = read_line(&*store, None, None);
            read.map(|(_, conversation)| conversation)
                .map_err(|failure| failure.to_string())
        });
        Read(Some((reading, read)))
    };
    world.spawn((Running::spawn(pool, &wake, work), ChildOf(root)));
    edit(world, root, |browser| {
        browser.reading.insert(id);
    });
}

/// The conversation of `which`: the live agent's, or the saved session's
/// main agent's once read.
fn conversation(world: &mut World, root: Entity, which: &Which) -> Option<Line> {
    match which {
        Which::Live => {
            let agent = world.get_resource::<Shown>()?.0?;
            world
                .get::<Conversation>(agent)
                .map(|conversation| conversation.view())
        }
        Which::Saved(id) => match world.get::<Browser>(root)?.read.get(id)? {
            Ok(conversation) => Some(conversation.clone()),
            Err(_) => None,
        },
    }
}

/// The tree of `conversation`'s lines: the active line first, then each
/// kept branch, each turn a node, turns the lines share one node.
fn build(conversation: &Line) -> Tree {
    let mut tree = Tree::default();
    let kept = conversation
        .branches()
        .into_iter()
        .map(|kept| Some(kept.id));
    for (n, id) in std::iter::once(None).chain(kept).enumerate() {
        let length = conversation.line(id).map_or(0, |line| line.len());
        let Some(line) = conversation.line_conversation(id, length) else {
            continue;
        };
        let mut parent: Option<usize> = None;
        for (first, last, title) in turns(&line) {
            let siblings = match parent {
                None => &tree.roots,
                Some(parent) => match tree.nodes.get(parent) {
                    Some(node) => &node.children,
                    None => break,
                },
            };
            let same = |at: &&usize| {
                let node = tree.nodes.get(**at);
                node.is_some_and(|node| node.first == first && node.last == last)
            };
            let found = siblings.iter().find(same).copied();
            let at = match found {
                Some(at) => at,
                None => {
                    let at = tree.nodes.len();
                    tree.nodes.push(Node {
                        first,
                        last,
                        title,
                        children: Vec::new(),
                        current: false,
                    });
                    match parent.and_then(|parent| tree.nodes.get_mut(parent)) {
                        Some(node) => node.children.push(at),
                        None => tree.roots.push(at),
                    }
                    at
                }
            };
            parent = Some(at);
        }
        if let (0, Some(end)) = (n, parent.and_then(|end| tree.nodes.get_mut(end))) {
            end.current = true;
        }
    }
    let firsts: Vec<MessageKey> = tree.nodes.iter().map(|node| node.first).collect();
    let made = |at: &usize| firsts.get(*at).map(|key| (key.made, key.id));
    tree.roots.sort_by_key(made);
    for node in &mut tree.nodes {
        node.children.sort_by_key(made);
    }
    tree
}

/// Each turn of `line`: the user's prompt that starts it, its last entry
/// and the prompt's first line.
fn turns(line: &Line) -> Vec<(MessageKey, MessageKey, String)> {
    let starts: Vec<usize> = line
        .entries()
        .filter(|entry| entry.is_users_own())
        .map(|entry| entry.index)
        .collect();
    let length = line.messages().len();
    let ends = starts.iter().skip(1).copied().chain([length]);
    starts
        .iter()
        .zip(ends)
        .filter_map(|(&start, end)| {
            let first = line.key(start)?;
            let last = line.key(end.checked_sub(1)?)?;
            let typed = typed(line, start).unwrap_or_default();
            let title = title_of(typed.lines().next().unwrap_or_default());
            Some((first, last, title))
        })
        .collect()
}

/// The rows of `tree`: a run of four or more turns without branching
/// shows its first and last, the rest folded, unless `unfolded`.
fn rows(tree: &Tree, unfolded: &HashSet<usize>) -> Vec<TreeRow> {
    let mut rows = Vec::new();
    let depth = usize::from(tree.roots.len() > 1);
    for &root in &tree.roots {
        walk(tree, root, depth, unfolded, &mut rows);
    }
    rows
}

/// The rows of the run from `start` on, then of each branch it ends in.
fn walk(
    tree: &Tree,
    start: usize,
    depth: usize,
    unfolded: &HashSet<usize>,
    rows: &mut Vec<TreeRow>,
) {
    let children = |at: usize| {
        tree.nodes
            .get(at)
            .map_or(&[][..], |node| node.children.as_slice())
    };
    let mut run = vec![start];
    while let (Some(&last), true) = (run.last(), run.len() <= tree.nodes.len()) {
        match children(last) {
            [only] => run.push(*only),
            _ => break,
        }
    }
    match (run.as_slice(), unfolded.contains(&start)) {
        ([first, middle @ .., last], false) if middle.len() >= 2 => {
            rows.push(TreeRow::Node {
                node: *first,
                depth,
            });
            let folded = middle.last().copied().unwrap_or(*first);
            rows.push(TreeRow::Fold {
                start,
                last: folded,
                hidden: middle.len(),
                depth,
            });
            rows.push(TreeRow::Node { node: *last, depth });
        }
        _ => rows.extend(run.iter().map(|&node| TreeRow::Node { node, depth })),
    }
    let end = run.last().copied().unwrap_or(start);
    for &child in children(end) {
        walk(tree, child, depth + 1, unfolded, rows);
    }
}

/// Where the turn ending with entry `last` is in `conversation`: the line
/// through it (the active one first) and the end of the turn there.
fn place(conversation: &Line, last: MessageKey) -> Option<(Option<AsideId>, usize)> {
    let kept = conversation
        .branches()
        .into_iter()
        .map(|kept| Some(kept.id));
    std::iter::once(None).chain(kept).find_map(|id| {
        let line = conversation.line(id)?;
        let at = line
            .iter()
            .position(|entry| entry.id == Some(last.id) && entry.made == Some(last.made))?;
        Some((id, at + 1))
    })
}

/// Makes the turn ending with entry `last` the end of `conversation`'s
/// active line: switches to a line through it, and keeps what follows it
/// there as a branch, so the next prompt starts a new one.
fn go_to(conversation: &mut Line, last: MessageKey) -> Result<(), String> {
    let failed = |failure: rig_harness::rig_core::conversation::BranchError| failure.to_string();
    let (line, _) = place(conversation, last).ok_or("that turn is no longer in the session")?;
    if let Some(line) = line {
        conversation.switch_branch(line, by()).map_err(failed)?;
    }
    let end = conversation
        .entries()
        .position(|entry| entry.key() == last)
        .map(|at| at + 1)
        .ok_or("that turn is no longer on the line")?;
    if end < conversation.messages().len() {
        let title = format!("branch {}", conversation.branches().len() + 1);
        conversation.branch_off(end, title, by()).map_err(failed)?;
    }
    Ok(())
}

/// Resumes the selected session at tree node `node`: the live one in
/// place, a saved one edited while stopped and then run. Refused while an
/// agent's turn runs.
fn resume(world: &mut World, root: Entity, node: usize) {
    let Some(browser) = world.get::<Browser>(root) else {
        return;
    };
    let which = selected(browser).cloned();
    let tree = browser.tree.as_ref().map(|(_, tree)| tree);
    let last = tree
        .and_then(|tree| tree.nodes.get(node))
        .map(|node| node.last);
    let (Some(which), Some(last)) = (which, last) else {
        return;
    };
    let mut turns = world.query_filtered::<(), With<TurnOf>>();
    let done = if turns.iter(world).next().is_some() {
        Err("An agent is running: stop it (Esc in the main view) before resuming.".to_owned())
    } else {
        match &which {
            Which::Live => resume_here(world, last),
            Which::Saved(id) => resume_saved(world, id, last),
        }
    };
    match done {
        Ok(()) if which == Which::Live => world.entity_mut(root).despawn(),
        Ok(()) => {}
        Err(why) => {
            world.write_message(Notice::warn(None, why.clone()));
            if let Some(mut browser) = world.get_mut::<Browser>(root) {
                browser.said = Some(why);
            }
        }
    }
}

/// Goes on in this session from the turn ending with `last`.
fn resume_here(world: &mut World, last: MessageKey) -> Result<(), String> {
    let agent = world.get_resource::<Shown>().and_then(|shown| shown.0);
    let agent = agent.ok_or("no agent is shown")?;
    let mut conversation = world
        .get_mut::<Conversation>(agent)
        .ok_or("the agent has no conversation")?;
    go_to(&mut conversation, last)
}

/// Runs saved session `id` from the turn ending with `last`.
fn resume_saved(world: &mut World, id: &SessionId, last: MessageKey) -> Result<(), String> {
    let paths = world
        .get_resource::<SessionPaths>()
        .cloned()
        .ok_or("this app keeps no session")?;
    let stores = world.resource::<SessionStores>().clone();
    let dir = world.resource::<RigHome>().0.session(id);
    edit_stopped(&stores, &paths, &dir, |session| {
        let main = session.main_agent().map(|agent| agent.id.clone());
        let agent = main.and_then(|main| session.agent_mut(&main));
        let agent = agent.ok_or("the session has no agent")?;
        go_to(&mut agent.conversation, last)
    })?;
    world.trigger(Exit(ExitReason::Switch(Some(id.clone()))));
    Ok(())
}

/// The style of a pane's border: bold while it has the keys.
fn border(focus: &KeyFocus, pane: Entity) -> Style {
    match focus.0 == Some(pane) {
        true => Style::new().cyan().bold(),
        false => Style::new().dim(),
    }
}

/// `rows` from the one that keeps row `at` in sight in `height` rows.
fn in_sight(rows: Vec<Row<'static>>, at: usize, height: u16) -> Vec<Row<'static>> {
    let height = usize::from(height.saturating_sub(2)).max(1);
    let skip = (at + 1).saturating_sub(height);
    rows.into_iter().skip(skip).collect()
}

/// Clears the screen under the browser's panes.
fn draw_box(mut boxes: Query<&mut PanelCanvas, With<Browser>>) {
    for mut canvas in &mut boxes {
        canvas.render(Clear);
    }
}

/// Draws the sessions, the selected one lit, and the search.
fn draw_sessions(
    mut panes: Query<(Entity, &ChildOf, &mut PanelCanvas), With<SessionsPane>>,
    browsers: Query<&Browser>,
    focus: Res<KeyFocus>,
) {
    for (pane, &ChildOf(root), mut canvas) in &mut panes {
        let Ok(browser) = browsers.get(root) else {
            continue;
        };
        let mut rows: Vec<Row<'static>> = match &browser.sessions {
            None => vec![Row::from("reading…").dim()],
            Some(_) => visible(browser)
                .into_iter()
                .enumerate()
                .map(|(at, row)| {
                    let text = format!("{}  · {}", row.title, row.detail);
                    match at == browser.selected {
                        true => Row::from(format!("▶ {text}")).reversed(),
                        false => Row::from(format!("  {text}")),
                    }
                })
                .collect(),
        };
        rows = in_sight(
            rows,
            browser.selected,
            canvas.area().height.saturating_sub(1),
        );
        let search = match (browser.searching, browser.search.is_empty()) {
            (true, _) => format!(" /{}▏", browser.search),
            (false, false) => format!(" /{} ", browser.search),
            (false, true) => " / search ".to_owned(),
        };
        let block = Block::bordered()
            .title(" sessions ")
            .title_bottom(search)
            .border_style(border(&focus, pane));
        canvas.render(Paragraph::new(rows).block(block));
    }
}

/// Draws the selected session's tree, the selected row lit, and what the
/// last action said.
fn draw_tree(
    mut panes: Query<(Entity, &ChildOf, &mut PanelCanvas), With<TreePane>>,
    browsers: Query<&Browser>,
    focus: Res<KeyFocus>,
) {
    for (pane, &ChildOf(root), mut canvas) in &mut panes {
        let Ok(browser) = browsers.get(root) else {
            continue;
        };
        let live = matches!(browser.tree, Some((Which::Live, _)));
        let rows: Vec<Row<'static>> = match &browser.tree {
            None => vec![
                Row::from(match selected(browser) {
                    Some(Which::Saved(id)) => match browser.read.get(id) {
                        Some(Err(why)) => format!("not readable: {why}"),
                        _ => "reading…".to_owned(),
                    },
                    _ => String::new(),
                })
                .dim(),
            ],
            Some((_, tree)) => (browser.rows.iter().enumerate())
                .map(|(at, row)| {
                    let text = label(tree, row, live);
                    match at == browser.row {
                        true => Row::from(format!("▶{text}")).reversed(),
                        false => Row::from(format!(" {text}")),
                    }
                })
                .collect(),
        };
        let rows = in_sight(rows, browser.row, canvas.area().height.saturating_sub(1));
        let help = match &browser.said {
            Some(said) => format!(" {said} "),
            None => " Enter resume · o open · Tab · Esc ".to_owned(),
        };
        let block = Block::bordered()
            .title(" tree ")
            .title_bottom(help)
            .border_style(border(&focus, pane));
        canvas.render(Paragraph::new(rows).block(block));
    }
}

/// One row of a tree: a turn's prompt, marked when it is a leaf, the
/// session's current leaf and, `live`, where this session is; or a fold.
fn label(tree: &Tree, row: &TreeRow, live: bool) -> String {
    match *row {
        TreeRow::Fold { hidden, depth, .. } => {
            format!("{}  ⋯ {hidden} more turns (Enter)", "  ".repeat(depth))
        }
        TreeRow::Node { node, depth } => {
            let Some(node) = tree.nodes.get(node) else {
                return String::new();
            };
            let leaf = node.children.is_empty();
            let mark = if leaf { "◆" } else { "•" };
            let current = match (node.current, live) {
                (true, true) => "  ◀ current, you are here",
                (true, false) => "  ◀ current",
                (false, _) if leaf => "  (leaf)",
                (false, _) => "",
            };
            format!("{}{mark} {}{current}", "  ".repeat(depth), node.title)
        }
    }
}

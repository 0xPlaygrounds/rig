//! Completion in the input.

use std::borrow::Cow;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_log::warn;
use bevy_tasks::{IoTaskPool, TaskPool};

use super::editor::{Editor, MainEditor};
use super::view::TuiView;
use rig_ecs::tasks::{Running, Wake};
use rig_harness::commands::SlashCommand;
use rig_harness::main_agent::ShownAgent;
use rig_harness::services::{ProjectRoot, WorkDir};
use rig_harness::watch::{FileChanges, FileWatch};
use rig_harness::workspace::ProjectWorkspace;
use rig_tools::attach::Reference;
use rig_tools::blocking;
use rig_tools::workspace::{Listing, Local, Workspace};

/// Items a completion lists.
const SHOWN: usize = 50;

/// Candidates a plugin adds to completion, on an entity of its own: for
/// a word starting with a marker anywhere in the input, such as `#` for
/// issues ([`CompletionSource::marker`]), or for an argument of a slash
/// command, such as `/model`'s models ([`CompletionSource::arg`]). The
/// view itself completes `/` at the start to a command name and `@` to a
/// path of the [`ProjectRoot`] (walked off the main thread, honouring
/// `.gitignore`, until a [`FileWatch`] of the root sees a change).
///
/// A candidate matches when it contains the word typed, ignoring case; the
/// earlier the match, the higher it ranks. Several sources for one trigger
/// add up; keep one current by inserting it again, such as when a lookup
/// of what is typed ([`InputContext::typed`]) finished: an open list
/// follows.
///
/// ```
/// use rig_harness::prelude::*;
/// use rig_tui::CompletionSource;
///
/// fn offer(mut commands: Commands) {
///     let branches = CompletionSource::arg("checkout").candidate("main", "").candidate("dev", "2 ahead");
///     let issues = CompletionSource::marker('#').candidate("142", "Crash on empty input");
///     commands.spawn_batch([branches, issues]);
/// }
/// ```
///
/// [`InputContext::typed`]: crate::InputContext::typed
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct CompletionSource {
    /// What it completes.
    pub trigger: CompletionTrigger,
    /// The candidates, in the order they are listed when they match alike.
    pub candidates: Vec<Candidate>,
}

/// What a [`CompletionSource`] completes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompletionTrigger {
    /// A word starting with this marker anywhere in the input, such as
    /// `#`. `/` and `@` are the view's own (commands and paths): a source
    /// with either completes nothing.
    Marker(char),
    /// The argument at `position` (from 0) of the slash command `command`
    /// (with or without its `/`).
    Arg {
        /// The command.
        command: String,
        /// The argument, from 0 for the first word after the command.
        position: usize,
    },
}

impl CompletionSource {
    /// No candidates yet for a word starting with `marker`.
    pub fn marker(marker: char) -> Self {
        Self::new(CompletionTrigger::Marker(marker))
    }

    /// No candidates yet for `command`'s first argument.
    pub fn arg(command: impl Into<String>) -> Self {
        Self::new(CompletionTrigger::Arg {
            command: command.into(),
            position: 0,
        })
    }

    fn new(trigger: CompletionTrigger) -> Self {
        Self {
            trigger,
            candidates: Vec::new(),
        }
    }

    /// The same, completing the argument at `position` (from 0) instead;
    /// a marker's source is left as it is.
    pub fn at(mut self, at: usize) -> Self {
        if let CompletionTrigger::Arg { position, .. } = &mut self.trigger {
            *position = at;
        }
        self
    }

    /// The same, also offering `text`, described by `detail` (may be
    /// empty).
    pub fn candidate(mut self, text: impl Into<String>, detail: impl Into<String>) -> Self {
        self.candidates.push(Candidate {
            text: text.into(),
            detail: detail.into(),
        });
        self
    }

    /// Whether it completes the token `kind` of `command` (without its
    /// `/`).
    fn is_for(&self, kind: CompletionKind, command: &str) -> bool {
        match (&self.trigger, kind) {
            (CompletionTrigger::Marker(marker), CompletionKind::Marker(typed)) => *marker == typed,
            (
                CompletionTrigger::Arg {
                    command: own,
                    position,
                },
                CompletionKind::Arg(typed),
            ) => own.trim_start_matches('/') == command && *position == typed,
            _ => false,
        }
    }

    /// The marker it adds to those [`token`] knows, if any.
    fn marker_char(&self) -> Option<char> {
        match self.trigger {
            CompletionTrigger::Marker(marker) if !VIEW_MARKERS.contains(&marker) => Some(marker),
            _ => None,
        }
    }
}

/// The markers the view completes itself: commands and paths.
const VIEW_MARKERS: [char; 2] = ['/', '@'];

/// What a [`Completion`] completes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CompletionKind {
    /// A command name after a leading `/`.
    Command,
    /// The argument at this position (from 0) of the command at the
    /// start, `/name`.
    Arg(usize),
    /// A path after `@`.
    Path,
    /// A word after a [`CompletionSource`]'s marker.
    Marker(char),
}

/// One candidate of a completion.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Candidate {
    /// What is shown and inserted, without the marker (`/`, `@`, `#`).
    pub text: String,
    /// A description shown after it.
    pub detail: String,
}

/// The open completion list of the input, as
/// [`InputContext::completion`](crate::InputContext::completion) gives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Completion {
    /// What it completes.
    pub kind: CompletionKind,
    /// The byte offset in the input of the token it completes: its
    /// marker, or an argument's first character.
    pub start: usize,
    /// The text typed after the marker, or the argument typed.
    pub query: String,
    /// What it lists, best first.
    pub items: Vec<Candidate>,
    /// The position of the item Tab and Enter put in, among `items`.
    pub selected: usize,
    /// Whether the items came from a finished walk.
    complete: bool,
    /// Whether the token is a quoted path not closed yet, which Enter
    /// completes and closes before it sends.
    open: bool,
}

impl Completion {
    /// What accepting the selected item puts in place of the token: the
    /// marker, the item, and a space unless it is a directory. A path is
    /// written as a [`Reference`] reads it back, a directory's left open
    /// for the path below it.
    pub(crate) fn chosen(&self) -> Option<String> {
        let item = self.items.get(self.selected)?;
        if self.kind == CompletionKind::Path {
            let reference = Reference::format(&item.text);
            return Some(match item.text.ends_with('/') {
                true => reference.strip_suffix('"').unwrap_or(&reference).to_owned(),
                false => format!("{reference} "),
            });
        }
        let space = if item.text.ends_with('/') { "" } else { " " };
        let marker = self.kind.marker().map(String::from).unwrap_or_default();
        Some(format!("{marker}{}{space}", item.text))
    }

    /// Whether the token already is the selected item exactly, so Enter
    /// sends rather than completes: for a path, as its [`Reference`]
    /// writes it.
    pub(crate) fn is_exact(&self) -> bool {
        let Some(item) = self.items.get(self.selected) else {
            return false;
        };
        match self.kind {
            CompletionKind::Path => {
                !self.open && Reference::format(&item.text) == format!("@{}", self.query)
            }
            _ => item.text == self.query,
        }
    }

    pub(crate) fn select(&mut self, step: isize) {
        let count = self.items.len();
        if count > 0 {
            self.selected = self.selected.saturating_add_signed(step).min(count - 1);
        }
    }
}

impl CompletionKind {
    /// What the token starts with; an argument starts with no marker.
    pub fn marker(self) -> Option<char> {
        match self {
            Self::Command => Some('/'),
            Self::Arg(_) => None,
            Self::Path => Some('@'),
            Self::Marker(marker) => Some(marker),
        }
    }
}

/// The project's paths for `@` completion.
#[derive(Resource, Default)]
pub(crate) struct FileIndex {
    paths: Arc<Vec<String>>,
    /// Whether a walk runs, to end as [`WalkedPaths`].
    walking: bool,
    /// Whether the paths are the project's as its watch last saw it.
    fresh: bool,
    /// The watch of the root walked, from the first `@` on.
    watch: Option<Entity>,
    /// The root the paths are relative to.
    root: Option<PathBuf>,
}

/// The root a walk went through and the paths it found, on the walk's
/// entity.
#[derive(Component)]
pub(crate) struct WalkedPaths(PathBuf, Arc<Vec<String>>);

impl From<(PathBuf, Vec<String>)> for WalkedPaths {
    fn from((root, paths): (PathBuf, Vec<String>)) -> Self {
        Self(root, Arc::new(paths))
    }
}

impl FileIndex {
    /// Watches `root`, then starts a walk of it in `workspace` unless one
    /// runs or nothing changed since the last.
    fn refresh(
        &mut self,
        wake: &Wake,
        (workspace, root): (Arc<dyn Workspace>, PathBuf),
        commands: &mut Commands,
    ) {
        if self.root.as_ref() != Some(&root) {
            if let Some(watch) = self.watch.take() {
                commands.entity(watch).despawn();
            }
            (self.paths, self.fresh) = (Arc::default(), false);
            self.root = Some(root.clone());
        }
        if self.watch.is_none() {
            let watch = (Name::new("rig-tui: @ paths"), FileWatch::tree(&root));
            self.watch = Some(commands.spawn(watch).id());
        }
        if self.walking || self.fresh {
            return;
        }
        let walked = root.clone();
        let walk = blocking(move || Ok(walk(&*workspace, &root)));
        let walk = async move {
            let paths = walk.await.unwrap_or_else(|failure| {
                warn!("could not list the project's files: {failure}");
                Vec::new()
            });
            (walked, paths)
        };
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        let walking = Running::spawn_into::<WalkedPaths, _>(pool, wake, walk);
        commands.spawn((Name::new("rig-tui: @ paths walk"), walking));
        // A change seen from here on makes the walk stale.
        (self.walking, self.fresh) = (true, true);
    }
}

/// The files under `root` in `workspace` and the directories that hold
/// them, relative to it, directories ending in `/`, skipping what
/// `.gitignore` and hidden names leave out.
fn walk(workspace: &dyn Workspace, root: &Path) -> Vec<String> {
    let files = match workspace.files(root, None, Listing::Visible) {
        Ok(listed) => listed.files,
        Err(failure) => {
            warn!("could not list the project's files: {failure}");
            return Vec::new();
        }
    };
    // `/` between directories; on Unix a `\` is part of a name. A first
    // name with a `:` goes after `./`, so it is not read as `HOST:PATH`.
    let text = |path: &Path| {
        let text = match cfg!(windows) {
            true => path.to_string_lossy().replace('\\', "/"),
            false => path.to_string_lossy().into_owned(),
        };
        match text
            .split('/')
            .next()
            .is_some_and(|first| first.contains(':'))
        {
            true => format!("./{text}"),
            false => text,
        }
    };
    let mut paths = Vec::new();
    let mut directories = HashSet::new();
    for file in &files {
        let Ok(relative) = file.strip_prefix(root) else {
            continue;
        };
        let mut above: Vec<&Path> = relative.ancestors().skip(1).collect();
        above.retain(|directory| !directory.as_os_str().is_empty());
        for directory in above.into_iter().rev() {
            if directories.insert(directory) {
                paths.push(format!("{}/", text(directory)));
            }
        }
        paths.push(text(relative));
    }
    paths
}

/// The `/command`, argument, `@path` or marked token that ends at the
/// cursor, if any, given the markers sources add; an argument's with the
/// command's name and its position.
fn token<'a>(
    text: &'a str,
    cursor: usize,
    markers: &[char],
) -> Option<(CompletionKind, usize, Cow<'a, str>, &'a str)> {
    let before = text.get(..cursor)?;
    if let Some(reference) = Reference::typing(before) {
        let start = reference.span.start;
        return Some((CompletionKind::Path, start, reference.path, ""));
    }
    let start = before
        .char_indices()
        .rfind(|(_, character)| character.is_whitespace())
        .map_or(0, |(index, character)| index + character.len_utf8());
    let word = before.get(start..)?;
    if let Some(query) = word.strip_prefix('/')
        && start == 0
    {
        return Some((CompletionKind::Command, start, query.into(), ""));
    }
    if let Some(marker) = word.chars().next().filter(|first| markers.contains(first)) {
        let query = word.get(marker.len_utf8()..)?;
        return Some((CompletionKind::Marker(marker), start, query.into(), ""));
    }
    let command = text.strip_prefix('/')?.split_whitespace().next()?;
    let position = before
        .get(..start)?
        .split_whitespace()
        .count()
        .checked_sub(1)?;
    Some((CompletionKind::Arg(position), start, word.into(), command))
}

/// What completion lists: the slash commands, and the candidates plugins
/// add ([`CompletionSource`]).
#[derive(SystemParam)]
pub(crate) struct Sources<'w, 's> {
    commands: Query<'w, 's, (&'static Name, &'static SlashCommand)>,
    added: Query<'w, 's, &'static CompletionSource>,
    workspace: Option<Res<'w, ProjectWorkspace>>,
    root: Option<Res<'w, ProjectRoot>>,
    shown: Option<Res<'w, ShownAgent>>,
    dirs: Query<'w, 's, &'static WorkDir>,
}

impl Sources<'_, '_> {
    /// Where `@` paths are: the app's workspace, else this machine, and
    /// where the shown agent's references resolve from, its [`WorkDir`],
    /// else the project's root, else the working directory.
    fn project(&self) -> (Arc<dyn Workspace>, PathBuf) {
        let workspace = self.workspace.as_ref();
        let workspace =
            workspace.map_or_else(|| Arc::new(Local) as Arc<dyn Workspace>, |it| it.0.clone());
        let shown = self.shown.as_ref().and_then(|shown| shown.0);
        let dir = shown.and_then(|agent| self.dirs.get(agent).ok());
        let root = match (dir, self.root.as_ref()) {
            (Some(dir), _) => dir.0.clone(),
            (None, Some(root)) => root.0.clone(),
            (None, None) => PathBuf::from("."),
        };
        (workspace, root)
    }
}

/// The markers `sources` add.
fn markers(sources: &Query<&CompletionSource>) -> Vec<char> {
    let mut markers: Vec<char> = sources
        .iter()
        .filter_map(CompletionSource::marker_char)
        .collect();
    markers.sort_unstable();
    markers.dedup();
    markers
}

/// The token at the cursor of `input` that completion completes, and what
/// is typed of it after its marker.
pub(crate) fn typed(
    input: &Editor,
    sources: &Query<&CompletionSource>,
) -> Option<(CompletionKind, String)> {
    let (kind, _, query, _) = token(input.text(), input.cursor(), &markers(sources))?;
    Some((kind, query.into_owned()))
}

/// Opens, updates or closes the completion after the input, the paths or
/// the sources changed (`renewed`: an open list is listed again). A list
/// closed with Esc stays closed until the token moves, and a prompt of the
/// history shown unchanged opens none, so Up and Down walk past it.
pub(crate) fn update(
    view: &mut TuiView,
    input: &Editor,
    index: &mut FileIndex,
    (sources, renewed): (&Sources, bool),
    wake: &Wake,
    commands: &mut Commands,
) {
    if input.recalled() {
        view.completion = None;
        view.dismissed = None;
        return;
    }
    let found = token(input.text(), input.cursor(), &markers(&sources.added));
    let Some((kind, start, query, command)) = found else {
        view.completion = None;
        view.dismissed = None;
        return;
    };
    let query = &*query;
    let open = (input.text().get(..input.cursor()))
        .and_then(Reference::typing)
        .is_some_and(|reference| reference.quoted);
    if view.dismissed == Some(start) {
        view.completion = None;
        return;
    }
    view.dismissed = None;
    if kind == CompletionKind::Path {
        index.refresh(wake, sources.project(), commands);
    }
    let complete = !index.walking;
    if let Some(open) = &view.completion
        && open.kind == kind
        && open.start == start
        && open.query == query
        && (kind != CompletionKind::Path || open.complete == complete)
        && !renewed
    {
        return;
    }
    let items = match kind {
        CompletionKind::Command => {
            let mut ranked: Vec<(usize, &str, &SlashCommand)> = sources
                .commands
                .iter()
                .filter_map(|(name, command)| {
                    let name = name.as_str().trim_start_matches('/');
                    Some((match_at(query, name)?, name, command))
                })
                .collect();
            ranked.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            ranked
                .into_iter()
                .map(|(_, name, command)| Candidate {
                    text: name.to_owned(),
                    detail: command.help.clone(),
                })
                .collect()
        }
        CompletionKind::Arg(_) | CompletionKind::Marker(_) => {
            let candidates = sources
                .added
                .iter()
                .filter(|source| source.is_for(kind, command))
                .flat_map(|source| &source.candidates);
            let mut ranked: Vec<(usize, &Candidate)> = candidates
                .filter_map(|item| Some((match_at(query, &item.text)?, item)))
                .collect();
            ranked.sort_by_key(|(at, _)| *at);
            ranked.into_iter().map(|(_, item)| item.clone()).collect()
        }
        CompletionKind::Path => path_items(&index.paths, query),
    };
    let query = query.to_owned();
    // A list listed again keeps the item selected, where it still is.
    let selected = (view.completion.as_ref())
        .filter(|open| open.kind == kind && open.start == start)
        .and_then(|open| open.items.get(open.selected))
        .and_then(|was| items.iter().position(|item| item == was))
        .unwrap_or(0);
    view.completion = (!items.is_empty()).then_some(Completion {
        kind,
        start,
        query,
        items,
        selected,
        complete,
        open,
    });
}

/// The best [`SHOWN`] paths for `query`: those whose file name matches
/// before those where only a directory does, then by where it matches,
/// shallower paths first.
fn path_items(paths: &[String], query: &str) -> Vec<Candidate> {
    let mut ranked: Vec<((bool, usize, usize), &String)> = paths
        .iter()
        .filter_map(|path| {
            let trimmed = path.trim_end_matches('/');
            let name = trimmed.rsplit('/').next().unwrap_or(trimmed);
            let (in_directory, at) = match match_at(query, name) {
                Some(at) => (false, at),
                None => (true, match_at(query, path)?),
            };
            Some(((in_directory, at, trimmed.matches('/').count()), path))
        })
        .collect();
    ranked.sort();
    ranked
        .into_iter()
        .take(SHOWN)
        .map(|(_, path)| Candidate {
            text: path.clone(),
            detail: String::new(),
        })
        .collect()
}

/// Walks the project again once its watch saw a change while an `@` path
/// is typed, and takes a finished walk to update the completion with it.
pub(crate) fn receive_paths(
    mut index: ResMut<FileIndex>,
    mut changes: MessageReader<FileChanges>,
    walked: Option<Single<(Entity, &WalkedPaths)>>,
    mut view: ResMut<TuiView>,
    input: Query<&Editor, With<MainEditor>>,
    sources: Sources,
    (wake, mut commands): (Res<Wake>, Commands),
) {
    let Ok(input) = input.single() else {
        return;
    };
    let watch = index.watch;
    let changed = changes.read().any(|change| Some(change.watch) == watch);
    if changed {
        index.fresh = false;
    }
    let received = walked.map(|walked| walked.into_inner());
    if let Some((walk, WalkedPaths(root, paths))) = received {
        // A walk of a root since left behind is stale.
        if index.root.as_ref() == Some(root) {
            index.paths = paths.clone();
        }
        index.walking = false;
        commands.entity(walk).despawn();
    }
    if changed || received.is_some() {
        update(
            &mut view,
            input,
            &mut index,
            (&sources, false),
            &wake,
            &mut commands,
        );
    }
}

/// Brings the completion up to date with the input once it changed, by a
/// key or a plugin, and lists it again once a plugin's sources changed.
pub(crate) fn follow_input(
    mut index: ResMut<FileIndex>,
    mut view: ResMut<TuiView>,
    input: Query<Ref<Editor>, With<MainEditor>>,
    sources: Sources,
    (changed, mut removed): (
        Query<(), Changed<CompletionSource>>,
        RemovedComponents<CompletionSource>,
    ),
    (wake, mut commands): (Res<Wake>, Commands),
) {
    let renewed = !changed.is_empty() || removed.read().count() > 0;
    let Ok(input) = input.single() else {
        return;
    };
    if input.is_changed() || renewed {
        update(
            &mut view,
            &input,
            &mut index,
            (&sources, renewed),
            &wake,
            &mut commands,
        );
    }
}

/// Where `query` first appears in `text`, ignoring case, or `None`.
fn match_at(query: &str, text: &str) -> Option<usize> {
    text.to_lowercase().find(&query.to_lowercase())
}

#[cfg(test)]
mod tests;

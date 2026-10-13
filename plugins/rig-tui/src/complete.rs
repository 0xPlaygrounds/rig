//! Completion in the input: `/` at the start completes a command name,
//! a word after a command one of the candidates plugins contribute for
//! its argument ([`ArgCompletion`]), and `@` completes a path of the
//! project. Paths come from a walk of the
//! [`ProjectRoot`] in the app's [`ProjectWorkspace`] that honours
//! `.gitignore`, done on a thread of its own, whatever the workspace, and
//! kept until a [`FileWatch`] of the root says something there changed.
//! An item matches when it contains the query, ignoring case; the earlier
//! the match, the higher it ranks.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_log::warn;
use crossbeam_channel::Receiver;

use super::editor::TuiInput;
use super::view::TuiView;
use rig_ecs::calls::Wake;
use rig_harness::commands::SlashCommand;
use rig_harness::services::ProjectRoot;
use rig_harness::watch::{FileChanges, FileWatch};
use rig_harness::workspace::ProjectWorkspace;
use rig_tools::workspace::{Local, Workspace};

/// Items a completion lists.
const SHOWN: usize = 50;

/// Candidates for an argument of a slash command, such as `/model`'s
/// models, on an entity of the plugin's: the word typed at the argument's
/// position, the first after the command unless [`ArgCompletion::at`] says
/// otherwise, completes to the ones it matches. Several for one command
/// add up; keep one current by inserting it again.
///
/// ```
/// use rig_harness::prelude::*;
/// use rig_tui::ArgCompletion;
///
/// fn offer_branches(mut commands: Commands) {
///     let branches = ArgCompletion::new("checkout").candidate("main", "").candidate("dev", "2 ahead");
///     commands.spawn(branches);
/// }
/// ```
#[derive(Component, Clone, Debug, Default, PartialEq, Eq)]
pub struct ArgCompletion {
    /// The command, with or without its `/`.
    pub command: String,
    /// The candidates, in the order they are listed when they match alike.
    pub candidates: Vec<Candidate>,
    /// The argument they complete, from 0 for the first word after the
    /// command.
    pub position: usize,
}

impl ArgCompletion {
    /// No candidates yet for `command`'s argument.
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            candidates: Vec::new(),
            position: 0,
        }
    }

    /// The same, completing the argument at `position` (from 0) instead.
    pub fn at(mut self, position: usize) -> Self {
        self.position = position;
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

    /// Whether it completes argument `position` of `command` (without its
    /// `/`).
    fn is_for(&self, command: &str, position: usize) -> bool {
        self.command.trim_start_matches('/') == command && self.position == position
    }
}

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
}

/// One candidate of a completion.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Candidate {
    /// What is shown and inserted, without the `/` or `@`.
    pub text: String,
    /// A description shown after it.
    pub detail: String,
}

/// The open completion list of the input, as
/// [`TuiInputState::completion`](crate::TuiInputState::completion) gives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Completion {
    /// What it completes.
    pub kind: CompletionKind,
    /// The byte offset in the input of the token it completes: its `/` or
    /// `@`, or an argument's first character.
    pub start: usize,
    /// The text typed after the `/` or `@`, or the argument typed.
    pub query: String,
    /// What it lists, best first.
    pub items: Vec<Candidate>,
    /// The position of the item Tab and Enter put in, among `items`.
    pub selected: usize,
    /// Whether the items came from a finished walk.
    complete: bool,
}

impl Completion {
    /// What accepting the selected item puts in place of the token: the
    /// marker, the item, and a space unless it is a directory.
    pub(crate) fn chosen(&self) -> Option<String> {
        let item = self.items.get(self.selected)?;
        let space = if item.text.ends_with('/') { "" } else { " " };
        Some(format!("{}{}{space}", self.kind.marker(), item.text))
    }

    /// Whether the token already is the selected item exactly, so Enter
    /// sends rather than completes.
    pub(crate) fn is_exact(&self) -> bool {
        self.items
            .get(self.selected)
            .is_some_and(|item| item.text == self.query)
    }

    pub(crate) fn select(&mut self, step: isize) {
        let count = self.items.len();
        if count > 0 {
            self.selected = self.selected.saturating_add_signed(step).min(count - 1);
        }
    }
}

impl CompletionKind {
    /// What the token starts with.
    pub(crate) fn marker(self) -> &'static str {
        match self {
            Self::Command => "/",
            Self::Arg(_) => "",
            Self::Path => "@",
        }
    }
}

/// The project's paths for `@` completion.
#[derive(Resource, Default)]
pub(crate) struct FileIndex {
    paths: Arc<Vec<String>>,
    walking: Option<Receiver<Vec<String>>>,
    /// Whether the paths are the project's as its watch last saw it.
    fresh: bool,
    /// The watch of the project's root, from the first `@` on.
    watch: Option<Entity>,
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
        if self.watch.is_none() {
            let watch = (Name::new("rig-tui: @ paths"), FileWatch::tree(&root));
            self.watch = Some(commands.spawn(watch).id());
        }
        if self.walking.is_some() || self.fresh {
            return;
        }
        let (sender, receiver) = crossbeam_channel::bounded(1);
        let wake = wake.clone();
        let started = std::thread::Builder::new()
            .name("rig-harness-paths".to_owned())
            .spawn(move || {
                sender.send(walk(&*workspace, &root)).ok();
                wake.wake();
            });
        match started {
            // A change seen from here on makes the walk stale.
            Ok(_) => (self.walking, self.fresh) = (Some(receiver), true),
            Err(failure) => warn!("could not list the project's files: {failure}"),
        }
    }

    /// Takes a finished walk's paths; returns whether there were any new.
    fn receive(&mut self) -> bool {
        let Some(paths) = self.walking.as_ref().and_then(|walk| walk.try_recv().ok()) else {
            return false;
        };
        self.paths = Arc::new(paths);
        self.walking = None;
        true
    }
}

/// The files under `root` in `workspace` and the directories that hold
/// them, relative to it, directories ending in `/`, skipping what
/// `.gitignore` and hidden names leave out.
fn walk(workspace: &dyn Workspace, root: &Path) -> Vec<String> {
    let files = match workspace.files(root, None, false) {
        Ok(files) => files,
        Err(failure) => {
            warn!("could not list the project's files: {failure}");
            return Vec::new();
        }
    };
    let text = |path: &Path| path.to_string_lossy().replace('\\', "/");
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

/// The `/command`, argument or `@path` token that ends at the cursor, if
/// any; an argument's with the command's name and its position.
fn token(text: &str, cursor: usize) -> Option<(CompletionKind, usize, &str, &str)> {
    let before = text.get(..cursor)?;
    let start = before
        .char_indices()
        .rfind(|(_, character)| character.is_whitespace())
        .map_or(0, |(index, character)| index + character.len_utf8());
    let word = before.get(start..)?;
    if let Some(query) = word.strip_prefix('/')
        && start == 0
    {
        return Some((CompletionKind::Command, start, query, ""));
    }
    if let Some(query) = word.strip_prefix('@') {
        return Some((CompletionKind::Path, start, query, ""));
    }
    let command = text.strip_prefix('/')?.split_whitespace().next()?;
    let position = before
        .get(..start)?
        .split_whitespace()
        .count()
        .checked_sub(1)?;
    Some((CompletionKind::Arg(position), start, word, command))
}

/// What completion lists: the slash commands, and the candidates plugins
/// contribute for their arguments.
#[derive(SystemParam)]
pub(crate) struct Sources<'w, 's> {
    commands: Query<'w, 's, (&'static Name, &'static SlashCommand)>,
    arguments: Query<'w, 's, &'static ArgCompletion>,
    workspace: Option<Res<'w, ProjectWorkspace>>,
    root: Option<Res<'w, ProjectRoot>>,
}

impl Sources<'_, '_> {
    /// Where `@` paths are: the app's workspace, else this machine, and the
    /// project's root, else the working directory.
    fn project(&self) -> (Arc<dyn Workspace>, PathBuf) {
        let workspace = self.workspace.as_ref();
        let workspace =
            workspace.map_or_else(|| Arc::new(Local) as Arc<dyn Workspace>, |it| it.0.clone());
        let root = self.root.as_ref();
        (
            workspace,
            root.map_or_else(|| PathBuf::from("."), |root| root.0.clone()),
        )
    }
}

/// Opens, updates or closes the completion after the input or the paths
/// changed. A list closed with Esc stays closed until the token moves.
pub(crate) fn update(
    view: &mut TuiView,
    input: &TuiInput,
    index: &mut FileIndex,
    sources: &Sources,
    wake: &Wake,
    commands: &mut Commands,
) {
    let found = token(input.text(), input.cursor());
    let Some((kind, start, query, command)) = found else {
        view.completion = None;
        view.dismissed = None;
        return;
    };
    if view.dismissed == Some(start) {
        view.completion = None;
        return;
    }
    view.dismissed = None;
    if kind == CompletionKind::Path {
        index.refresh(wake, sources.project(), commands);
    }
    let complete = index.walking.is_none();
    if let Some(open) = &view.completion
        && open.kind == kind
        && open.start == start
        && open.query == query
        && (kind != CompletionKind::Path || open.complete == complete)
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
        CompletionKind::Arg(position) => {
            let candidates = sources
                .arguments
                .iter()
                .filter(|completion| completion.is_for(command, position))
                .flat_map(|completion| &completion.candidates);
            let mut ranked: Vec<(usize, &Candidate)> = candidates
                .filter_map(|item| Some((match_at(query, &item.text)?, item)))
                .collect();
            ranked.sort_by_key(|(at, _)| *at);
            ranked.into_iter().map(|(_, item)| item.clone()).collect()
        }
        CompletionKind::Path => path_items(&index.paths, query),
    };
    let query = query.to_owned();
    view.completion = (!items.is_empty()).then_some(Completion {
        kind,
        start,
        query,
        items,
        selected: 0,
        complete,
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
    mut view: ResMut<TuiView>,
    input: Res<TuiInput>,
    sources: Sources,
    (wake, mut commands): (Res<Wake>, Commands),
) {
    let watch = index.watch;
    let changed = changes.read().any(|change| Some(change.watch) == watch);
    if changed {
        index.fresh = false;
    }
    let received = index.walking.is_some() && index.receive();
    if changed || received {
        update(
            &mut view,
            &input,
            &mut index,
            &sources,
            &wake,
            &mut commands,
        );
    }
}

/// Brings the completion up to date with the input once it changed, by a
/// key or a plugin.
pub(crate) fn follow_input(
    mut index: ResMut<FileIndex>,
    mut view: ResMut<TuiView>,
    input: Res<TuiInput>,
    sources: Sources,
    (wake, mut commands): (Res<Wake>, Commands),
) {
    if input.is_changed() {
        update(
            &mut view,
            &input,
            &mut index,
            &sources,
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

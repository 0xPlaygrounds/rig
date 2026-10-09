//! Completion in the input: `/` at the start completes a command name and
//! `@` completes a path of the project. Paths come from a walk of the
//! working directory that honours `.gitignore`, done on a thread of its own
//! and kept for a while. An item matches when it contains the query,
//! ignoring case; the earlier the match, the higher it ranks.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy_ecs::prelude::*;
use bevy_log::warn;
use crossbeam_channel::Receiver;

use super::view::TuiView;
use rig_ecs::calls::Wake;
use rig_ecs::commands::SlashCommand;

/// Items a completion lists.
const SHOWN: usize = 50;
/// Paths the walk collects at most.
const MOST_PATHS: usize = 50_000;
/// How long a walk's paths are used before the next `@` walks again.
const FRESH: Duration = Duration::from_secs(30);

/// What is being completed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// A command name after a leading `/`.
    Command,
    /// A path after `@`.
    Path,
}

/// One candidate.
pub(crate) struct Item {
    /// What is shown and inserted, without the `/` or `@`.
    pub(crate) text: String,
    /// A description shown after it.
    pub(crate) detail: String,
}

/// The open completion list.
pub(crate) struct Completion {
    pub(crate) kind: Kind,
    /// The byte offset of the `/` or `@` in the input.
    pub(crate) start: usize,
    /// The text typed after it.
    query: String,
    pub(crate) items: Vec<Item>,
    pub(crate) selected: usize,
    /// Whether the items came from a finished walk.
    complete: bool,
}

impl Completion {
    /// What accepting the selected item puts in place of the token: the
    /// marker, the item, and a space unless it is a directory.
    pub(crate) fn chosen(&self) -> Option<String> {
        let item = self.items.get(self.selected)?;
        let marker = match self.kind {
            Kind::Command => '/',
            Kind::Path => '@',
        };
        let space = if item.text.ends_with('/') { "" } else { " " };
        Some(format!("{marker}{}{space}", item.text))
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

/// The project's paths for `@` completion.
#[derive(Resource, Default)]
pub(crate) struct FileIndex {
    paths: Arc<Vec<String>>,
    walking: Option<Receiver<Vec<String>>>,
    walked: Option<Instant>,
}

impl FileIndex {
    /// Starts a walk unless one runs or the last one is fresh.
    fn refresh(&mut self, wake: &Wake) {
        if self.walking.is_some() || self.walked.is_some_and(|walked| walked.elapsed() < FRESH) {
            return;
        }
        let (sender, receiver) = crossbeam_channel::bounded(1);
        let wake = wake.clone();
        let started = std::thread::Builder::new()
            .name("rig-harness-paths".to_owned())
            .spawn(move || {
                sender.send(walk()).ok();
                wake.wake();
            });
        match started {
            Ok(_) => self.walking = Some(receiver),
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
        self.walked = Some(Instant::now());
        true
    }
}

/// The working directory's files and directories, relative, directories
/// ending in `/`, skipping what `.gitignore` and hidden names leave out.
fn walk() -> Vec<String> {
    let mut paths = Vec::new();
    for entry in ignore::WalkBuilder::new(".").build().flatten() {
        if entry.depth() == 0 {
            continue;
        }
        let path = entry.path();
        let path = path.strip_prefix(".").unwrap_or(path);
        let mut text = path.to_string_lossy().replace('\\', "/");
        if entry.file_type().is_some_and(|kind| kind.is_dir()) {
            text.push('/');
        }
        paths.push(text);
        if paths.len() >= MOST_PATHS {
            break;
        }
    }
    paths
}

/// The `/command` or `@path` token that ends at the cursor, if any.
fn token(text: &str, cursor: usize) -> Option<(Kind, usize, &str)> {
    let before = text.get(..cursor)?;
    let start = before
        .char_indices()
        .rfind(|(_, character)| character.is_whitespace())
        .map_or(0, |(index, character)| index + character.len_utf8());
    let word = before.get(start..)?;
    if let Some(query) = word.strip_prefix('/')
        && start == 0
    {
        return Some((Kind::Command, start, query));
    }
    let query = word.strip_prefix('@')?;
    Some((Kind::Path, start, query))
}

/// Opens, updates or closes the completion after the input or the paths
/// changed. A list closed with Esc stays closed until the token moves.
pub(crate) fn update(
    view: &mut TuiView,
    index: &mut FileIndex,
    commands: &Query<&SlashCommand>,
    wake: &Wake,
) {
    let found = token(view.editor.text(), view.editor.cursor());
    let Some((kind, start, query)) = found else {
        view.completion = None;
        view.dismissed = None;
        return;
    };
    if view.dismissed == Some(start) {
        view.completion = None;
        return;
    }
    view.dismissed = None;
    if kind == Kind::Path {
        index.refresh(wake);
    }
    let complete = index.walking.is_none();
    if let Some(open) = &view.completion
        && open.kind == kind
        && open.start == start
        && open.query == query
        && (kind == Kind::Command || open.complete == complete)
    {
        return;
    }
    let items = match kind {
        Kind::Command => {
            let mut ranked: Vec<(usize, &SlashCommand)> = commands
                .iter()
                .filter_map(|command| Some((match_at(query, &command.name)?, command)))
                .collect();
            ranked.sort_by(|a, b| (a.0, &a.1.name).cmp(&(b.0, &b.1.name)));
            ranked
                .into_iter()
                .map(|(_, command)| Item {
                    text: command.name.clone(),
                    detail: command.help.clone(),
                })
                .collect()
        }
        Kind::Path => path_items(&index.paths, query),
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
fn path_items(paths: &[String], query: &str) -> Vec<Item> {
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
        .map(|(_, path)| Item {
            text: path.clone(),
            detail: String::new(),
        })
        .collect()
}

/// Takes a finished walk and updates an open `@` completion with it.
pub(crate) fn receive_paths(
    mut index: ResMut<FileIndex>,
    mut view: ResMut<TuiView>,
    commands: Query<&SlashCommand>,
    wake: Res<Wake>,
) {
    if index.walking.is_none() || !index.receive() {
        return;
    }
    update(&mut view, &mut index, &commands, &wake);
}

/// Where `query` first appears in `text`, ignoring case, or `None`.
fn match_at(query: &str, text: &str) -> Option<usize> {
    text.to_lowercase().find(&query.to_lowercase())
}

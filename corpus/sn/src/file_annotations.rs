//! Notes on lines of code that follow the code (spec SN-10).
//!
//! The `annotate` tool, and the user's `/note path:12-14 text` and
//! `/note rm n3`, attach a note to lines of a file, or update or delete one.
//! The notes are kept per project in `RIG_HOME/annotations/<project>.json`,
//! never in the file, so every later session has them. Each successful
//! `read` of a file with notes gets them in its result, in Review, right
//! after their lines (a partial read only those whose lines it shows), and
//! each stale note at its end, once per turn. Before that, the notes are
//! found in the file as it is again, by their lines' text and context: they
//! follow the code wherever it moved, whoever moved it, and a note whose
//! lines are gone is stale. A file that is gone gives its notes to a read
//! file that has their lines unchanged, as after a rename. `/notes` lists
//! them by file, stale ones first.
//!
//! The files are read off the main thread, under a hold on the read's
//! outcome, so a workspace on another machine never stalls a frame.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{FilesRead, ToolErrorKind, ToolOutput, content_hash};
use rig_tools::Read;
use rig_tools::workspace::{Stat, Workspace};
use schemars::JsonSchema;
use serde::Deserialize;

mod book;

use book::{Book, Note};

/// Who decides what this plugin decides.
const NAME: &str = "file-annotations";

/// The Review system that adds the notes to each `read`'s result, in
/// [`ReviewSystems::Edit`], for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AnnotationSystems;

/// The `annotate` tool, `/note`, `/notes`, and the notes in every read.
#[derive(Default)]
pub struct FileAnnotationsPlugin;

impl Plugin for FileAnnotationsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<RigHome>()
            .init_resource::<Annotations>()
            .register_required_components::<TurnOf, StaleShown>()
            .configure_sets(AgentPass, AnnotationSystems.in_set(ReviewSystems::Edit))
            .add_systems(Startup, load)
            .add_systems(AgentPass, annotate_reads.in_set(AnnotationSystems))
            .add_observer(fetched_for_read)
            .add_observer(fetched_for_change)
            .add_observer(fetched_for_listing)
            .add_open_tool(
                "annotate",
                "Attach a note to lines of a file that every later read of them shows, \
                 without changing the file; or update or delete a note by its id. Use it to \
                 note why code that looks wrong is intentional. A `read` shows notes after \
                 their lines as `┆ note n3 (date, who): text`; heed them. Update or delete a \
                 note shown as stale, whose lines are gone.",
                ToolOptions {
                    footprint: Some(Touches::Reads(Scope::Anything)),
                },
                annotate,
            )
            .add_command(
                "note",
                "Note lines: /note path:12-14 text, or /note rm n3",
                note,
            )
            .add_command(
                "notes",
                "List the project's notes by file, stale ones first",
                notes,
            );
    }
}

/// The project's notes and where they are saved.
#[derive(Resource, Default, Debug)]
struct Annotations {
    /// The project's root, which note paths are relative to.
    root: PathBuf,
    /// Where they are saved; `None` until loaded.
    file: Option<PathBuf>,
    book: Book,
}

impl Annotations {
    /// Saves them, or says why not.
    fn save(&self) -> Result<(), String> {
        let Some(file) = &self.file else {
            return Err("the notes were never loaded".to_owned());
        };
        let json = serde_json::to_vec_pretty(&self.book).map_err(|error| error.to_string())?;
        if let Some(dir) = file.parent() {
            std::fs::create_dir_all(dir).map_err(|error| error.to_string())?;
        }
        std::fs::write(file, json).map_err(|error| format!("{}: {error}", file.display()))
    }
}

/// On a turn: the stale notes its reads showed already.
#[derive(Component, Default)]
struct StaleShown(HashSet<u32>);

/// Loads the project's notes from `RIG_HOME/annotations/`, named after a
/// hash of the project's root.
fn load(
    home: Res<RigHome>,
    root: Res<ProjectRoot>,
    mut annotations: ResMut<Annotations>,
    mut notices: MessageWriter<Notice>,
) {
    let hash = content_hash(root.0.to_string_lossy().as_bytes()).0;
    let name: String = hash.chars().take(16).collect();
    let file = home
        .0
        .root()
        .join("annotations")
        .join(format!("{name}.json"));
    let book = match std::fs::read(&file) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|error| {
            notices.write(Notice::warn(None, format!("{}: {error}", file.display())));
            Book::default()
        }),
        Err(_) => Book::default(),
    };
    *annotations = Annotations {
        root: root.0.clone(),
        file: Some(file),
        book: Book {
            root: root.0.to_string_lossy().into_owned(),
            ..book
        },
    };
}

/// What a read off the main thread found: the text of each file it read
/// (none for one it could not read), the path it resolved, and which of
/// the files it checked are gone.
#[derive(Component, Clone, Default)]
struct Fetched {
    resolved: Option<PathBuf>,
    texts: HashMap<PathBuf, String>,
    gone: HashSet<PathBuf>,
}

/// What to read off the main thread.
#[derive(Default)]
struct Fetch {
    /// A path as named, to resolve from a directory and read.
    resolve: Option<(PathBuf, String)>,
    /// Files to read.
    read: Vec<PathBuf>,
    /// Files to check are still there.
    check: Vec<PathBuf>,
}

impl Fetch {
    /// Work on the IO pool that ends as [`Fetched`] on its entity.
    fn running(self, workspace: Arc<dyn Workspace>, wake: &Wake) -> Running {
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        let work = blocking(move || {
            let mut read = self.read;
            let resolved = self
                .resolve
                .map(|(base, path)| workspace.resolve(Some(&base), &path));
            read.extend(resolved.clone());
            let texts = read
                .into_iter()
                .filter_map(|path| Some((path.clone(), workspace.read_text(&path).ok()?)))
                .collect();
            let gone = self
                .check
                .into_iter()
                .filter(|path| matches!(workspace.stat(path), Ok(Stat::Missing)))
                .collect();
            Ok(Fetched {
                resolved,
                texts,
                gone,
            })
        });
        Running::spawn_into::<Fetched, _>(pool, wake, async move { work.await.unwrap_or_default() })
    }
}

/// On the hold that keeps a `read`'s outcome while its file is read again.
#[derive(Component)]
struct ReadHold;

/// On a `read` whose result has its notes, or needs none.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Annotated;

/// Adds the notes to each successful `read` of a file of a project with
/// notes: first holds its outcome while the file is read again off the
/// main thread, then, if the file is as the tool read it, finds its notes
/// in it again and puts them in the result.
fn annotate_reads(
    mut calls: Query<
        (
            Entity,
            &ToolRequest,
            &mut Outcome,
            Option<&Published>,
            &CallOf,
            Option<&Fetched>,
        ),
        (ReadyToSettle, ToolCalls, Without<Annotated>),
    >,
    mut turns: Query<&mut StaleShown>,
    mut annotations: ResMut<Annotations>,
    (workspace, wake): (Res<ProjectWorkspace>, Res<Wake>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for (call, request, mut outcome, published, &CallOf(turn), fetched) in &mut calls {
        commands.entity(call).insert(Annotated);
        let result = outcome.tool_result();
        if request.call.function.name.as_str() != Read::NAME
            || !result.is_success()
            || annotations.book.notes.is_empty()
        {
            continue;
        }
        let read = published.and_then(|published| published.0.get::<FilesRead>().ok().flatten());
        let Some((file, shown)) = read
            .and_then(|read| read.0.into_iter().next())
            .and_then(|file| file.lines.map(|lines| (file, lines)))
        else {
            continue;
        };
        let path = PathBuf::from(&file.path);
        let Some(fetched) = fetched else {
            let check = annotations.book.other_files(&annotations.root, &path);
            let fetch = Fetch {
                read: vec![path],
                check,
                ..Fetch::default()
            };
            // Not done yet: seen again once the hold lets go of it.
            commands.entity(call).remove::<Annotated>();
            commands.spawn((
                Name::new("finding the file's notes"),
                ReadHold,
                HoldOf(call),
                fetch.running(workspace.0.clone(), &wake),
            ));
            continue;
        };
        commands.entity(call).remove::<Fetched>();
        // Changed again since the tool read it: its lines are not these.
        let Some(text) = fetched.texts.get(&path) else {
            continue;
        };
        if content_hash(text.as_bytes()) != file.hash {
            continue;
        }
        let Annotations { root, book, .. } = &mut *annotations;
        if book.follow(root, &path, text, &fetched.gone)
            && let Err(why) = annotations.save()
        {
            notices.write(Notice::warn(None, format!("Notes not saved: {why}")));
        }
        let Some(output) = result.output().as_text() else {
            continue;
        };
        let mut shown_stale = turns.get_mut(turn).ok();
        let mut none = HashSet::new();
        let already = shown_stale.as_mut().map_or(&mut none, |shown| &mut shown.0);
        let key = Book::key(&annotations.root, &path);
        let annotated =
            annotations
                .book
                .annotate(&key, output, (shown.first, shown.count), already);
        if annotated != output {
            *outcome = Outcome::from(result.clone().with_output(ToolOutput::text(annotated)));
        }
    }
}

/// Lets go of the `read` whose file was read again, with what was found.
fn fetched_for_read(
    fetched: On<Insert<Fetched>>,
    holds: Query<(&HoldOf, &Fetched), With<ReadHold>>,
    mut commands: Commands,
) {
    if let Ok((&HoldOf(call), found)) = holds.get(fetched.entity) {
        commands.entity(call).try_insert(found.clone());
        commands.entity(fetched.entity).despawn();
    }
}

/// A change to the notes.
#[derive(Clone, Debug)]
enum Change {
    /// A new note on lines `start..=end` of the file named `path`, from
    /// `base`.
    Add {
        base: PathBuf,
        path: String,
        lines: (usize, usize),
        text: String,
    },
    /// A note's text, its lines, or both, changed.
    Update {
        id: u32,
        text: Option<String>,
        place: Option<(PathBuf, String, (usize, usize))>,
    },
    /// A note deleted.
    Delete { id: u32 },
}

/// Who asked for a change, and so gets its answer.
#[derive(Clone, Copy, Debug)]
enum Asker {
    /// An `annotate` call.
    Call(Entity),
    /// The user, through `/note`, for this agent.
    User(Entity),
}

impl Asker {
    /// Who the note says wrote it.
    fn by(self) -> &'static str {
        match self {
            Self::Call(_) => "agent",
            Self::User(_) => "user",
        }
    }
}

/// A change waiting for its file to be read.
#[derive(Component)]
struct Pending(Change, Asker);

/// The arguments of `annotate`.
#[derive(Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
struct AnnotateArgs {
    /// `add` (the default), `update` or `delete`.
    #[serde(default)]
    action: Action,
    /// The note to update or delete, such as `n3`.
    id: Option<String>,
    /// The file, to add a note (or move one with `update`).
    path: Option<String>,
    /// The lines, `start-end` or one line, counted from 1.
    lines: Option<String>,
    /// What the note says.
    note: Option<String>,
}

/// What `annotate` does.
#[derive(Deserialize, JsonSchema, Default, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum Action {
    /// A new note.
    #[default]
    Add,
    /// A note changed.
    Update,
    /// A note deleted.
    Delete,
}

/// `12-14`, `12–14` or `12` as the lines it names.
fn parse_lines(lines: &str) -> Option<(usize, usize)> {
    let (start, end) = lines.split_once(['-', '–']).unwrap_or((lines, lines));
    Some((start.trim().parse().ok()?, end.trim().parse().ok()?))
}

/// `n3` or `3` as the note's number.
fn parse_id(id: &str) -> Option<u32> {
    id.trim().trim_start_matches('n').parse().ok()
}

impl AnnotateArgs {
    /// The change these arguments ask for, from `base`, or why they ask
    /// for none.
    fn change(self, base: PathBuf) -> Result<Change, String> {
        let id = || {
            self.id
                .as_deref()
                .and_then(parse_id)
                .ok_or("give the note's id, such as `n3`")
        };
        let lines = self.lines.as_deref().map(|lines| {
            parse_lines(lines).ok_or(format!("`{lines}` is no lines, such as `12-14`"))
        });
        match self.action {
            Action::Delete => Ok(Change::Delete { id: id()? }),
            Action::Add => {
                let (Some(path), Some(lines), Some(text)) =
                    (self.path.clone(), lines, self.note.clone())
                else {
                    return Err("give `path`, `lines` and `note`".to_owned());
                };
                Ok(Change::Add {
                    base,
                    path,
                    lines: lines?,
                    text,
                })
            }
            Action::Update => {
                let place = match lines.transpose()? {
                    Some(lines) => Some((base, self.path.clone().unwrap_or_default(), lines)),
                    None => None,
                };
                Ok(Change::Update {
                    id: id()?,
                    text: self.note.clone(),
                    place,
                })
            }
        }
    }
}

/// An `annotate` call: its change, made at once, or once its file is read.
fn annotate(
    called: On<ToolCalled<AnnotateArgs>>,
    mut calls: Query<&mut Decision, ToolCalls>,
    dirs: Query<&WorkDir>,
    root: Res<ProjectRoot>,
    mut annotations: ResMut<Annotations>,
    (workspace, wake): (Res<ProjectWorkspace>, Res<Wake>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let base = dirs.get(called.agent).map_or(&root.0, |dir| &dir.0).clone();
    let args = called.args.clone();
    let asker = Asker::Call(called.call);
    match args.change(base) {
        Ok(change) => start(
            change,
            asker,
            &mut annotations,
            &workspace,
            &wake,
            &mut commands,
        )
        .map_or((), |answer| {
            answer_to(asker, answer, &mut calls, &mut notices)
        }),
        Err(why) => answer_to(asker, Err(why), &mut calls, &mut notices),
    }
}

/// Makes `change` now, giving its answer, or reads its file first, giving
/// none yet.
fn start(
    change: Change,
    asker: Asker,
    annotations: &mut Annotations,
    workspace: &ProjectWorkspace,
    wake: &Wake,
    commands: &mut Commands,
) -> Option<Result<String, String>> {
    let resolve = match &change {
        Change::Add { base, path, .. } => Some((base.clone(), path.clone())),
        Change::Update {
            id,
            place: Some((base, path, _)),
            ..
        } => {
            let own = annotations.book.notes.iter().find(|note| note.id == *id);
            let path = match (path.is_empty(), own) {
                (true, Some(note)) => annotations
                    .root
                    .join(&note.path)
                    .to_string_lossy()
                    .into_owned(),
                _ => path.clone(),
            };
            Some((base.clone(), path))
        }
        Change::Update { .. } | Change::Delete { .. } => None,
    };
    let Some(resolve) = resolve else {
        return Some(apply(annotations, &change, asker, &Fetched::default()));
    };
    let fetch = Fetch {
        resolve: Some(resolve),
        ..Fetch::default()
    };
    commands.spawn((
        Name::new("reading the noted file"),
        Pending(change, asker),
        fetch.running(workspace.0.clone(), wake),
    ));
    None
}

/// Makes `change` with the file `fetched` read, saves the notes, and says
/// what it did, or why it did nothing.
fn apply(
    annotations: &mut Annotations,
    change: &Change,
    asker: Asker,
    fetched: &Fetched,
) -> Result<String, String> {
    let anchored = |annotations: &Annotations, (start, end): (usize, usize)| {
        let path = fetched.resolved.clone().unwrap_or_default();
        let text = fetched
            .texts
            .get(&path)
            .ok_or(format!("{} cannot be read", path.display()))?;
        let lines: Vec<&str> = text.lines().collect();
        if start == 0 || start > end || end > lines.len() {
            return Err(format!(
                "lines {start}–{end} are not in {}, which has {} lines",
                path.display(),
                lines.len()
            ));
        }
        let mut note = Note {
            id: 0,
            path: Book::key(&annotations.root, &path),
            start,
            end,
            anchor: Vec::new(),
            before: Vec::new(),
            after: Vec::new(),
            text: String::new(),
            date: chrono::Local::now().format("%Y-%m-%d").to_string(),
            by: asker.by().to_owned(),
            stale: false,
        };
        note.anchor_at(&lines, start - 1, end);
        let shown: String = lines
            .get(start - 1..end)
            .unwrap_or_default()
            .iter()
            .zip(start..)
            .map(|(line, number)| format!("{number:>6}\t{line}\n"))
            .collect();
        Ok((note, shown))
    };
    let (note, shown, verb) = match change {
        Change::Add { lines, text, .. } => {
            let (mut note, shown) = anchored(annotations, *lines)?;
            annotations.book.last += 1;
            note.id = annotations.book.last;
            note.text.clone_from(text);
            annotations.book.notes.push(note.clone());
            (note, shown, "Added")
        }
        Change::Update { id, text, place } => {
            let moved = place
                .as_ref()
                .map(|(_, _, lines)| anchored(annotations, *lines))
                .transpose()?;
            let Some(note) = annotations
                .book
                .notes
                .iter_mut()
                .find(|note| note.id == *id)
            else {
                return Err(format!("there is no note n{id}"));
            };
            if let Some((place, _)) = &moved {
                note.path.clone_from(&place.path);
                note.start = place.start;
                note.end = place.end;
                note.anchor.clone_from(&place.anchor);
                note.before.clone_from(&place.before);
                note.after.clone_from(&place.after);
                note.stale = false;
            }
            if let Some(text) = text {
                note.text.clone_from(text);
            }
            let shown = moved.map(|(_, shown)| shown).unwrap_or_default();
            (note.clone(), shown, "Updated")
        }
        Change::Delete { id } => {
            let Some(at) = annotations
                .book
                .notes
                .iter()
                .position(|note| note.id == *id)
            else {
                return Err(format!("there is no note n{id}"));
            };
            let note = annotations.book.notes.remove(at);
            annotations.save()?;
            return Ok(format!(
                "Deleted note n{id} on {}: {}",
                note.path, note.text
            ));
        }
    };
    annotations.save()?;
    let line = if note.stale {
        note.stale_line()
    } else {
        note.line()
    };
    Ok(format!(
        "{verb} note n{} on {} lines {}–{}:\n{shown}{line}",
        note.id, note.path, note.start, note.end
    ))
}

/// Gives the asker the answer: the call's result, or a notice to the user.
fn answer_to(
    asker: Asker,
    answer: Result<String, String>,
    calls: &mut Query<&mut Decision, ToolCalls>,
    notices: &mut MessageWriter<Notice>,
) {
    match (asker, answer) {
        (Asker::Call(call), answer) => {
            let result = match answer {
                Ok(text) => ToolResult::success(ToolOutput::text(text)),
                Err(why) => {
                    ToolResult::failed(ToolExecutionError::new(ToolErrorKind::InvalidArgs, why))
                }
            };
            if let Ok(mut decision) = calls.get_mut(call) {
                decision.decide(Decider::plugin(NAME), Outcome::from(result));
            }
        }
        (Asker::User(agent), Ok(text)) => {
            notices.write(Notice::info(agent, text));
        }
        (Asker::User(agent), Err(why)) => {
            notices.write(Notice::warn(agent, format!("No note changed: {why}")));
        }
    }
}

/// Makes a change once its file was read.
fn fetched_for_change(
    fetched: On<Insert<Fetched>>,
    pending: Query<(&Pending, &Fetched)>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut annotations: ResMut<Annotations>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((Pending(change, asker), found)) = pending.get(fetched.entity) else {
        return;
    };
    commands.entity(fetched.entity).despawn();
    // The call was interrupted meanwhile: it changes nothing.
    if let Asker::Call(call) = asker
        && calls.get(*call).is_err()
    {
        return;
    }
    let answer = apply(&mut annotations, change, *asker, found);
    answer_to(*asker, answer, &mut calls, &mut notices);
}

/// `/note path:12-14 text` and `/note rm n3`.
fn note(
    In(args): In<CommandArgs>,
    dirs: Query<&WorkDir>,
    root: Res<ProjectRoot>,
    mut annotations: ResMut<Annotations>,
    (workspace, wake): (Res<ProjectWorkspace>, Res<Wake>),
    mut calls: Query<&mut Decision, ToolCalls>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let line = args.args.trim();
    let change = if let Some(id) = line.strip_prefix("rm ") {
        parse_id(id).map(|id| Change::Delete { id })
    } else {
        line.split_once(char::is_whitespace)
            .and_then(|(place, text)| {
                let (path, lines) = place.rsplit_once(':')?;
                let base = dirs.get(args.agent).map_or(&root.0, |dir| &dir.0).clone();
                let (path, text) = (path.to_owned(), text.trim().to_owned());
                Some(Change::Add {
                    base,
                    path,
                    lines: parse_lines(lines)?,
                    text,
                })
            })
    };
    let Some(change) = change else {
        notices.write(Notice::error(
            args.agent,
            "Use /note path:12-14 text, or /note rm n3",
        ));
        return;
    };
    let asker = Asker::User(args.agent);
    if let Some(answer) = start(
        change,
        asker,
        &mut annotations,
        &workspace,
        &wake,
        &mut commands,
    ) {
        answer_to(asker, answer, &mut calls, &mut notices);
    }
}

/// On the work that reads every noted file for `/notes`, for this agent.
#[derive(Component)]
struct Listing(Entity);

/// `/notes`: reads every noted file again, then lists the notes.
fn notes(
    In(args): In<CommandArgs>,
    annotations: Res<Annotations>,
    (workspace, wake): (Res<ProjectWorkspace>, Res<Wake>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if annotations.book.notes.is_empty() {
        notices.write(Notice::info(args.agent, "No notes in this project."));
        return;
    }
    let root = &annotations.root;
    let files: HashSet<PathBuf> = annotations
        .book
        .notes
        .iter()
        .map(|note| root.join(&note.path))
        .collect();
    let fetch = Fetch {
        read: files.into_iter().collect(),
        ..Fetch::default()
    };
    commands.spawn((
        Name::new("reading the noted files"),
        Listing(args.agent),
        fetch.running(workspace.0.clone(), &wake),
    ));
}

/// Finds every note again in the files read for `/notes`, then lists them.
fn fetched_for_listing(
    fetched: On<Insert<Fetched>>,
    listings: Query<(&Listing, &Fetched)>,
    mut annotations: ResMut<Annotations>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((&Listing(agent), found)) = listings.get(fetched.entity) else {
        return;
    };
    commands.entity(fetched.entity).despawn();
    let Annotations { root, book, .. } = &mut *annotations;
    let mut changed = false;
    for note in &mut book.notes {
        let lines: Option<Vec<&str>> = found
            .texts
            .get(&root.join(&note.path))
            .map(|text| text.lines().collect());
        changed |= match lines {
            Some(lines) => note.follow(&lines),
            None => !std::mem::replace(&mut note.stale, true),
        };
    }
    if changed && let Err(why) = annotations.save() {
        notices.write(Notice::warn(agent, format!("Notes not saved: {why}")));
    }
    notices.write(Notice::info(
        agent,
        annotations.book.listing().trim_end().to_owned(),
    ));
}

#[cfg(test)]
mod tests;

//! Plugin corpus: 25. Semantic code search (RAG) (spec `25-code-search-index.md`).
//!
//! [`CodeSearchPlugin`] keeps an index of the project's files (those git
//! tracks or would track) in `<project>/.rig/index/`: each file is cut into
//! chunks at top-level items and paragraphs, and each chunk is embedded by
//! the [`Embedder`] resource. The index is built in the background, a batch
//! of files per task, with its progress in the status line; the
//! `code_search` tool answers from what is built so far. A file any tool
//! call says it changed (rig-core's `FilesChanged`) is indexed again at
//! once, and a search first indexes again the files whose time or size
//! changed. `/index rebuild` and `/index status` manage it. With
//! `context_chunks` set in the `[code_search]` settings, the best chunks for
//! the user's newest message go with that turn's requests, not into the
//! conversation.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use bevy_log::warn;
use bevy_tasks::{AsyncComputeTaskPool, IoTaskPool, TaskPool};
use rig_core::DynModel;
use rig_core::operation::Embedding;
use rig_core::tool::FilesChanged;
use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Where the index is kept, under the project.
const INDEX_FILE: &str = ".rig/index/index.json";
/// The most lines in a chunk.
const MAX_LINES: usize = 60;
/// Larger files are left out.
const MAX_FILE_BYTES: u64 = 512 * 1024;
/// Files per background task.
const BATCH: usize = 16;
/// The hashing embedder's dimensions.
const DIMS: usize = 1024;
/// Hits a search returns unless it asks for another number, and the most.
const DEFAULT_K: usize = 5;
const MAX_K: usize = 20;
/// Lines of a chunk shown in a hit.
const SNIPPET_LINES: usize = 12;

/// The code index, the `code_search` tool and the `/index` command.
#[derive(Default)]
pub struct CodeSearchPlugin;

impl Plugin for CodeSearchPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .init_resource::<Embedder>()
            .init_resource::<CodeIndex>()
            .add_setting::<CodeSearchSettings>("code_search")
            .add_open_tool(
                "code_search",
                "Search the project's code by meaning. Returns the best matching chunks: \
                 path, line range, score and snippet. Use it to find code by what it does \
                 (\"where do we retry HTTP requests?\"); use `search` for an exact name or \
                 pattern.",
                ToolOptions {
                    footprint: Some(Touches::Reads(Scope::Anything)),
                },
                search,
            )
            .add_command(
                "index",
                "/index rebuild | status: the code search index",
                index,
            )
            .add_systems(Startup, start)
            .configure_sets(
                AgentPass,
                (
                    // A line the harness took over is no message to look up.
                    CodeSearchSystems::LookUp
                        .in_set(Stage::Gate)
                        .after(AskSystems),
                    // Compaction measures the request with the chunks in it.
                    CodeSearchSystems::AddContext.in_set(PrepareSystems::Edit),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    look_up.in_set(CodeSearchSystems::LookUp),
                    add_context.in_set(CodeSearchSystems::AddContext),
                ),
            )
            .add_observer(listed)
            .add_observer(indexed)
            .add_observer(searched)
            .add_observer(reindex_changed)
            .add_observer(context_found)
            .add_observer(delivered);
        #[cfg(feature = "tui")]
        status::add(app);
    }
}

/// The plugin's stage systems, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodeSearchSystems {
    /// In [`Stage::Gate`]: holds each new message of the user's while the
    /// index is searched for it, after [`AskSystems`].
    LookUp,
    /// In [`PrepareSystems::Edit`]: adds the chunks found for a user's
    /// message to its turn's requests.
    AddContext,
}

/// The `[code_search]` settings.
#[derive(Resource, Deserialize, Clone, Default)]
#[serde(default, deny_unknown_fields)]
pub struct CodeSearchSettings {
    /// How many chunks go with the requests of each turn on a new user
    /// message; none when zero.
    pub context_chunks: usize,
}

/// What turns text into vectors, inserted before the plugin is added to
/// change it.
#[derive(Resource, Clone, Default)]
pub enum Embedder {
    /// Words (snake and camel case split) hashed into buckets: no model,
    /// deterministic.
    #[default]
    Hashing,
    /// Any rig-core embedding model, such as a local Fastembed one
    /// (`rig_fastembed::Fastembed::embedding(..)?.erase()`).
    Model(DynModel<Embedding>),
}

impl Embedder {
    /// What the saved vectors were made with, so another embedder's are
    /// not mixed in.
    fn id(&self) -> String {
        match self {
            Self::Hashing => format!("hashing-{DIMS}"),
            Self::Model(model) => format!("{}:{}", model.name(), model.id().unwrap_or_default()),
        }
    }

    /// The unit vectors of `texts`, in order.
    async fn embed(self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        match self {
            Self::Hashing => Ok(texts.iter().map(|text| hashed(text)).collect()),
            Self::Model(model) => {
                let count = texts.len();
                let response = model.call(texts).await.map_err(|error| error.to_string())?;
                let vectors: Vec<Vec<f32>> = response
                    .embeddings
                    .into_iter()
                    .map(|embedding| unit(embedding.vec.into_iter().map(|x| x as f32).collect()))
                    .collect();
                if vectors.len() == count {
                    Ok(vectors)
                } else {
                    Err(format!("{} vectors for {count} texts", vectors.len()))
                }
            }
        }
    }

    /// The unit vector of `query`, or why there is none.
    async fn embed_one(self, query: String) -> Result<Vec<f32>, String> {
        let vectors = self.embed(vec![query]).await?;
        vectors
            .into_iter()
            .next()
            .ok_or_else(|| "no vector for the query".to_owned())
    }
}

/// A file's modification time and size, by which a change is noticed.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Stamp {
    modified_ns: u64,
    len: u64,
}

/// One indexed file.
#[derive(Serialize, Deserialize, Clone)]
struct FileEntry {
    stamp: Stamp,
    chunks: Vec<Chunk>,
}

/// Lines `start..=end` of a file and their vector.
#[derive(Serialize, Deserialize, Clone)]
struct Chunk {
    start: usize,
    end: usize,
    text: String,
    vector: Vec<f32>,
}

type Files = HashMap<String, Arc<FileEntry>>;

/// A file indexed again: its entry, or `None` when it is gone or not text.
type Update = (String, Option<FileEntry>);

/// The index of the project's files, by path relative to the project.
#[derive(Resource, Default)]
pub struct CodeIndex {
    /// The project, canonical, as the build found it.
    root: PathBuf,
    files: Files,
    /// Bumped by a rebuild, so work started before it is dropped.
    generation: u64,
    building: bool,
    queue: VecDeque<String>,
    total: usize,
    done: usize,
}

impl CodeIndex {
    /// Whether the background build finished.
    pub fn is_built(&self) -> bool {
        !self.building
    }

    /// How much of the build is done, in percent.
    pub fn percent(&self) -> usize {
        match self.building {
            true => self.done * 100 / self.total.max(1),
            false => 100,
        }
    }

    /// The files and chunks indexed.
    pub fn size(&self) -> (usize, usize) {
        let chunks = self.files.values().map(|entry| entry.chunks.len()).sum();
        (self.files.len(), chunks)
    }

    /// Whether a chunk of `path` contains `text`.
    pub fn contains(&self, path: &str, text: &str) -> bool {
        self.files
            .get(path)
            .is_some_and(|entry| entry.chunks.iter().any(|chunk| chunk.text.contains(text)))
    }

    /// Takes in `updates`, an entry only when newer than the one it
    /// replaces; whether anything changed.
    fn merge(&mut self, updates: Vec<Update>) -> bool {
        let changed = !updates.is_empty();
        for (path, entry) in updates {
            match entry {
                Some(entry) => {
                    if self
                        .files
                        .get(&path)
                        .is_none_or(|old| old.stamp <= entry.stamp)
                    {
                        self.files.insert(path, Arc::new(entry));
                    }
                }
                None => {
                    self.files.remove(&path);
                }
            }
        }
        changed
    }
}

/// The background tasks' pool.
fn pool() -> &'static AsyncComputeTaskPool {
    AsyncComputeTaskPool::get_or_init(TaskPool::default)
}

/// The project's files and the saved index, listed off the loop.
#[derive(Component)]
struct Listed {
    generation: u64,
    saved: HashMap<String, FileEntry>,
    files: Vec<(String, Stamp)>,
}

/// Files indexed off the loop: a batch of the build, or changed files.
#[derive(Component)]
struct Indexed {
    generation: u64,
    batch: Option<usize>,
    updates: Vec<Update>,
}

/// A `code_search` call's answer, and the files it indexed again, on the
/// call.
#[derive(Component)]
struct Searched {
    generation: u64,
    updates: Vec<Update>,
    result: ToolResult,
}

/// The chunks found for a user's message, on the hold that kept it back
/// meanwhile.
#[derive(Component)]
struct ContextFound(TurnContext);

/// On a user's message, then on the turn it went into: the chunks that go
/// with the turn's requests, for the message `key`.
#[derive(Component, Clone)]
struct TurnContext {
    key: String,
    text: Option<String>,
}

fn start(
    mut index: ResMut<CodeIndex>,
    project: Res<ProjectRoot>,
    embedder: Res<Embedder>,
    processes: Res<Processes>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    index.root = std::fs::canonicalize(&project.0).unwrap_or_else(|_| project.0.clone());
    begin(
        &mut index,
        &embedder,
        &processes,
        &wake,
        &mut commands,
        true,
    );
}

/// Starts a build: lists the files and, with `saved`, reads the saved
/// index, whose entries still current are kept.
fn begin(
    index: &mut CodeIndex,
    embedder: &Embedder,
    processes: &Processes,
    wake: &Wake,
    commands: &mut Commands,
    saved: bool,
) {
    index.generation += 1;
    index.building = true;
    index.queue.clear();
    (index.total, index.done) = (0, 0);
    if !saved {
        index.files.clear();
    }
    let (root, embedder, generation) = (index.root.clone(), embedder.id(), index.generation);
    let git = processes.launcher().command("git");
    let work = async move {
        let listing = blocking(move || {
            let saved = match saved {
                true => load(&root, &embedder),
                false => HashMap::new(),
            };
            Ok((saved, list(git, &root)))
        });
        let (saved, files) = listing.await.unwrap_or_default();
        Listed {
            generation,
            saved,
            files,
        }
    };
    commands.spawn((
        Name::new("code index listing"),
        Running::spawn_into::<Listed, _>(pool(), wake, work),
    ));
}

/// Queues every listed file whose saved entry is missing or out of date.
fn listed(
    add: On<Add<Listed>>,
    mut done: Query<&mut Listed>,
    mut index: ResMut<CodeIndex>,
    embedder: Res<Embedder>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok(mut listed) = done.get_mut(add.entity) else {
        return;
    };
    commands.entity(add.entity).despawn();
    if listed.generation != index.generation {
        return;
    }
    let mut saved = std::mem::take(&mut listed.saved);
    let mut queue = VecDeque::new();
    for (path, stamp) in std::mem::take(&mut listed.files) {
        match saved.remove(&path) {
            Some(entry) if entry.stamp == stamp => {
                index.files.entry(path).or_insert_with(|| Arc::new(entry));
            }
            _ => queue.push_back(path),
        }
    }
    index.total = queue.len();
    index.queue = queue;
    next_batch(&mut index, &embedder, &wake, &mut commands);
}

/// Indexes the next batch of the build, or ends it and saves the index.
fn next_batch(index: &mut CodeIndex, embedder: &Embedder, wake: &Wake, commands: &mut Commands) {
    let take = index.queue.len().min(BATCH);
    let paths: Vec<String> = index.queue.drain(..take).collect();
    if paths.is_empty() {
        index.building = false;
        save(index, embedder);
        return;
    }
    spawn_indexing(index, embedder, wake, commands, paths, true);
}

/// Indexes `paths` off the loop; `batch` when they are the build's.
fn spawn_indexing(
    index: &CodeIndex,
    embedder: &Embedder,
    wake: &Wake,
    commands: &mut Commands,
    paths: Vec<String>,
    batch: bool,
) {
    let (batch, generation) = (batch.then_some(paths.len()), index.generation);
    let work = index_files(index.root.clone(), embedder.clone(), paths);
    let work = async move {
        Indexed {
            generation,
            batch,
            updates: work.await,
        }
    };
    commands.spawn((
        Name::new("code index task"),
        Running::spawn_into::<Indexed, _>(pool(), wake, work),
    ));
}

/// Takes in indexed files; a batch of the build starts the next.
fn indexed(
    add: On<Add<Indexed>>,
    mut done: Query<&mut Indexed>,
    mut index: ResMut<CodeIndex>,
    embedder: Res<Embedder>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok(mut indexed) = done.get_mut(add.entity) else {
        return;
    };
    commands.entity(add.entity).despawn();
    if indexed.generation != index.generation {
        return;
    }
    let changed = index.merge(std::mem::take(&mut indexed.updates));
    match indexed.batch {
        Some(count) => {
            index.done += count;
            next_batch(&mut index, &embedder, &wake, &mut commands);
        }
        None if changed && !index.building => save(&index, &embedder),
        None => {}
    }
}

/// Indexes again the project files a tool call says it changed, whatever
/// the tool.
fn reindex_changed(
    add: On<Add<Published>>,
    published: Query<&Published>,
    index: Res<CodeIndex>,
    embedder: Res<Embedder>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok(Ok(Some(changed))) = published
        .get(add.entity)
        .map(|published| published.0.get::<FilesChanged>())
    else {
        return;
    };
    let paths: Vec<String> = changed
        .paths()
        .filter_map(|path| relative(&index.root, path))
        .collect();
    if !paths.is_empty() {
        spawn_indexing(&index, &embedder, &wake, &mut commands, paths, false);
    }
}

/// `path`, which is relative to the project unless absolute, relative to
/// the project; `None` outside it and in `.rig/`.
fn relative(root: &Path, path: &str) -> Option<String> {
    let path = root.join(path);
    let path = std::fs::canonicalize(&path).unwrap_or(path);
    let path = path.strip_prefix(root).ok()?.to_str()?.to_owned();
    (!path.starts_with(".rig/")).then_some(path)
}

/// The arguments of `code_search`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SearchArgs {
    /// What the code does, in words.
    query: String,
    /// How many chunks to return (default 5, at most 20).
    #[serde(default)]
    k: Option<usize>,
}

/// Answers a `code_search` call off the loop: indexes again what changed,
/// then ranks the chunks against the query.
fn search(
    called: On<ToolCalled<SearchArgs>>,
    index: Res<CodeIndex>,
    embedder: Res<Embedder>,
    processes: Res<Processes>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let k = called.args.k.unwrap_or(DEFAULT_K).clamp(1, MAX_K);
    let note = match index.building {
        true => format!("index {}% built; results are partial.\n", index.percent()),
        false => String::new(),
    };
    let (root, embedder) = (index.root.clone(), embedder.clone());
    let (files, generation) = (index.files.clone(), index.generation);
    // New files are looked for only once the build, which lists them, is done.
    let git = (!index.building).then(|| processes.launcher().command("git"));
    let query = called.args.query.clone();
    let work = async move {
        let updates = refresh(root, embedder.clone(), &files, git).await;
        let result = match embedder.embed_one(query).await {
            Ok(query) => {
                let hits = rank(current(&files, &updates), &query, k);
                let text = match hits.is_empty() {
                    true => "No code matches.".to_owned(),
                    false => describe(&hits),
                };
                ToolResult::success(format!("{note}{text}").into())
            }
            Err(why) => ToolResult::failed(ToolExecutionError::other(format!(
                "embedding failed: {why}"
            ))),
        };
        Searched {
            generation,
            updates,
            result,
        }
    };
    commands
        .entity(called.call)
        .insert(Running::spawn_into::<Searched, _>(pool(), &wake, work));
}

/// Takes in what a search indexed again, and answers its call.
fn searched(
    add: On<Add<Searched>>,
    mut done: Query<(&mut Searched, &mut Decision), ToolCalls>,
    mut index: ResMut<CodeIndex>,
    embedder: Res<Embedder>,
    mut commands: Commands,
) {
    let Ok((mut searched, mut decision)) = done.get_mut(add.entity) else {
        return;
    };
    if searched.generation == index.generation
        && index.merge(std::mem::take(&mut searched.updates))
        && !index.building
    {
        save(&index, &embedder);
    }
    decision.decide(
        Decider::plugin("code-search-index"),
        searched.result.clone(),
    );
    commands.entity(add.entity).remove::<Searched>();
}

/// Indexes again the indexed files whose stamp changed, and, with a
/// `git` command to list them, the project's files not indexed yet.
async fn refresh(
    root: PathBuf,
    embedder: Embedder,
    files: &Files,
    git: Option<Command>,
) -> Vec<Update> {
    let stamps: HashMap<String, Stamp> = files
        .iter()
        .map(|(path, entry)| (path.clone(), entry.stamp))
        .collect();
    let base = root.clone();
    let stale = blocking(move || {
        let mut stale: Vec<String> = stamps
            .iter()
            .filter(|(path, stamp)| stamp_of(&base.join(path)).as_ref() != Some(*stamp))
            .map(|(path, _)| path.clone())
            .collect();
        if let Some(git) = git {
            let new = list(git, &base).into_iter().map(|(path, _)| path);
            stale.extend(new.filter(|path| !stamps.contains_key(path)));
        }
        Ok(stale)
    });
    let stale = stale.await.unwrap_or_default();
    index_files(root, embedder, stale).await
}

/// The entries of `files` with `updates` in place of theirs.
fn current<'a>(
    files: &'a Files,
    updates: &'a [Update],
) -> impl Iterator<Item = (&'a str, &'a FileEntry)> {
    let updated: HashSet<&str> = updates.iter().map(|(path, _)| path.as_str()).collect();
    let kept = files
        .iter()
        .filter(move |(path, _)| !updated.contains(path.as_str()))
        .map(|(path, entry)| (path.as_str(), entry.as_ref()));
    let fresh = updates
        .iter()
        .filter_map(|(path, entry)| Some((path.as_str(), entry.as_ref()?)));
    kept.chain(fresh)
}

/// A chunk that matched.
struct Hit<'a> {
    path: &'a str,
    chunk: &'a Chunk,
    score: f32,
}

/// The `k` chunks of `files` closest to `query`, best first.
fn rank<'a>(
    files: impl Iterator<Item = (&'a str, &'a FileEntry)>,
    query: &[f32],
    k: usize,
) -> Vec<Hit<'a>> {
    let mut hits: Vec<Hit<'a>> = files
        .flat_map(|(path, entry)| {
            entry.chunks.iter().map(move |chunk| Hit {
                path,
                chunk,
                score: dot(query, &chunk.vector),
            })
        })
        .filter(|hit| hit.score > 0.0)
        .collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.path.cmp(b.path)));
    hits.truncate(k);
    hits
}

/// Each hit as `path:start-end (score)` and its first lines.
fn describe(hits: &[Hit<'_>]) -> String {
    let hits = hits.iter().map(|hit| {
        let snippet: Vec<&str> = hit.chunk.text.lines().take(SNIPPET_LINES).collect();
        format!(
            "{}:{}-{} (score {:.2})\n{}",
            rig_tools::fs::shown(Path::new(hit.path)).display(),
            hit.chunk.start,
            hit.chunk.end,
            hit.score,
            snippet.join("\n")
        )
    });
    hits.collect::<Vec<_>>().join("\n\n")
}

/// Holds each new message the user sends, unless a gate decided it, while
/// the index is searched for it, so the chunks found are ready when its
/// turn's calls are prepared.
fn look_up(
    messages: Query<
        (Entity, &Incoming, &Decision),
        (ReadyToStart, IncomingMessages, Without<TurnContext>),
    >,
    settings: Res<CodeSearchSettings>,
    index: Res<CodeIndex>,
    embedder: Res<Embedder>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let k = settings.context_chunks;
    if k == 0 || index.files.is_empty() {
        return;
    }
    for (message, incoming, decision) in &messages {
        let key = incoming.text().trim();
        if incoming.author.kind != AuthorKind::User
            || decision.is_decided()
            || incoming.command().is_some()
            || key.is_empty()
        {
            continue;
        }
        let (files, embedder, key) = (index.files.clone(), embedder.clone(), key.to_owned());
        let work = async move {
            let query = embedder.embed_one(key.clone()).await.unwrap_or_default();
            let hits = rank(current(&files, &[]), &query, k);
            let text = (!hits.is_empty()).then(|| {
                format!(
                    "<code-context>\nCode that may be relevant:\n\n{}\n</code-context>",
                    describe(&hits)
                )
            });
            ContextFound(TurnContext { key, text })
        };
        commands.spawn((
            Name::new("code context"),
            HoldOf(message),
            Running::spawn_into::<ContextFound, _>(pool(), &wake, work),
        ));
    }
}

/// Puts the chunks found on the message they were found for, and lets it
/// go.
fn context_found(
    add: On<Add<ContextFound>>,
    found: Query<(&ContextFound, &HoldOf)>,
    mut commands: Commands,
) {
    if let Ok((ContextFound(context), &HoldOf(message))) = found.get(add.entity) {
        commands.entity(message).try_insert(context.clone());
    }
    commands.entity(add.entity).despawn();
}

/// Keeps a message's chunks on the turn it went into, for the turn's calls.
fn delivered(
    add: On<Add<DeliveredIn>>,
    messages: Query<(&TurnContext, &DeliveredIn)>,
    mut commands: Commands,
) {
    if let Ok((context, &DeliveredIn(turn))) = messages.get(add.entity) {
        commands.entity(turn).try_insert(context.clone());
    }
}

/// Adds the chunks of a turn's message to each of the turn's model calls.
fn add_context(
    mut calls: Query<(&mut ModelRequest, &CallOf), (Added<Admitted>, With<TurnCall>)>,
    turns: Query<&TurnContext>,
) {
    for (mut request, &CallOf(turn)) in &mut calls {
        if let Ok(context) = turns.get(turn) {
            context.add_to(&mut request);
        }
    }
}

impl TurnContext {
    /// Adds the chunks to the newest of `request`'s user messages that holds
    /// the text they were found for.
    fn add_to(&self, request: &mut ModelRequest) {
        let Some(text) = &self.text else {
            return;
        };
        let typed = |item: &UserContent| matches!(item, UserContent::Text(typed) if typed.text.ends_with(&self.key));
        let Some(history) = &mut request.history else {
            return;
        };
        let found = history.messages().iter().rposition(|message| {
            matches!(&**message, message::Message::User { content } if content.iter().any(typed))
        });
        let entry = found.and_then(|at| Some((at, history.entry(at)?)));
        if let Some((at, mut entry)) = entry
            && let message::Message::User { content } = std::sync::Arc::make_mut(&mut entry.message)
        {
            content.push(UserContent::text(text.clone()));
            history.splice(at..=at, [entry], Author::plugin("code-search-index"));
        }
    }
}

/// `/index rebuild` or `/index status`.
fn index(
    In(args): In<CommandArgs>,
    mut index: ResMut<CodeIndex>,
    embedder: Res<Embedder>,
    processes: Res<Processes>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    match args.args.trim() {
        "rebuild" => {
            begin(
                &mut index,
                &embedder,
                &processes,
                &wake,
                &mut commands,
                false,
            );
            notices.write(Notice::info(args.agent, "Rebuilding the code index."));
        }
        "status" | "" => {
            let (files, chunks) = index.size();
            let mut text = format!(
                "Code index of {}: {files} files, {chunks} chunks, embedder {}.",
                index.root.display(),
                embedder.id()
            );
            if index.building {
                text.push_str(&format!(
                    " Building: {}% ({} of {} files).",
                    index.percent(),
                    index.done,
                    index.total
                ));
            }
            notices.write(Notice::info(args.agent, text));
        }
        other => {
            notices.write(Notice::error(
                args.agent,
                format!("/index takes `rebuild` or `status`, not `{other}`"),
            ));
        }
    }
}

/// Writes the index to the project's `.rig/index/`, off the loop.
fn save(index: &CodeIndex, embedder: &Embedder) {
    #[derive(Serialize)]
    struct SavedRef<'a> {
        embedder: String,
        files: BTreeMap<&'a str, &'a FileEntry>,
    }
    let saved = SavedRef {
        embedder: embedder.id(),
        files: index
            .files
            .iter()
            .map(|(path, entry)| (path.as_str(), entry.as_ref()))
            .collect(),
    };
    let Ok(bytes) = serde_json::to_vec(&saved) else {
        return;
    };
    let path = index.root.join(INDEX_FILE);
    let write = async move {
        let temporary = path.with_extension("json.tmp");
        let written = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(&temporary, bytes))
            .and_then(|()| std::fs::rename(&temporary, &path));
        if let Err(error) = written {
            warn!("could not save the code index: {error}");
        }
    };
    IoTaskPool::get_or_init(TaskPool::default)
        .spawn(write)
        .detach();
}

/// The saved index of `root`, if it was made by the embedder `embedder`.
fn load(root: &Path, embedder: &str) -> HashMap<String, FileEntry> {
    #[derive(Deserialize)]
    struct Saved {
        embedder: String,
        files: HashMap<String, FileEntry>,
    }
    let saved = std::fs::read(root.join(INDEX_FILE))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Saved>(&bytes).ok());
    saved
        .filter(|saved| saved.embedder == embedder)
        .map(|saved| saved.files)
        .unwrap_or_default()
}

/// The files git tracks or would track in `root` (not ignored), with
/// their stamps, listed by `git`, a command made as every child process
/// of the agent is; none outside a git repository.
fn list(mut git: Command, root: &Path) -> Vec<(String, Stamp)> {
    let output = git
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(root)
        .output();
    let Some(output) = output.ok().filter(|output| output.status.success()) else {
        return Vec::new();
    };
    let mut files: Vec<(String, Stamp)> = output
        .stdout
        .split(|&byte| byte == 0)
        .filter_map(|path| std::str::from_utf8(path).ok())
        .filter(|path| !path.is_empty() && !path.starts_with(".rig/"))
        .filter_map(|path| Some((path.to_owned(), stamp_of(&root.join(path))?)))
        .collect();
    files.sort();
    files.dedup_by(|a, b| a.0 == b.0);
    files
}

/// The stamp of the regular file at `path`.
fn stamp_of(path: &Path) -> Option<Stamp> {
    let meta = std::fs::metadata(path)
        .ok()
        .filter(std::fs::Metadata::is_file)?;
    let modified = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
    Some(Stamp {
        modified_ns: u64::try_from(modified.as_nanos()).unwrap_or(u64::MAX),
        len: meta.len(),
    })
}

/// The text of the file at `path` and its stamp; `None` for a file that is
/// gone, too large or binary.
fn read_source(path: &Path) -> Option<(Stamp, String)> {
    let stamp = stamp_of(path).filter(|stamp| stamp.len <= MAX_FILE_BYTES)?;
    let bytes = std::fs::read(path).ok()?;
    if bytes.iter().take(8000).any(|&byte| byte == 0) {
        return None;
    }
    String::from_utf8(bytes).ok().map(|text| (stamp, text))
}

/// Reads, chunks and embeds `paths` of `root`.
async fn index_files(root: PathBuf, embedder: Embedder, paths: Vec<String>) -> Vec<Update> {
    if paths.is_empty() {
        return Vec::new();
    }
    let read = blocking(move || {
        Ok(paths
            .into_iter()
            .map(|path| {
                let source = read_source(&root.join(&path));
                (path, source.map(|(stamp, text)| (stamp, chunk(&text))))
            })
            .collect::<Vec<_>>())
    });
    let read = read.await.unwrap_or_default();
    let texts: Vec<String> = read
        .iter()
        .flat_map(|(path, source)| {
            let chunks = source.iter().flat_map(|(_, chunks)| chunks);
            chunks.map(move |(_, _, text)| format!("{path}\n{text}"))
        })
        .collect();
    let mut vectors = match embedder.embed(texts).await {
        Ok(vectors) => vectors.into_iter(),
        Err(why) => {
            warn!("code index: embedding failed: {why}");
            return Vec::new();
        }
    };
    read.into_iter()
        .map(|(path, source)| {
            let entry = source.map(|(stamp, chunks)| FileEntry {
                stamp,
                chunks: chunks
                    .into_iter()
                    .map(|(start, end, text)| Chunk {
                        start,
                        end,
                        text,
                        vector: vectors.next().unwrap_or_default(),
                    })
                    .collect(),
            });
            (path, entry)
        })
        .collect()
}

/// `text` cut into chunks of whole lines, `(first, last, text)` with lines
/// counted from 1: a chunk ends before a line at the left margin that
/// follows a blank line (a top-level item, or a paragraph), or at
/// [`MAX_LINES`].
fn chunk(text: &str) -> Vec<(usize, usize, String)> {
    let mut chunks = Vec::new();
    let mut lines: Vec<&str> = Vec::new();
    let mut start = 1;
    let mut after_blank = false;
    for (number, line) in (1..).zip(text.lines()) {
        let item = after_blank && line.starts_with(|c: char| !c.is_whitespace());
        if item || lines.len() >= MAX_LINES {
            push_chunk(&mut chunks, start, &lines);
            lines.clear();
        }
        if lines.is_empty() {
            start = number;
        }
        lines.push(line);
        after_blank = line.trim().is_empty();
    }
    push_chunk(&mut chunks, start, &lines);
    chunks
}

/// Adds `lines`, from line `start`, without their trailing blank lines,
/// unless all are blank.
fn push_chunk(chunks: &mut Vec<(usize, usize, String)>, start: usize, lines: &[&str]) {
    let Some(last) = lines.iter().rposition(|line| !line.trim().is_empty()) else {
        return;
    };
    let kept = lines.iter().take(last + 1).copied().collect::<Vec<_>>();
    chunks.push((start, start + last, kept.join("\n")));
}

/// The hashing embedder: each word of `text` counted in one of [`DIMS`]
/// buckets, as a unit vector.
fn hashed(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0; DIMS];
    for word in words(text) {
        let bucket = fnv1a(&word) % DIMS as u64;
        if let Some(slot) = vector.get_mut(bucket as usize) {
            *slot += 1.0;
        }
    }
    unit(vector)
}

/// The lowercase words of `text`, identifiers split at `_` and at case
/// changes; single letters left out.
fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .flat_map(|word| {
            let mut parts = Vec::new();
            let mut part = String::new();
            let mut lower = false;
            for c in word.chars() {
                if c.is_uppercase() && lower {
                    parts.push(std::mem::take(&mut part));
                }
                lower = c.is_lowercase() || c.is_ascii_digit();
                part.extend(c.to_lowercase());
            }
            parts.push(part);
            parts
        })
        .filter(|word| word.chars().count() > 1)
}

fn fnv1a(word: &str) -> u64 {
    word.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// `vector` scaled to length 1, or as it is when all zero.
fn unit(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vector.iter_mut().for_each(|x| *x /= norm);
    }
    vector
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The build's progress in the terminal view's status line.
#[cfg(feature = "tui")]
mod status {
    use rig_harness::prelude::*;
    use rig_tui::{AppStatus, Side, StatusItem, StatusSystems, Tone};

    use super::CodeIndex;

    /// After the other app-wide items; among the first to go.
    const INDEX: StatusItem = StatusItem::new("code-search-index", Side::Left, 95, 6);

    pub(super) fn add(app: &mut App) {
        app.init_resource::<AppStatus>().add_systems(
            PostUpdate,
            show.in_set(StatusSystems::Show)
                .run_if(resource_changed::<CodeIndex>),
        );
    }

    fn show(index: Res<CodeIndex>, mut status: ResMut<AppStatus>) {
        let text = match index.is_built() {
            true => String::new(),
            false => format!("index {}%", index.percent()),
        };
        status.0.show(INDEX.says(text, Tone::Dim));
    }
}

#[cfg(test)]
mod tests;

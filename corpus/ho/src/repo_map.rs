//! Plugin corpus (held-out): H10: Repo map in the prompt (spec `H10-repo-map.md`).
//!
//! [`RepoMapPlugin`] maps the project's git-tracked files in the background
//! at startup: each file's top-level symbols with their lines (Rust, Python,
//! JavaScript and TypeScript, by simple line parsing) and the module names
//! it refers to. Each agent's system prompt gets a `repo_map` section of its
//! own, rendered within a token budget when a turn starts and only then, so
//! the requests of one turn share their prompt: first the files the agent
//! read or changed (the calls' published `FilesRead` and `FilesChanged`),
//! then the files those refer to, then the rest by size. The `repo_map`
//! tool shows a subtree in full or the files defining a symbol. A file a
//! call changed is scanned again in the background, and after a `shell`
//! call so is every file `git status` reports.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use rig_core::tool::{FilesChanged, FilesRead};
use rig_harness::prelude::*;
use rig_harness::rig_tools::Shell;
use schemars::JsonSchema;
use serde::Deserialize;

/// The most files mapped.
const MAX_FILES: usize = 20_000;
/// Larger files are left out.
const MAX_FILE_BYTES: u64 = 1024 * 1024;
/// The most files remembered per agent as worked on.
const MAX_WORKED: usize = 50;
/// The most symbols of one file in the prompt.
const MAX_PROMPT_SYMBOLS: usize = 40;
/// The most characters of a `repo_map` answer.
const MAX_ANSWER_CHARS: usize = 40_000;
/// Characters per token, for the budget.
const CHARS_PER_TOKEN: usize = 4;
/// After the environment: the section changes from turn to turn.
const ORDER: i32 = PromptSection::ORDER_ENVIRONMENT + 10;
const TAG: &str = "repo_map";

/// The repo map, its prompt section and the `repo_map` tool.
#[derive(Default)]
pub struct RepoMapPlugin;

impl Plugin for RepoMapPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .init_resource::<RepoMap>()
            .add_setting::<RepoMapSettings>("repo_map")
            .register_type::<WorkedOn>()
            .register_required_components::<Agent, WorkedOn>()
            .add_open_tool(
                "repo_map",
                "The project's map: with `path_prefix`, every file under it with its top-level \
                 symbols and their lines; with `symbol`, the files and lines that define it. \
                 The `repo_map` prompt section lists the project's files with their top-level \
                 symbols; call this to find where something is defined, or for a directory's \
                 full map, before searching.",
                ToolOptions {
                    footprint: Some(Touches::Reads(Scope::Anything)),
                },
                answer,
            )
            .add_systems(Startup, map_project)
            .add_observer(give_section)
            .add_observer(render_on_turn)
            .add_observer(note_work)
            .add_observer(scanned);
    }
}

/// The `[repo_map]` settings.
#[derive(Resource, Deserialize, Clone)]
#[serde(default, deny_unknown_fields)]
pub struct RepoMapSettings {
    /// The prompt section's size, in tokens (about four characters each).
    pub budget: usize,
}

impl Default for RepoMapSettings {
    fn default() -> Self {
        Self { budget: 1500 }
    }
}

/// A top-level symbol of a file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol {
    /// Its line, from 1.
    pub line: usize,
    /// What it is: `fn`, `struct`, `impl`, `def`, `class`, `export const`, ….
    pub kind: String,
    /// Its name; an `impl`'s header.
    pub name: String,
}

/// One mapped file.
#[derive(Clone, Debug)]
struct FileMap {
    /// When it was read, so an older scan never replaces a newer one.
    at: Instant,
    size: u64,
    symbols: Vec<Symbol>,
    /// The module names its `mod`, `use` and `import` lines name.
    refs: Vec<String>,
}

/// The project's map, by path relative to the project.
#[derive(Resource, Default)]
pub struct RepoMap {
    root: PathBuf,
    built: bool,
    truncated: bool,
    files: BTreeMap<String, FileMap>,
}

impl RepoMap {
    /// Whether the startup scan is done.
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// The symbols of the file at `path`, relative to the project.
    pub fn symbols(&self, path: &str) -> Option<&[Symbol]> {
        self.files.get(path).map(|file| file.symbols.as_slice())
    }

    /// The files in prompt order for an agent that worked on `worked`, most
    /// recent first: those, then the files they refer to, then the rest by
    /// size.
    fn ranked<'a>(&'a self, worked: &'a [String]) -> Vec<&'a str> {
        let mut ranked: Vec<&str> = worked
            .iter()
            .map(String::as_str)
            .filter(|path| self.files.contains_key(*path))
            .collect();
        let first: HashSet<&str> = ranked.iter().copied().collect();
        let names: HashSet<&str> = ranked
            .iter()
            .filter_map(|path| self.files.get(*path))
            .flat_map(|file| file.refs.iter().map(String::as_str))
            .collect();
        let mut others: Vec<(bool, u64, &str)> = self
            .files
            .iter()
            .filter(|(path, _)| !first.contains(path.as_str()))
            .map(|(path, file)| {
                let referenced = module_name(path).is_some_and(|name| names.contains(name));
                (referenced, file.size, path.as_str())
            })
            .collect();
        others.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(b.2)));
        ranked.extend(others.into_iter().map(|(_, _, path)| path));
        ranked
    }

    /// The prompt section for an agent that worked on `worked`, within
    /// `budget` tokens: the high-ranked files with their symbols, the
    /// others by path alone, as many as fit.
    fn render(&self, worked: &[String], budget: usize) -> String {
        let budget = budget.saturating_mul(CHARS_PER_TOKEN);
        let symbols_budget = budget / 5 * 3;
        let mut text = String::from(
            "The project's files, those you worked on first; the first ones with their \
             top-level symbols (line, kind, name).\n",
        );
        let ranked = self.ranked(worked);
        let mut with_symbols = true;
        for (shown, path) in ranked.iter().enumerate() {
            if with_symbols {
                let entry = self.entry(path, MAX_PROMPT_SYMBOLS);
                if text.len() + entry.len() <= symbols_budget {
                    text.push_str(&entry);
                    continue;
                }
                with_symbols = false;
            }
            // Room is kept for the note on what is left out.
            if text.len() + path.len() + 64 > budget {
                let left = ranked.len() - shown;
                text.push_str(&format!("({left} more files: call `repo_map`.)\n"));
                break;
            }
            text.push_str(&rig_tools::fs::shown(Path::new(path)).to_string_lossy());
            text.push('\n');
        }
        if self.truncated {
            text.push_str(&format!("(Only the first {MAX_FILES} files are mapped.)\n"));
        }
        text
    }

    /// `path` and up to `max` of its symbols, a line each.
    fn entry(&self, path: &str, max: usize) -> String {
        let mut entry = format!("{}\n", rig_tools::fs::shown(Path::new(path)).display());
        let symbols = self.files.get(path).map(|file| file.symbols.as_slice());
        let symbols = symbols.unwrap_or_default();
        for symbol in symbols.iter().take(max) {
            entry.push_str(&format!(
                "  {} {} {}\n",
                symbol.line, symbol.kind, symbol.name
            ));
        }
        if symbols.len() > max {
            entry.push_str(&format!("  … {} more\n", symbols.len() - max));
        }
        entry
    }

    /// What a `repo_map` call asks for.
    fn look_up(&self, args: &MapArgs) -> String {
        let prefix = args.path_prefix.as_deref().unwrap_or_default();
        let prefix = prefix.trim_start_matches("./");
        let under = self
            .files
            .iter()
            .filter(|(path, _)| path.starts_with(prefix));
        let mut lines: Vec<String> = match args.symbol.as_deref() {
            Some(symbol) => {
                let found = |exact: bool| -> Vec<String> {
                    let lower = symbol.to_lowercase();
                    under
                        .clone()
                        .flat_map(|(path, file)| file.symbols.iter().map(move |s| (path, s)))
                        .filter(|(_, s)| match exact {
                            true => s.name == symbol,
                            false => s.name.to_lowercase().contains(&lower),
                        })
                        .map(|(path, s)| {
                            let path = rig_tools::fs::shown(Path::new(path));
                            format!("{}:{} {} {}", path.display(), s.line, s.kind, s.name)
                        })
                        .collect()
                };
                let exact = found(true);
                if exact.is_empty() {
                    found(false)
                } else {
                    exact
                }
            }
            None => under
                .map(|(path, _)| self.entry(path, usize::MAX))
                .collect(),
        };
        if lines.is_empty() {
            lines.push("Nothing in the map matches.".to_owned());
        }
        let mut answer = match self.built {
            true => String::new(),
            false => "(The map is still being built; this is part of it.)\n".to_owned(),
        };
        for line in lines {
            if answer.len() + line.len() > MAX_ANSWER_CHARS {
                answer.push_str("(Cut: ask for a narrower `path_prefix`.)\n");
                break;
            }
            answer.push_str(line.trim_end());
            answer.push('\n');
        }
        answer
    }

    /// Takes in a scan's files, unless a later scan of a file came first.
    fn merge(&mut self, files: Vec<(String, Instant, Option<FileMap>)>) {
        for (path, at, file) in files {
            if self.files.get(&path).is_some_and(|old| old.at > at) {
                continue;
            }
            match file {
                Some(file) => {
                    self.files.insert(path, file);
                }
                None => {
                    self.files.remove(&path);
                }
            }
        }
    }
}

/// The files an agent read or changed this session, most recent first.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved)]
pub struct WorkedOn(pub Vec<String>);

impl WorkedOn {
    fn touch(&mut self, path: String) {
        self.0.retain(|old| *old != path);
        self.0.insert(0, path);
        self.0.truncate(MAX_WORKED);
    }
}

/// On the prompt section of an agent's map.
#[derive(Component)]
struct MapSection;

/// A scan done off the loop: each file, `None` when it is gone or not
/// mapped; `full` for the startup scan.
#[derive(Component)]
struct Scanned {
    full: bool,
    truncated: bool,
    files: Vec<(String, Instant, Option<FileMap>)>,
}

/// The background scans' pool.
fn pool() -> &'static AsyncComputeTaskPool {
    AsyncComputeTaskPool::get_or_init(TaskPool::default)
}

/// Starts the startup scan.
fn map_project(
    mut map: ResMut<RepoMap>,
    project: Res<ProjectRoot>,
    processes: Res<Processes>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    map.root = std::fs::canonicalize(&project.0).unwrap_or_else(|_| project.0.clone());
    let (root, git) = (map.root.clone(), processes.launcher().command("git"));
    spawn_scan(&mut commands, &wake, move || full_scan(git, &root));
}

/// Runs `scan` on a thread of its own, ending as [`Scanned`] on an entity.
fn spawn_scan(
    commands: &mut Commands,
    wake: &Wake,
    scan: impl FnOnce() -> Scanned + Send + 'static,
) {
    let work = async move {
        blocking(move || Ok(scan())).await.unwrap_or(Scanned {
            full: false,
            truncated: false,
            files: Vec::new(),
        })
    };
    commands.spawn((
        Name::new("repo map scan"),
        Running::spawn_into::<Scanned, _>(pool(), wake, work),
    ));
}

/// Takes a finished scan into the map.
fn scanned(
    add: On<Add<Scanned>>,
    mut done: Query<&mut Scanned>,
    mut map: ResMut<RepoMap>,
    mut commands: Commands,
) {
    let Ok(mut scan) = done.get_mut(add.entity) else {
        return;
    };
    commands.entity(add.entity).despawn();
    map.merge(std::mem::take(&mut scan.files));
    if scan.full {
        map.built = true;
        map.truncated = scan.truncated;
    }
}

/// Gives each agent its own map section, empty until its first turn.
fn give_section(added: On<Add<WorkedOn>>, mut commands: Commands) {
    commands.spawn((
        Name::new("repo map section"),
        MapSection,
        PromptSection::new(ORDER, TAG, ""),
        SectionOf(added.entity),
    ));
}

/// Renders the agent's map into its section as each of its turns starts,
/// before the turn's first request is built, so the requests of one turn
/// share it.
fn render_on_turn(
    turn: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    agents: Query<&WorkedOn>,
    mut sections: Query<(&mut PromptSection, &SectionOf), With<MapSection>>,
    map: Res<RepoMap>,
    settings: Res<RepoMapSettings>,
) {
    let Ok(&TurnOf(agent)) = turns.get(turn.entity) else {
        return;
    };
    if !map.built {
        return;
    }
    let worked = agents.get(agent).map(|worked| worked.0.as_slice());
    let text = map.render(worked.unwrap_or_default(), settings.budget);
    for (mut section, of) in &mut sections {
        if of.0 == agent && section.text != text {
            section.text.clone_from(&text);
        }
    }
}

/// Notes the files each settled tool call read or changed for its agent,
/// scans the changed ones again, and after a `shell` call those
/// `git status` reports.
fn note_work(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &ForAgent, Option<&Published>)>,
    mut agents: Query<&mut WorkedOn>,
    map: Res<RepoMap>,
    processes: Res<Processes>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok((request, &ForAgent(agent), published)) = calls.get(settled.entity) else {
        return;
    };
    let facts = published.map(|published| &published.0);
    let read = facts.and_then(|facts| facts.get::<FilesRead>().ok().flatten());
    let changed = facts.and_then(|facts| facts.get::<FilesChanged>().ok().flatten());
    let root = map.root.clone();
    let changed: Vec<String> = changed
        .iter()
        .flat_map(FilesChanged::paths)
        .filter_map(|path| relative(&root, path))
        .collect();
    if let Ok(mut worked) = agents.get_mut(agent) {
        let read = read.iter().flat_map(FilesRead::paths);
        for path in read.filter_map(|path| relative(&root, path)) {
            worked.touch(path);
        }
        for path in &changed {
            worked.touch(path.clone());
        }
    }
    if !changed.is_empty() {
        let root = root.clone();
        spawn_scan(&mut commands, &wake, move || scan_paths(&root, changed));
    }
    if request.call.function.name.as_str() == Shell::NAME {
        let launcher = processes.launcher();
        let (status, top) = (launcher.command("git"), launcher.command("git"));
        spawn_scan(&mut commands, &wake, move || {
            let paths = git_status(status, top, &root);
            scan_paths(&root, paths)
        });
    }
}

/// The arguments of `repo_map`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct MapArgs {
    /// Only files whose path, relative to the project, starts with this,
    /// such as `src/net/`; the whole project without it.
    #[serde(default)]
    path_prefix: Option<String>,
    /// The files and lines that define a symbol of this name, instead of
    /// the full map.
    #[serde(default)]
    symbol: Option<String>,
}

/// Answers a `repo_map` call from the map as it stands.
fn answer(
    called: On<ToolCalled<MapArgs>>,
    map: Res<RepoMap>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("ho-repo-map"),
            ToolResult::success(map.look_up(&called.args).into()),
        );
    }
}

/// `path`, relative to the project unless absolute, made relative to
/// `root`; `None` outside it.
fn relative(root: &Path, path: &str) -> Option<String> {
    let path = root.join(path);
    let path = std::fs::canonicalize(&path).unwrap_or(path);
    let path = path.strip_prefix(root).ok()?.to_str()?;
    (!path.is_empty()).then(|| path.to_owned())
}

/// The name other files use for the module at `path`: its stem, or its
/// directory's name for `mod.rs`, `lib.rs`, `index.js`, `__init__.py`, ….
fn module_name(path: &str) -> Option<&str> {
    let path = Path::new(path);
    let stem = path.file_stem()?.to_str()?;
    match stem {
        "mod" | "lib" | "main" | "index" | "__init__" => path.parent()?.file_name()?.to_str(),
        _ => Some(stem),
    }
}

/// The startup scan: the files git tracks in `root`, the first
/// [`MAX_FILES`] of them.
fn full_scan(mut git: Command, root: &Path) -> Scanned {
    let output = git
        .args(["ls-files", "-z", "--cached"])
        .current_dir(root)
        .output();
    let stdout = output
        .ok()
        .filter(|output| output.status.success())
        .map(|output| output.stdout)
        .unwrap_or_default();
    let mut paths: Vec<&str> = stdout
        .split(|&byte| byte == 0)
        .filter_map(|path| std::str::from_utf8(path).ok())
        .filter(|path| !path.is_empty())
        .collect();
    paths.dedup();
    let truncated = paths.len() > MAX_FILES;
    paths.truncate(MAX_FILES);
    let files = paths
        .into_iter()
        .filter_map(|path| {
            let at = Instant::now();
            let file = scan_file(&root.join(path))?;
            Some((path.to_owned(), at, Some(file)))
        })
        .collect();
    Scanned {
        full: true,
        truncated,
        files,
    }
}

/// Scans `paths` of `root` again; one that is gone or not mapped leaves
/// the map.
fn scan_paths(root: &Path, paths: Vec<String>) -> Scanned {
    let files = paths
        .into_iter()
        .map(|path| {
            let at = Instant::now();
            let file = scan_file(&root.join(&path));
            (path, at, file)
        })
        .collect();
    Scanned {
        full: false,
        truncated: false,
        files,
    }
}

/// The files `git status` reports changed, relative to `root`.
fn git_status(mut status: Command, mut top: Command, root: &Path) -> Vec<String> {
    let run = |command: &mut Command| {
        let output = command.current_dir(root).output().ok()?;
        output.status.success().then_some(output.stdout)
    };
    let top = run(top.args(["rev-parse", "--show-toplevel"]));
    let top = top.and_then(|top| String::from_utf8(top).ok());
    let status = run(status.args(["status", "--porcelain", "-z", "--untracked-files=all"]));
    let (Some(top), Some(status)) = (top, status) else {
        return Vec::new();
    };
    let top = PathBuf::from(top.trim_end());
    let mut paths = Vec::new();
    let mut fields = status
        .split(|&byte| byte == 0)
        .filter_map(|field| std::str::from_utf8(field).ok());
    while let Some(field) = fields.next() {
        let (Some(code), Some(path)) = (field.get(..2), field.get(3..)) else {
            continue;
        };
        paths.push(path.to_owned());
        // A rename or copy names the old path next.
        if code.contains(['R', 'C'])
            && let Some(old) = fields.next()
        {
            paths.push(old.to_owned());
        }
    }
    paths
        .iter()
        .filter_map(|path| relative(root, &top.join(path).to_string_lossy()))
        .collect()
}

/// The map of the file at `path`; `None` when it is gone, too large or
/// binary.
fn scan_file(path: &Path) -> Option<FileMap> {
    let meta = std::fs::metadata(path).ok().filter(|meta| meta.is_file())?;
    if meta.len() > MAX_FILE_BYTES {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    if bytes.iter().take(8000).any(|&byte| byte == 0) {
        return None;
    }
    let text = String::from_utf8_lossy(&bytes);
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default();
    let (symbols, refs) = parse(extension, &text);
    Some(FileMap {
        at: Instant::now(),
        size: meta.len(),
        symbols,
        refs,
    })
}

/// The top-level symbols of a file of `extension` and the module names it
/// refers to.
fn parse(extension: &str, text: &str) -> (Vec<Symbol>, Vec<String>) {
    let parse_line: fn(&str) -> Line = match extension {
        "rs" => rust_line,
        "py" | "pyi" => python_line,
        "js" | "jsx" | "mjs" | "cjs" | "ts" | "tsx" | "mts" | "cts" => script_line,
        _ => return (Vec::new(), Vec::new()),
    };
    let mut symbols = Vec::new();
    let mut refs: Vec<String> = Vec::new();
    for (index, line) in text.lines().enumerate() {
        // Top level only: nothing indented.
        if line.starts_with(char::is_whitespace) {
            continue;
        }
        match parse_line(line) {
            Line::Symbol(kind, name) => symbols.push(Symbol {
                line: index + 1,
                kind,
                name,
            }),
            Line::Refs(names) => refs.extend(names),
            Line::Nothing => {}
        }
    }
    refs.sort();
    refs.dedup();
    (symbols, refs)
}

/// What one top-level line says.
enum Line {
    Symbol(String, String),
    Refs(Vec<String>),
    Nothing,
}

/// The identifier `text` starts with.
fn ident(text: &str) -> &str {
    let end = text
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$'))
        .unwrap_or(text.len());
    text.get(..end).unwrap_or_default()
}

/// The identifiers in `text` but `skip`.
fn words(text: &str, skip: &[&str]) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|word| !word.is_empty() && !skip.contains(word))
        .map(str::to_owned)
        .collect()
}

/// `text` after `prefix` and the spaces after it.
fn after<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    text.strip_prefix(prefix).map(str::trim_start)
}

/// `text` after the word `word` and the spaces after it.
fn keyword<'a>(text: &'a str, word: &str) -> Option<&'a str> {
    let rest = text.strip_prefix(word)?;
    rest.starts_with(char::is_whitespace)
        .then(|| rest.trim_start())
}

fn symbol(kind: &str, name: &str) -> Line {
    match name.is_empty() {
        true => Line::Nothing,
        false => Line::Symbol(kind.to_owned(), name.to_owned()),
    }
}

fn rust_line(line: &str) -> Line {
    let mut rest = line.trim_end();
    if let Some(public) = rest.strip_prefix("pub") {
        rest = match public.strip_prefix('(') {
            Some(scope) => scope.split_once(')').map_or("", |(_, rest)| rest),
            None => public,
        }
        .trim_start();
    }
    for modifier in ["default ", "async ", "const ", "unsafe ", "extern \"C\" "] {
        rest = rest.strip_prefix(modifier).unwrap_or(rest).trim_start();
    }
    for kind in ["fn", "struct", "enum", "trait"] {
        if let Some(name) = keyword(rest, kind) {
            return symbol(kind, ident(name));
        }
    }
    if let Some(header) = rest.strip_prefix("impl")
        && header.starts_with([' ', '<'])
    {
        let header = header.split(['{', ';']).next().unwrap_or_default();
        let header = header.split(" where").next().unwrap_or_default();
        return symbol("impl", header.trim());
    }
    if let Some(name) = after(rest, "mod ") {
        return Line::Refs(vec![ident(name).to_owned()]);
    }
    if let Some(path) = after(rest, "use ") {
        let skip = ["crate", "self", "super", "as", "std", "core", "alloc"];
        return Line::Refs(words(path, &skip));
    }
    Line::Nothing
}

fn python_line(line: &str) -> Line {
    let rest = line.strip_prefix("async ").unwrap_or(line).trim_start();
    if let Some(name) = after(rest, "def ") {
        return symbol("def", ident(name));
    }
    if let Some(name) = after(rest, "class ") {
        return symbol("class", ident(name));
    }
    if rest.starts_with("import ") || rest.starts_with("from ") {
        return Line::Refs(words(rest, &["import", "from", "as"]));
    }
    Line::Nothing
}

fn script_line(line: &str) -> Line {
    if let Some(module) = quoted_module(line) {
        return Line::Refs(vec![module]);
    }
    let (exported, mut rest) = match after(line, "export ") {
        Some(rest) => (true, rest),
        None => (false, line),
    };
    let default = after(rest, "default ");
    rest = default.unwrap_or(rest);
    for modifier in ["declare ", "abstract ", "async "] {
        rest = after(rest, modifier).unwrap_or(rest);
    }
    if let Some(name) = after(rest, "function") {
        return symbol("function", ident(name.trim_start_matches('*').trim_start()));
    }
    if let Some(name) = after(rest, "class ") {
        return symbol("class", ident(name));
    }
    if !exported {
        return Line::Nothing;
    }
    for kind in ["const", "let", "var", "interface", "type", "enum"] {
        if let Some(name) = keyword(rest, kind) {
            return symbol(&format!("export {kind}"), ident(name));
        }
    }
    match default {
        Some(_) => symbol("export default", ident(rest)),
        None => Line::Nothing,
    }
}

/// The module a JavaScript `import … from '…'`, `import '…'` or
/// `require('…')` line names: the last part of its path, without its
/// extension.
fn quoted_module(line: &str) -> Option<String> {
    let start = if line.starts_with("import ") || line.starts_with("export ") {
        line.find(" from ")
            .map(|at| at + 6)
            .or_else(|| line.starts_with("import ").then_some(7))?
    } else {
        line.find("require(")? + 8
    };
    let quoted = line.get(start..)?.trim_start();
    let quote = quoted
        .chars()
        .next()
        .filter(|c| matches!(c, '\'' | '"' | '`'))?;
    let path = quoted.get(1..)?.split(quote).next()?;
    let last = path.rsplit('/').next()?;
    let stem = Path::new(last).file_stem()?.to_str()?;
    Some(stem.to_owned())
}

#[cfg(test)]
mod tests;

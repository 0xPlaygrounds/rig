//! Protected paths (corpus spec 02): the file tools never read secrets
//! nor write outside the project or into `.git/`.
//!
//! The rules are `deny_read` and `deny_write` globs: defaults, then
//! `RIG_HOME/paths.toml`, then the project's `.rig/paths.toml`. A Gate
//! system checks every path a tool call names (its `path`, `file` and
//! `paths` arguments) when the call is about to run, resolved first (`~`,
//! `..` and symbolic links) so none sneaks past, and refuses a denied one.
//! A Prepare system runs each `search` through rig-tools' search as data,
//! hidden files included, and drops the matches in read-protected files,
//! publishing what it dropped. `/paths` lists the rules and what each
//! blocked.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use glob::{MatchOptions, Pattern};
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ToolAdapter;
use rig_core::tool::{ContextValue, ToolContext};
use rig_harness::prelude::*;
use rig_harness::rig_ecs::effects::Handler;
use rig_tools::{Found, Search, SearchArgs, find};
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// The rules every project starts with.
const DEFAULT_READ: &[&str] = &[".env*", "**/*.pem", "~/.ssh/**"];
const DEFAULT_WRITE: &[&str] = &[".git/**"];

/// The arguments that name paths, in any tool's schema.
const PATH_ARGUMENTS: &[&str] = &["path", "file", "paths"];

/// The tool whose matches are filtered.
const SEARCH: &str = "search";

/// Adds the path guard and `/paths`.
#[derive(Default)]
pub struct ProtectedPathsPlugin;

impl Plugin for ProtectedPathsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>();
        if !app.world().contains_resource::<PathPolicy>() {
            let world = app.world();
            let root = world.resource::<ProjectRoot>().0.clone();
            let files = [
                RigHome::of(world).root().join("paths.toml"),
                root.join(".rig").join("paths.toml"),
            ];
            app.insert_resource(PathPolicy::load(&root, &files));
        }
        app.init_resource::<PathGuardCounts>()
            .add_systems(
                AgentPass,
                (
                    filter_searches.in_set(Stage::Prepare),
                    guard.in_set(Stage::Gate),
                ),
            )
            .add_observer(count_hidden)
            .add_command(
                "paths",
                "List the protected-path rules and what each blocked",
                list_rules,
            );
    }
}

/// Whether a rule guards reading or writing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Access {
    /// `deny_read`.
    Read,
    /// `deny_write`.
    Write,
}

/// What a rule matches against a resolved path.
#[derive(Clone, Debug)]
enum Matcher {
    /// A glob without `/`: any component of the path, such as `.env*`.
    Component(Pattern),
    /// A glob relative to the project root; one starting with `**/` also
    /// matches anywhere outside it.
    Root(Pattern, Option<Pattern>),
    /// An absolute glob, `~/` expanded.
    Absolute(Pattern),
    /// Anything outside the project root.
    Outside,
}

/// One rule: what it denies, as written, and where it comes from.
#[derive(Clone, Debug)]
pub struct Rule {
    /// Reading or writing.
    pub access: Access,
    /// The glob as written, or "outside the project".
    pub glob: String,
    /// Where it was read: "default" or the file.
    pub source: String,
    matcher: Matcher,
}

impl Rule {
    /// The key its counts are kept under.
    fn label(&self) -> String {
        let key = match self.access {
            Access::Read => "deny_read",
            Access::Write => "deny_write",
        };
        format!("{key} {}", self.glob)
    }

    fn matches(&self, path: &Path, root: &Path) -> bool {
        let options = MatchOptions {
            case_sensitive: true,
            require_literal_separator: true,
            require_literal_leading_dot: false,
        };
        let relative = path.strip_prefix(root).ok();
        match &self.matcher {
            Matcher::Component(pattern) => relative
                .unwrap_or(path)
                .components()
                .any(|part| pattern.matches_with(&part.as_os_str().to_string_lossy(), options)),
            Matcher::Root(pattern, anywhere) => {
                relative.is_some_and(|relative| pattern.matches_path_with(relative, options))
                    || anywhere
                        .as_ref()
                        .is_some_and(|anywhere| anywhere.matches_path_with(path, options))
            }
            Matcher::Absolute(pattern) => pattern.matches_path_with(path, options),
            Matcher::Outside => relative.is_none(),
        }
    }
}

/// `paths.toml`.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct PathsFile {
    /// Globs no tool reads.
    #[serde(default)]
    pub deny_read: Vec<String>,
    /// Globs no tool writes.
    #[serde(default)]
    pub deny_write: Vec<String>,
}

/// The active rules, the project root they are relative to, and what could
/// not be loaded.
#[derive(Resource, Clone, Debug)]
pub struct PathPolicy(Arc<Policy>);

#[derive(Debug)]
struct Policy {
    root: PathBuf,
    home: Option<PathBuf>,
    rules: Vec<Rule>,
    problems: Vec<String>,
}

impl PathPolicy {
    /// The default rules for the project at `root`, then those of each of
    /// `files` that exists, in order.
    pub fn load(root: &Path, files: &[PathBuf]) -> Self {
        let root = real(&std::path::absolute(root).unwrap_or_else(|_| root.to_path_buf()));
        let home = std::env::home_dir().map(|home| real(&home));
        let mut policy = Policy {
            root,
            home,
            rules: Vec::new(),
            problems: Vec::new(),
        };
        policy.add(Access::Read, DEFAULT_READ, "default");
        policy.add(Access::Write, DEFAULT_WRITE, "default");
        policy.rules.push(Rule {
            access: Access::Write,
            glob: "outside the project".to_owned(),
            source: "default".to_owned(),
            matcher: Matcher::Outside,
        });
        for file in files {
            let Ok(text) = std::fs::read_to_string(file) else {
                continue;
            };
            let source = file.display().to_string();
            match toml::from_str::<PathsFile>(&text) {
                Ok(paths) => {
                    policy.add(Access::Read, &paths.deny_read, &source);
                    policy.add(Access::Write, &paths.deny_write, &source);
                }
                Err(error) => policy.problems.push(format!("{source}: {error}")),
            }
        }
        Self(Arc::new(policy))
    }

    /// The rule that denies `access` to `raw`, if any.
    fn check(&self, access: Access, raw: &str) -> Option<&Rule> {
        let path = self.0.resolve(raw);
        self.0
            .rules
            .iter()
            .filter(|rule| rule.access == access)
            .find(|rule| rule.matches(&path, &self.0.root))
    }

    /// `found` as the `search` tool shows it, without the matches in
    /// read-protected files nor those in other hidden files (which the tool
    /// never shows), and the rule that hid each dropped protected match.
    fn filter(&self, args: &SearchArgs, found: Found) -> (String, Hidden) {
        let root = args.path.as_deref().unwrap_or(".");
        let root = Path::new(root.strip_prefix("./").unwrap_or(root));
        let mut hidden = Hidden::default();
        let mut kept = Vec::new();
        for hit in found.hits {
            if let Some(rule) = self.check(Access::Read, &hit.path.to_string_lossy()) {
                hidden.0.push(rule.label());
            } else if !is_hidden(hit.path.strip_prefix(root).unwrap_or(&hit.path)) {
                kept.push(hit.to_string());
            }
        }
        if kept.is_empty() {
            kept.push("No matches.".to_owned());
        }
        if found.stopped {
            kept.push("[stopped at 200 matches]".to_owned());
        }
        if let count @ 1.. = hidden.0.len() {
            let noun = if count == 1 { "match" } else { "matches" };
            kept.push(format!("[{count} {noun} in protected files hidden]"));
        }
        (kept.join("\n"), hidden)
    }
}

/// Whether a walk below its root skips `path` as hidden: a component starts
/// with `.`.
fn is_hidden(path: &Path) -> bool {
    path.components().any(|part| match part {
        Component::Normal(name) => name.to_string_lossy().starts_with('.'),
        _ => false,
    })
}

impl Policy {
    fn add(&mut self, access: Access, globs: &[impl AsRef<str>], source: &str) {
        for glob in globs.iter().map(AsRef::as_ref) {
            match self.matcher(glob) {
                Ok(matcher) => self.rules.push(Rule {
                    access,
                    glob: glob.to_owned(),
                    source: source.to_owned(),
                    matcher,
                }),
                Err(error) => self.problems.push(format!("{source}: `{glob}`: {error}")),
            }
        }
    }

    fn matcher(&self, glob: &str) -> Result<Matcher, glob::PatternError> {
        if let Some(rest) = glob.strip_prefix("~/") {
            let home = self.home.as_deref().unwrap_or(Path::new("/"));
            return Pattern::new(&home.join(rest).to_string_lossy()).map(Matcher::Absolute);
        }
        if glob.starts_with('/') {
            return Pattern::new(glob).map(Matcher::Absolute);
        }
        if !glob.contains('/') {
            return Pattern::new(glob).map(Matcher::Component);
        }
        let anywhere = match glob.starts_with("**/") {
            true => Some(Pattern::new(&format!("/{glob}"))?),
            false => None,
        };
        Ok(Matcher::Root(Pattern::new(glob)?, anywhere))
    }

    /// `raw` as the file system sees it: `~` expanded, relative to the
    /// root, symbolic links followed as far as the path exists, and `..`
    /// and `.` taken out of the rest.
    fn resolve(&self, raw: &str) -> PathBuf {
        let home = self.home.as_deref().unwrap_or(Path::new("/"));
        let path = match raw.strip_prefix('~') {
            Some("") => home.to_path_buf(),
            Some(rest) if rest.starts_with('/') => home.join(rest.trim_start_matches('/')),
            _ => PathBuf::from(raw),
        };
        real(&self.root.join(path))
    }
}

/// `path` (absolute) with its longest existing prefix canonicalized and
/// the rest normalized without the file system.
fn real(path: &Path) -> PathBuf {
    let parts: Vec<Component> = path.components().collect();
    for split in (0..=parts.len()).rev() {
        let (head, tail) = parts.split_at(split);
        let head: PathBuf = head.iter().collect();
        let Ok(mut resolved) = head.canonicalize() else {
            continue;
        };
        for part in tail {
            match part {
                Component::ParentDir => {
                    resolved.pop();
                }
                Component::Normal(name) => resolved.push(name),
                Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
            }
        }
        return resolved;
    }
    path.to_path_buf()
}

/// How many calls each rule blocked, and how many `search` matches it
/// hid, this session (kept across `/reload` and `/resume`).
#[derive(Resource, Reflect, Default, Debug)]
#[reflect(Resource, Saved)]
pub struct PathGuardCounts {
    /// Calls refused, by rule.
    pub blocked: HashMap<String, u32>,
    /// `search` matches hidden, by rule.
    pub hidden: HashMap<String, u32>,
}

/// What a `search` call dropped: the rule that hid each match, published
/// beside its text.
#[derive(Serialize, Deserialize, Default)]
struct Hidden(Vec<String>);

impl ContextValue for Hidden {
    const KEY: &'static str = "protected_paths_hidden";
}

/// rig-tools' `search`, hidden files included, its matches in
/// read-protected files dropped.
struct GuardedSearch(PathPolicy);

impl Tool for GuardedSearch {
    const NAME: &'static str = SEARCH;
    type Args = SearchArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        PortableTool::description(&Search)
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<SearchArgs>()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: SearchArgs,
    ) -> Result<String, ToolExecutionError> {
        let policy = self.0.clone();
        let (text, hidden) = blocking(move || {
            let found = find(&args, true)?;
            Ok::<_, ToolExecutionError>(policy.filter(&args, found))
        })
        .await?;
        if !hidden.0.is_empty() {
            context.insert_result(hidden)?;
        }
        Ok(text)
    }
}

/// What a call of the tool `name` is checked for: the built-in file tools
/// as they use their paths; any other for reading, and for writing too
/// unless it changes nothing.
fn accesses(name: &str, footprint: Footprint) -> &'static [Access] {
    match name {
        "read" | SEARCH => &[Access::Read],
        "write" => &[Access::Write],
        _ if footprint == Footprint::ReadOnly => &[Access::Read],
        _ => &[Access::Read, Access::Write],
    }
}

/// The paths a call's arguments name; a `search` without a path searches
/// the working directory.
fn named_paths(arguments: &serde_json::Map<String, serde_json::Value>, search: bool) -> Vec<&str> {
    let mut paths: Vec<&str> = PATH_ARGUMENTS
        .iter()
        .filter_map(|name| arguments.get(*name))
        .flat_map(|value| match value {
            serde_json::Value::String(path) => vec![path.as_str()],
            serde_json::Value::Array(paths) => {
                paths.iter().filter_map(serde_json::Value::as_str).collect()
            }
            _ => Vec::new(),
        })
        .collect();
    if search && paths.is_empty() {
        paths.push(".");
    }
    paths
}

/// Runs each `search` call through [`GuardedSearch`] as it is admitted.
fn filter_searches(
    mut calls: Query<(&ToolRequest, &mut CallHandler), Added<Admitted>>,
    policy: Res<PathPolicy>,
) {
    for (request, mut handler) in &mut calls {
        if request.call.function.name.as_str() == SEARCH {
            let tool = ToolAdapter::new(GuardedSearch(policy.clone()));
            *handler = CallHandler(Handler(ErasedHandler::new(tool)));
        }
    }
}

/// Refuses each tool call about to run that names a path a rule denies to
/// it.
fn guard(
    calls: Query<(Entity, &ToolRequest, Option<&Footprint>, Has<Restored>), Undecided>,
    policy: Res<PathPolicy>,
    mut counts: ResMut<PathGuardCounts>,
    mut commands: Commands,
) {
    for (call, ToolRequest { call: request }, footprint, restored) in &calls {
        let name = request.function.name.as_str();
        let accesses = accesses(name, footprint.copied().unwrap_or_default());
        let named = named_paths(&request.function.arguments, name == SEARCH);
        let denied = named.into_iter().find_map(|raw| {
            accesses
                .iter()
                .find_map(|&access| policy.check(access, raw).map(|rule| (raw, rule)))
        });
        let Some((raw, rule)) = denied else {
            continue;
        };
        if !restored {
            *counts.blocked.entry(rule.label()).or_default() += 1;
        }
        let what = match rule.access {
            Access::Read => "read-protected",
            Access::Write => "write-protected",
        };
        let why = format!(
            "`{raw}` is {what} by `{}` ({}). The call did not run; do not try another path to it.",
            rule.glob, rule.source
        );
        commands.entity(call).insert_if_new(Outcome::refused(why));
    }
}

/// Counts the matches each `search` hid, as it publishes them.
fn count_hidden(
    published: On<Add<Published>>,
    calls: Query<&Published, Without<Restored>>,
    mut counts: ResMut<PathGuardCounts>,
) {
    let Ok(Published(context)) = calls.get(published.entity) else {
        return;
    };
    for rule in context.get::<Hidden>().ok().flatten().unwrap_or_default().0 {
        *counts.hidden.entry(rule).or_default() += 1;
    }
}

/// `/paths`: the rules, what each blocked, and what is not guarded.
fn list_rules(
    In(args): In<CommandArgs>,
    policy: Res<PathPolicy>,
    counts: Res<PathGuardCounts>,
    mut notices: MessageWriter<Notice>,
) {
    let mut lines = vec![format!("Protected paths in {}:", policy.0.root.display())];
    for rule in &policy.0.rules {
        let label = rule.label();
        let blocked = counts.blocked.get(&label).copied().unwrap_or_default();
        let mut line = format!("  {label} ({}): {blocked} calls blocked", rule.source);
        if let Some(hidden) = counts.hidden.get(&label) {
            line.push_str(&format!(", {hidden} search matches hidden"));
        }
        lines.push(line);
    }
    for problem in &policy.0.problems {
        lines.push(format!("  not loaded: {problem}"));
    }
    lines.push("Guarded: the `path`, `file` and `paths` arguments of every tool.".to_owned());
    lines.push(
        "`shell` commands are not parsed: a command can still read or write any path.".to_owned(),
    );
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

//! Protected paths (corpus spec 02): the file tools never read secrets
//! nor write outside the project or into `.git/`.
//!
//! The rules are `deny_read` and `deny_write` globs: defaults, then
//! `RIG_HOME/paths.toml`, then the project's `.rig/paths.toml`. A Gate
//! system checks every path a tool call names (its `path`, `file` and
//! `paths` arguments) when the call is about to run, resolved first (`~`,
//! `..` and symbolic links) so none sneaks past, and refuses a denied one.
//! A Review system drops the matches in read-protected files from each
//! search's result, read from the hits it published, before the model sees
//! it. `/paths` lists the rules and what each blocked.

use std::collections::HashMap;
use std::path::{Component as Part, Path, PathBuf};
use std::sync::Arc;

use crate::confirm_destructive::ConfirmSystems;
use crate::plan_mode::PlanModeSystems;
use glob::{MatchOptions, Pattern};
use rig_core::effect::Answered;
use rig_core::tool::ToolOutput;
use rig_harness::prelude::*;
use rig_tools::{Found, SearchHits, render};
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The rules every project starts with.
const DEFAULT_READ: &[&str] = &[".env*", "**/*.pem", "~/.ssh/**"];
const DEFAULT_WRITE: &[&str] = &[".git/**"];

/// The arguments that name paths, in any tool's schema.
const PATH_ARGUMENTS: &[&str] = &["path", "file", "paths"];

/// The built-in search tool, which reads every file below its path.
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
            .configure_sets(
                AgentPass,
                (
                    ProtectedPathsSystems::Gate
                        .in_set(Stage::Gate)
                        .after(PlanModeSystems::Gate)
                        .before(ConfirmSystems),
                    ProtectedPathsSystems::Review.in_set(Stage::Review),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    guard.in_set(ProtectedPathsSystems::Gate),
                    filter_searches.in_set(ProtectedPathsSystems::Review),
                ),
            )
            .add_command(
                "paths",
                "List the protected-path rules and what each blocked",
                list_rules,
            );
    }
}

/// The plugin's stage systems, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProtectedPathsSystems {
    /// Gate: refuses a call that names a protected path, after plan mode
    /// refused it and before confirm-destructive asks about it.
    Gate,
    /// Review: hides the matches in read-protected files from a search's
    /// result.
    Review,
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
        // The coding tools' rule: a first component `~`.
        let path = match Path::new(raw).strip_prefix("~") {
            Ok(rest) => home.join(rest),
            Err(_) => PathBuf::from(raw),
        };
        real(&self.root.join(path))
    }
}

/// `path` (absolute) with its longest existing prefix canonicalized and
/// the rest normalized without the file system.
fn real(path: &Path) -> PathBuf {
    let parts: Vec<Part> = path.components().collect();
    for split in (0..=parts.len()).rev() {
        let (head, tail) = parts.split_at(split);
        let head: PathBuf = head.iter().collect();
        let Ok(mut resolved) = head.canonicalize() else {
            continue;
        };
        for part in tail {
            match part {
                Part::ParentDir => {
                    resolved.pop();
                }
                Part::Normal(name) => resolved.push(name),
                Part::CurDir | Part::RootDir | Part::Prefix(_) => {}
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

/// What a call of the tool `name` is checked for: the built-in file tools
/// as they use their paths; any other for reading, and for writing too
/// unless it changes nothing.
fn accesses(name: &str, footprint: &Footprint) -> &'static [Access] {
    match name {
        "read" | SEARCH => &[Access::Read],
        "write" => &[Access::Write],
        _ if footprint.0.reads_only() => &[Access::Read],
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

/// Drops the matches in read-protected files from the result of each call
/// that published search hits, before it is committed, and counts them by
/// the rule that hid them.
fn filter_searches(
    mut calls: Query<
        (Entity, &mut Outcome, &Published, Has<Restored>),
        (ReadyToSettle, ToolCalls, Without<Filtered>),
    >,
    policy: Res<PathPolicy>,
    mut counts: ResMut<PathGuardCounts>,
    mut commands: Commands,
) {
    for (call, mut outcome, Published(published), restored) in &mut calls {
        commands.entity(call).insert(Filtered);
        let Ok(Answered::ToolResult { result }) = &outcome.0 else {
            continue;
        };
        let Ok(Some(SearchHits(found))) = published.get::<SearchHits>() else {
            continue;
        };
        if result.is_error() {
            continue;
        }
        let mut hidden = Vec::new();
        let mut kept = Found {
            stopped: found.stopped,
            ..Found::default()
        };
        for hit in found.hits {
            match policy.check(Access::Read, &hit.path.to_string_lossy()) {
                Some(rule) => hidden.push(rule.label()),
                None => kept.hits.push(hit),
            }
        }
        if hidden.is_empty() {
            continue;
        }
        let noun = if hidden.len() == 1 {
            "match"
        } else {
            "matches"
        };
        let text = format!(
            "{}\n[{} {noun} in protected files hidden]",
            render(&kept),
            hidden.len()
        );
        *outcome = Outcome::from(ToolResult::success(ToolOutput::text(text)));
        if !restored {
            for rule in hidden {
                *counts.hidden.entry(rule).or_default() += 1;
            }
        }
    }
}

/// On a call whose search hits were filtered: once is enough.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Filtered;

/// Refuses each tool call about to run that names a path a rule denies to
/// it.
fn guard(
    mut calls: Query<
        (&ToolRequest, &Footprint, Has<Restored>, &mut Decision),
        (ReadyToStart, ToolCalls),
    >,
    policy: Res<PathPolicy>,
    mut counts: ResMut<PathGuardCounts>,
) {
    for (ToolRequest { call: request }, footprint, restored, mut decision) in &mut calls {
        let name = request.function.name.as_str();
        let accesses = accesses(name, footprint);
        let named = named_paths(&request.function.arguments, name == SEARCH);
        let denied = named.into_iter().find_map(|raw| {
            accesses
                .iter()
                .find_map(|&access| policy.check(access, raw).map(|rule| (raw, rule)))
        });
        let Some((raw, rule)) = denied else {
            continue;
        };
        let what = match rule.access {
            Access::Read => "read-protected",
            Access::Write => "write-protected",
        };
        let why = format!(
            "`{raw}` is {what} by `{}` ({}). The call did not run; do not try another path to it.",
            rule.glob, rule.source
        );
        if decision.decide(Decider::plugin("protected-paths"), Outcome::refused(why)) && !restored {
            *counts.blocked.entry(rule.label()).or_default() += 1;
        }
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

//! Protected paths (corpus spec 02): the file tools never read secrets
//! nor write outside the project or into `.git/`.
//!
//! The rules are `deny_read` and `deny_write` globs: defaults, then the
//! `[protected-paths]` table of the user's `settings.toml`, then the
//! project's `.rig/paths.toml`. A Gate system
//! checks every path a tool call names (its `path`, `file` and `paths`
//! arguments) when the call is about to run, resolved first (`~`, `..` and
//! symbolic links) so none sneaks past, and refuses a denied one. A Review
//! system takes the matches in read-protected files out of `search`
//! results. `/paths` lists the rules and what each blocked.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use glob::{MatchOptions, Pattern};
use rig_core::effect;
use rig_core::tool::ToolOutput as Output;
use rig_harness::prelude::*;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The rules every project starts with.
const DEFAULT_READ: &[&str] = &[".env*", "**/*.pem", "~/.ssh/**"];
const DEFAULT_WRITE: &[&str] = &[".git/**"];

/// The arguments that name paths, in any tool's schema.
const PATH_ARGUMENTS: &[&str] = &["path", "file", "paths"];

/// The tool whose results name files, `path:line: text`.
const SEARCH: &str = "search";

/// Adds the path guard and `/paths`.
#[derive(Default)]
pub struct ProtectedPathsPlugin;

impl Plugin for ProtectedPathsPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<PathPolicy>() {
            app.add_setting::<PathsFile>("protected-paths")
                .init_resource::<ProjectRoot>();
            let world = app.world();
            let root = world.resource::<ProjectRoot>().0.clone();
            let user = (
                "settings.toml [protected-paths]",
                world.resource::<PathsFile>(),
            );
            let policy = PathPolicy::load(&root, user, &[root.join(".rig").join("paths.toml")]);
            app.insert_resource(policy);
        }
        app.init_resource::<PathGuardCounts>()
            .add_systems(
                AgentPass,
                (
                    guard.in_set(Stage::Gate),
                    hide_matches.in_set(Stage::Review),
                ),
            )
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

impl Access {
    fn key(self) -> &'static str {
        match self {
            Self::Read => "deny_read",
            Self::Write => "deny_write",
        }
    }
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
        format!("{} {}", self.access.key(), self.glob)
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

/// `paths.toml`, and the user's `[protected-paths]` settings.
#[derive(Resource, Deserialize, Default)]
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

/// A call a rule refused.
struct Denied<'a> {
    rule: &'a Rule,
    raw: String,
}

impl Denied<'_> {
    fn message(&self) -> String {
        let what = match self.rule.access {
            Access::Read => "read-protected",
            Access::Write => "write-protected",
        };
        format!(
            "`{}` is {what} by `{}` ({}). The call did not run; do not try another path to it.",
            self.raw, self.rule.glob, self.rule.source
        )
    }
}

impl PathPolicy {
    /// The default rules for the project at `root`, then the user's from
    /// `source`, then those of each of `files` that exists, in order.
    pub fn load(root: &Path, (source, user): (&str, &PathsFile), files: &[PathBuf]) -> Self {
        let root = real(&std::path::absolute(root).unwrap_or_else(|_| root.to_path_buf()));
        let home = std::env::home_dir().map(|home| real(&home));
        let mut policy = Policy {
            root,
            home,
            rules: Vec::new(),
            problems: Vec::new(),
        };
        policy.add(Access::Read, DEFAULT_READ.iter().copied(), "default");
        policy.add(Access::Write, DEFAULT_WRITE.iter().copied(), "default");
        policy.rules.push(Rule {
            access: Access::Write,
            glob: "outside the project".to_owned(),
            source: "default".to_owned(),
            matcher: Matcher::Outside,
        });
        policy.add_file(user, source);
        for file in files {
            let Ok(text) = std::fs::read_to_string(file) else {
                continue;
            };
            let source = file.display().to_string();
            match toml::from_str::<PathsFile>(&text) {
                Ok(paths) => policy.add_file(&paths, &source),
                Err(error) => policy.problems.push(format!("{source}: {error}")),
            }
        }
        Self(Arc::new(policy))
    }

    /// The rule that denies `access` to `raw`, if any.
    fn check(&self, access: Access, raw: &str) -> Option<Denied<'_>> {
        let path = self.0.resolve(raw);
        self.0
            .rules
            .iter()
            .filter(|rule| rule.access == access)
            .find(|rule| rule.matches(&path, &self.0.root))
            .map(|rule| Denied {
                rule,
                raw: raw.to_owned(),
            })
    }

    /// The read rule hiding the file `raw`, as `search` names it.
    fn hides(&self, raw: &str) -> Option<String> {
        self.check(Access::Read, raw)
            .map(|denied| denied.rule.label())
    }
}

impl Policy {
    /// The rules of `paths`, from `source`.
    fn add_file(&mut self, paths: &PathsFile, source: &str) {
        self.add(
            Access::Read,
            paths.deny_read.iter().map(String::as_str),
            source,
        );
        self.add(
            Access::Write,
            paths.deny_write.iter().map(String::as_str),
            source,
        );
    }

    fn add<'a>(&mut self, access: Access, globs: impl Iterator<Item = &'a str>, source: &str) {
        for glob in globs {
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
fn named_paths(
    arguments: &serde_json::Map<String, serde_json::Value>,
    search: bool,
) -> Vec<String> {
    let mut paths: Vec<String> = PATH_ARGUMENTS
        .iter()
        .filter_map(|name| arguments.get(*name))
        .flat_map(|value| match value {
            serde_json::Value::String(path) => vec![path.clone()],
            serde_json::Value::Array(paths) => paths
                .iter()
                .filter_map(|path| path.as_str().map(str::to_owned))
                .collect(),
            _ => Vec::new(),
        })
        .collect();
    if search && paths.is_empty() {
        paths.push(".".to_owned());
    }
    paths
}

/// A `search` result and the read rules that hid its matches.
struct Guarded {
    result: ToolResult,
    hidden: Vec<String>,
}

/// Refuses each tool call about to run that names a path a rule denies to
/// it.
fn guard(
    calls: Query<(Entity, &ToolRequest, Option<&Footprint>), Undecided>,
    policy: Res<PathPolicy>,
    mut counts: ResMut<PathGuardCounts>,
    mut commands: Commands,
) {
    for (call, ToolRequest { call: request }, footprint) in &calls {
        let name = request.function.name.as_str();
        let accesses = accesses(name, footprint.copied().unwrap_or_default());
        let named = named_paths(&request.function.arguments, name == SEARCH);
        let denied = named.into_iter().find_map(|raw| {
            accesses
                .iter()
                .find_map(|&access| policy.check(access, &raw))
        });
        if let Some(denied) = denied {
            *counts.blocked.entry(denied.rule.label()).or_default() += 1;
            commands
                .entity(call)
                .insert_if_new(Outcome::refused(denied.message()));
        }
    }
}

/// Takes the matches in read-protected files out of each `search` result,
/// counting them. A result with none is left as it is, so reviewing it again
/// changes nothing.
fn hide_matches(
    calls: Query<(Entity, &ToolRequest, &Outcome), Unreviewed>,
    policy: Res<PathPolicy>,
    mut counts: ResMut<PathGuardCounts>,
    mut commands: Commands,
) {
    for (call, request, outcome) in &calls {
        let Ok(effect::Outcome::ToolResult { result }) = &outcome.0 else {
            continue;
        };
        if request.call.function.name.as_str() != SEARCH {
            continue;
        }
        let guarded = hide_protected(result.clone(), &policy);
        if guarded.hidden.is_empty() {
            continue;
        }
        for rule in guarded.hidden {
            *counts.hidden.entry(rule).or_default() += 1;
        }
        commands.entity(call).insert(Outcome::from(guarded.result));
    }
}

/// Drops the `path:line: text` lines of read-protected files from a
/// successful text result, and says how many it dropped.
fn hide_protected(result: ToolResult, policy: &PathPolicy) -> Guarded {
    let text = match result.output().as_text() {
        Some(text) if result.is_success() => text.to_owned(),
        _ => {
            return Guarded {
                result,
                hidden: Vec::new(),
            };
        }
    };
    let mut hidden = Vec::new();
    let mut kept: Vec<&str> = Vec::new();
    for line in text.lines() {
        match match_path(line).and_then(|path| policy.hides(path)) {
            Some(rule) => hidden.push(rule),
            None => kept.push(line),
        }
    }
    if hidden.is_empty() {
        return Guarded { result, hidden };
    }
    if kept.is_empty() {
        kept.push("No matches.");
    }
    let count = hidden.len();
    let noun = if count == 1 { "match" } else { "matches" };
    let note = format!("[{count} {noun} in protected files hidden]");
    let text = format!("{}\n{note}", kept.join("\n"));
    Guarded {
        result: result.with_output(Output::text(text)),
        hidden,
    }
}

/// The file of a `search` output line, `path:line: text`.
fn match_path(line: &str) -> Option<&str> {
    line.match_indices(':').find_map(|(at, _)| {
        let (path, rest) = line.split_at(at);
        let rest = rest.strip_prefix(':')?;
        let (number, text) = rest.split_once(':')?;
        let numeric = !number.is_empty() && number.bytes().all(|byte| byte.is_ascii_digit());
        (numeric && text.starts_with(' ')).then_some(path)
    })
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

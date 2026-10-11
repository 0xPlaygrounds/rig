//! Protected paths (corpus spec 02): the built-in file tools never read
//! secrets nor write outside the project or into `.git/`.
//!
//! The rules are `deny_read` and `deny_write` globs: defaults, then
//! `RIG_HOME/paths.toml`, then `<project>/.rig/paths.toml`. Every path a
//! call names is resolved first (`~`, `..` and symbolic links), so none
//! sneaks past. The kernel has no hook between a model's tool call and its
//! run, so the plugin replaces the built-in `read`, `edit`, `write` and
//! `search` tools (in `finish`, as the plugin guide shows) with open tools
//! of the same name, schema, rules and footprint, which check the call and
//! then run rig-tools' own tool. `/paths` lists the rules and what each
//! blocked.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use glob::{MatchOptions, Pattern};
use rig_core::tool::{IntoToolOutput, ToolOutput as Output};
use rig_harness::prelude::*;
use rig_tools::{Edit, Read, Search, Write};
use schemars::JsonSchema;
use serde::Deserialize;
use serde::de::DeserializeOwned;

#[cfg(test)]
mod tests;

/// The rules every project starts with.
const DEFAULT_READ: &[&str] = &[".env*", "**/*.pem", "~/.ssh/**"];
const DEFAULT_WRITE: &[&str] = &[".git/**"];

/// The arguments that name paths, in any tool's schema.
const PATH_ARGUMENTS: &[&str] = &["path", "file", "paths"];

/// Adds the path guard and `/paths`.
#[derive(Default)]
pub struct ProtectedPathsPlugin;

impl Plugin for ProtectedPathsPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<PathPolicy>() {
            let root = std::env::current_dir().unwrap_or_default();
            let home = Home::from_env().root().join("paths.toml");
            let project = root.join(".rig").join("paths.toml");
            app.insert_resource(PathPolicy::load(&root, &[home, project]));
        }
        app.init_resource::<PathGuardCounts>()
            .init_resource::<Unguarded>()
            .add_observer(finish_guarded)
            .add_command(
                "paths",
                "List the protected-path rules and what each blocked",
                list_rules,
            );
    }

    fn finish(&self, app: &mut App) {
        guard(app, Read, &[Access::Read], false);
        guard(app, Search, &[Access::Read], true);
        guard(app, Edit, &[Access::Read, Access::Write], false);
        guard(app, Write, &[Access::Write], false);
        // Any other tool naming a path cannot be guarded: its handler is
        // the kernel's alone.
        let world = app.world_mut();
        let names: Vec<String> = world
            .query::<&ToolDef>()
            .iter(world)
            .filter(|def| names_a_path(&def.0.parameters))
            .map(|def| def.0.name.as_str().to_owned())
            .filter(|name| !matches!(name.as_str(), "read" | "search" | "edit" | "write"))
            .collect();
        world.resource_mut::<Unguarded>().0 = names;
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

/// `paths.toml`.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct PathsFile {
    #[serde(default)]
    deny_read: Vec<String>,
    #[serde(default)]
    deny_write: Vec<String>,
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
        policy.add(Access::Read, DEFAULT_READ.iter().copied(), "default");
        policy.add(Access::Write, DEFAULT_WRITE.iter().copied(), "default");
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
                    policy.add(
                        Access::Read,
                        paths.deny_read.iter().map(String::as_str),
                        &source,
                    );
                    policy.add(
                        Access::Write,
                        paths.deny_write.iter().map(String::as_str),
                        &source,
                    );
                }
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

/// The tools that name a path but that the guard cannot check.
#[derive(Resource, Default)]
struct Unguarded(Vec<String>);

/// Whether a tool's parameters name a path argument.
fn names_a_path(parameters: &serde_json::Value) -> bool {
    parameters
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .is_some_and(|properties| {
            PATH_ARGUMENTS
                .iter()
                .any(|name| properties.contains_key(*name))
        })
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

/// A guarded call's result and the read rules that hid `search` matches.
struct Guarded {
    result: ToolResult,
    hidden: Vec<String>,
}

/// Replaces the registered tool named `T::NAME`, if any, with an open tool
/// of the same schema, description, rules and footprint that checks each
/// call's paths for `accesses`, then runs `tool`; `filter` drops output
/// lines (`path:line: text`) of read-protected files.
fn guard<T>(app: &mut App, tool: T, accesses: &'static [Access], filter: bool)
where
    T: PortableTool + 'static,
    T::Args: DeserializeOwned + JsonSchema,
{
    let world = app.world_mut();
    let mut tools = world.query::<(Entity, &ToolDef, &ToolRules, &Footprint)>();
    let Some((entity, description, rules, footprint)) = tools
        .iter(world)
        .find(|(_, def, ..)| def.0.name.as_str() == T::NAME)
        .map(|(entity, def, rules, footprint)| {
            (
                entity,
                def.0.description.clone(),
                rules.0.clone(),
                *footprint,
            )
        })
    else {
        return;
    };
    world.entity_mut(entity).insert(Disabled);
    let rules: Vec<&str> = rules.iter().map(String::as_str).collect();
    let options = ToolOptions {
        rules: &rules,
        footprint,
    };
    let tool = Arc::new(tool);
    let observer = move |called: On<ToolCalled<T::Args>>,
                         policy: Res<PathPolicy>,
                         mut counts: ResMut<PathGuardCounts>,
                         wake: Res<Wake>,
                         mut commands: Commands| {
        let arguments = &called.run.call.function.arguments;
        let denied = named_paths(arguments, filter).into_iter().find_map(|raw| {
            accesses
                .iter()
                .find_map(|&access| policy.check(access, &raw))
        });
        if let Some(denied) = denied {
            *counts.blocked.entry(denied.rule.label()).or_default() += 1;
            let refusal = ToolResult::failed(ToolExecutionError::refused(denied.message()));
            commands.entity(called.call).insert(ToolOutput(refusal));
            return;
        }
        let args = match T::Args::deserialize(arguments) {
            Ok(args) => args,
            Err(error) => {
                let why = ToolExecutionError::invalid_args(error.to_string());
                commands
                    .entity(called.call)
                    .insert(ToolOutput(ToolResult::failed(why)));
                return;
            }
        };
        let (tool, policy) = (tool.clone(), policy.clone());
        let work = async move {
            let result = match tool.call(args).await {
                Ok(output) => match output.into_tool_output() {
                    Ok(output) => ToolResult::success(output),
                    Err(error) => ToolResult::failed(error),
                },
                Err(error) => ToolResult::failed(tool.map_error(error)),
            };
            match filter {
                true => hide_protected(result, &policy),
                false => Guarded {
                    result,
                    hidden: Vec::new(),
                },
            }
        };
        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::default);
        commands
            .entity(called.call)
            .insert(Running::spawn(pool, &wake, work));
    };
    app.add_open_tool(T::NAME, &description, options, observer);
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

/// A guarded call's result becomes its `ToolOutput`, its hidden matches
/// counted.
fn finish_guarded(
    done: On<Add<Done<Guarded>>>,
    mut calls: Query<&mut Done<Guarded>>,
    mut counts: ResMut<PathGuardCounts>,
    mut commands: Commands,
) {
    let Ok(mut guarded) = calls.get_mut(done.entity) else {
        return;
    };
    for rule in guarded.0.hidden.drain(..) {
        *counts.hidden.entry(rule).or_default() += 1;
    }
    let result = guarded.0.result.clone();
    commands
        .entity(done.entity)
        .remove::<Done<Guarded>>()
        .insert(ToolOutput(result));
}

/// `/paths`: the rules, what each blocked, and what is not guarded.
fn list_rules(
    In(args): In<CommandArgs>,
    policy: Res<PathPolicy>,
    counts: Res<PathGuardCounts>,
    unguarded: Res<Unguarded>,
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
    lines.push("Guarded tools: read, edit, write, search.".to_owned());
    if !unguarded.0.is_empty() {
        lines.push(format!(
            "Not guarded, though they take a path: {}.",
            unguarded.0.join(", ")
        ));
    }
    lines.push(
        "`shell` commands are not parsed: a command can still read or write any path.".to_owned(),
    );
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

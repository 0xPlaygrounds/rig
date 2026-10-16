//! Path-scoped project rules (spec `09-glob-rules.md`), like Cursor rules.
//!
//! A rule is a Markdown file in `<project>/.rig/rules/` or, for the user's
//! own, `RIG_HOME/rules/` (a project rule replaces a personal one of the
//! same name), whose front matter says when it reaches the agent:
//!
//! ```markdown
//! ---
//! globs: ["src/db/**", "*.sql"]
//! always: false
//! description: Database conventions
//! ---
//! Never use SELECT *.
//! ```
//!
//! - `always: true`: the body is in every agent's system prompt.
//! - `globs`: the body is added to the result of an agent's first tool
//!   call that touches a matching path, as "Rules for `src/db/**`
//!   (db.md): …". The paths a call touched are what it published: the
//!   files it read (rig-core's `FilesRead`, a read or a search's hits) or
//!   changed (`FilesChanged`), and the directory and entries it listed
//!   (rig-tools' `FilesListed`, the `list` tool). Globs follow gitignore's
//!   rules from the agent's directory: one without `/` matches a name at
//!   any depth. Each rule reaches an agent once, again once its text
//!   changed, and again once the result that carried it is no longer in
//!   the agent's conversation as it was given (condensed into a summary,
//!   its output cleared, or cut).
//! - `description`: the rule is listed in the system prompt by name, and
//!   the `rule` tool returns its body; a glob rule loaded so counts as
//!   delivered.
//!
//! `/rules` lists the rules, where each comes from and which reached the
//! agent. Both rule directories are watched, so a rule added, changed or
//! removed applies once the watch sees it; a file whose front matter is
//! broken is skipped with a warning naming it. The Review system is in the
//! public [`GlobRulesSystems`] set.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use ignore::Match;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::Answered;
use rig_harness::rig_core::message::{CallId, Message, ToolResultContent, UserContent};
use rig_harness::rig_core::tool::{FilesChanged, FilesRead, ToolOutput};
use rig_harness::rig_tools::FilesListed;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// The tool that returns a rule's body.
pub const RULE: &str = "rule";

/// Delivers project rules: always-on ones in the system prompt, glob rules
/// with the tool results that touch their paths, the others through
/// `rule`; and `/rules`.
#[derive(Default)]
pub struct GlobRulesPlugin;

impl Plugin for GlobRulesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<RigHome>()
            .init_resource::<Rules>()
            .add_message::<FileChanges>()
            .register_required_components::<Agent, DeliveredRules>()
            .add_open_tool(
                RULE,
                "Return the project rule of this name, as listed under \"Project rules\". \
                 Before working on what a project rule's description covers, read it.",
                ToolOptions {
                    footprint: Some(Touches::Reads(Scope::Anything)),
                },
                look_up,
            )
            .add_command(
                "rules",
                "List the project rules and which reached this agent",
                list_rules,
            )
            .add_systems(Startup, watch_dirs)
            .add_systems(Update, refresh.before(AgentLoop))
            .configure_sets(AgentPass, GlobRulesSystems.in_set(ReviewSystems::Edit))
            .add_systems(AgentPass, deliver.in_set(GlobRulesSystems));
        let section = PromptSection::new(PromptSection::ORDER_PROJECT + 20, "rules", "");
        app.world_mut()
            .spawn((Name::new("prompt:rules"), RulesSection, section));
    }
}

/// The Review system that adds glob rules to tool results, for another
/// reviewer of tool outcomes to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobRulesSystems;

/// The glob rules given to this agent, by name. Saved with the session.
#[derive(Component, Reflect, Clone, Default, Debug, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct DeliveredRules(pub BTreeMap<String, Given>);

/// A rule as an agent was given it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Given {
    /// The body given.
    pub body: String,
    /// The tool call whose result carried it.
    pub call: CallId,
}

impl Given {
    /// Whether the agent's `conversation` still holds this rule: the
    /// call's result carries the body, or the call is there and its
    /// result is still to come. A summary takes the call away; cleared
    /// output keeps the call and loses the body.
    fn holds(&self, conversation: &Conversation) -> bool {
        let mut called = false;
        for message in conversation.messages() {
            match &**message {
                Message::Assistant(reply) => {
                    called |= reply.tool_calls().any(|call| call.id == self.call);
                }
                Message::User { content } => {
                    let result = content.iter().find_map(|item| match item {
                        UserContent::ToolResult(result) if result.call == self.call => Some(result),
                        _ => None,
                    });
                    if let Some(result) = result {
                        let mut text = result.content.iter().filter_map(ToolResultContent::as_text);
                        return text.any(|text| text.contains(&self.body));
                    }
                }
                Message::System { .. } => {}
            }
        }
        called
    }
}

impl DeliveredRules {
    /// Whether the agent of `conversation` holds the rule `name` as `body`.
    fn holds(&self, name: &str, body: &str, conversation: &Conversation) -> bool {
        let given = self.0.get(name);
        given.is_some_and(|given| given.body == body && given.holds(conversation))
    }
}

/// Where a rule comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    /// `RIG_HOME/rules`.
    Personal,
    /// `<project>/.rig/rules`.
    Project,
}

/// A watched rule directory.
#[derive(Component)]
struct RulesDir(Source);

/// The prompt section with the always-on rules and the list.
#[derive(Component)]
struct RulesSection;

/// On a call whose result was checked: once is enough.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Checked;

/// A rule file read.
struct Rule {
    source: Source,
    file: PathBuf,
    globs: Vec<String>,
    matcher: Gitignore,
    always: bool,
    description: Option<String>,
    body: String,
}

/// The rules found, by name, and the files skipped, with why.
#[derive(Resource, Default)]
struct Rules {
    found: BTreeMap<String, Rule>,
    broken: Vec<(PathBuf, String)>,
    /// The broken files warned about already.
    reported: HashSet<PathBuf>,
}

/// A rule file's front matter.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Front {
    #[serde(default)]
    globs: Vec<String>,
    #[serde(default)]
    always: bool,
    description: Option<String>,
}

impl Rule {
    /// Reads `file`, or says why it is skipped.
    fn read(file: &Path, source: Source) -> Result<Self, String> {
        let text = std::fs::read_to_string(file).map_err(|error| error.to_string())?;
        let rest = text
            .strip_prefix("---")
            .and_then(|rest| rest.split_once('\n'))
            .filter(|(after, _)| after.trim().is_empty())
            .map(|(_, rest)| rest)
            .ok_or("no front matter: the file must start with a `---` line")?;
        let (yaml, body) = format!("\n{rest}")
            .split_once("\n---")
            .map(|(yaml, after)| {
                let body = after.split_once('\n').map_or("", |(_, body)| body);
                (yaml.to_owned(), body.trim().to_owned())
            })
            .ok_or("the front matter has no closing `---` line")?;
        let front: Front =
            serde_yaml::from_str(&yaml).map_err(|error| format!("broken front matter: {error}"))?;
        if front.globs.is_empty() && !front.always && front.description.is_none() {
            return Err("the front matter says none of `globs`, `always` and `description`".into());
        }
        let mut matcher = GitignoreBuilder::new(".");
        for glob in &front.globs {
            matcher
                .add_line(None, glob)
                .map_err(|error| format!("bad glob `{glob}`: {error}"))?;
        }
        let matcher = matcher.build().map_err(|error| error.to_string())?;
        Ok(Self {
            source,
            file: file.to_path_buf(),
            globs: front.globs,
            matcher,
            always: front.always,
            description: front.description,
            body,
        })
    }

    /// The glob that matches `path` (relative to the agent's directory),
    /// or a directory it is in.
    fn glob_for(&self, path: &Path, is_dir: bool) -> Option<&str> {
        let mut is_dir = is_dir;
        for at in path.ancestors().filter(|at| !at.as_os_str().is_empty()) {
            if let Match::Ignore(glob) = self.matcher.matched(at, is_dir) {
                return Some(glob.original());
            }
            is_dir = true;
        }
        None
    }

    /// The rule file's name.
    fn file_name(&self) -> String {
        self.file
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}

impl Rules {
    /// Reads the rule files of `dirs`, most general first.
    fn scan<'a>(&mut self, dirs: impl IntoIterator<Item = (Source, &'a Path)>) {
        self.found.clear();
        self.broken.clear();
        for (source, dir) in dirs {
            let Ok(entries) = std::fs::read_dir(dir) else {
                continue;
            };
            let mut files: Vec<PathBuf> = entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|file| file.is_file() && file.extension().is_some_and(|ext| ext == "md"))
                .collect();
            files.sort();
            for file in files {
                let Some(name) = file.file_stem().map(|stem| stem.to_string_lossy()) else {
                    continue;
                };
                let name = name.into_owned();
                match Rule::read(&file, source) {
                    Ok(rule) => {
                        self.found.insert(name, rule);
                    }
                    Err(why) => self.broken.push((file, why)),
                }
            }
        }
    }

    /// The prompt section: the always-on rules, then the rules to look up.
    fn section(&self) -> String {
        let mut parts = Vec::new();
        let always: Vec<&str> = self
            .found
            .values()
            .filter(|rule| rule.always && !rule.body.is_empty())
            .map(|rule| rule.body.as_str())
            .collect();
        if !always.is_empty() {
            parts.push(format!("Project rules:\n{}", always.join("\n\n")));
        }
        let listed: Vec<String> = self
            .found
            .iter()
            .filter(|(_, rule)| !rule.always)
            .map(|(name, rule)| {
                let globs = (!rule.globs.is_empty())
                    .then(|| format!(" (given with files matching {})", rule.globs.join(", ")));
                let description = rule.description.as_deref().map(|text| format!(": {text}"));
                format!(
                    "- {name}{}{}",
                    description.unwrap_or_default(),
                    globs.unwrap_or_default()
                )
            })
            .collect();
        if !listed.is_empty() {
            parts.push(format!(
                "Project rules you can read with `{RULE}`:\n{}",
                listed.join("\n")
            ));
        }
        parts.join("\n\n")
    }
}

/// Watches `RIG_HOME/rules` and the project's `.rig/rules`.
fn watch_dirs(home: Res<RigHome>, project: Res<ProjectRoot>, mut commands: Commands) {
    let dirs = [
        (Source::Personal, home.0.root().join("rules")),
        (Source::Project, project.0.join(".rig").join("rules")),
    ];
    for (source, dir) in dirs {
        commands.spawn((RulesDir(source), FileWatch::new(dir)));
    }
}

/// Reads the rules when the rule directories are first watched, so the
/// first request has them, and again when one changes; keeps the prompt
/// section current, and warns once of each newly broken file.
fn refresh(
    dirs: Query<(&RulesDir, Ref<FileWatch>)>,
    mut changes: MessageReader<FileChanges>,
    mut rules: ResMut<Rules>,
    mut sections: Query<&mut PromptSection, With<RulesSection>>,
    mut notices: MessageWriter<Notice>,
) {
    let changed = changes.read().any(|change| dirs.contains(change.watch));
    if !changed && !dirs.iter().any(|(_, watch)| watch.is_added()) {
        return;
    }
    let mut ordered: Vec<(Source, &Path)> = dirs
        .iter()
        .map(|(dir, watch)| (dir.0, watch.into_inner().path.as_path()))
        .collect();
    ordered.sort_by_key(|(source, _)| *source == Source::Project);
    rules.scan(ordered);
    let text = rules.section();
    for mut section in &mut sections {
        if section.text != text {
            section.text.clone_from(&text);
        }
    }
    let Rules {
        broken, reported, ..
    } = &mut *rules;
    reported.retain(|file| broken.iter().any(|(still, _)| still == file));
    for (file, why) in broken.iter() {
        if reported.insert(file.clone()) {
            let text = format!("Rule file {} skipped: {why}.", file.display());
            notices.write(Notice::warn(None, text));
        }
    }
}

/// Adds to each tool result, before it is committed, the glob rules of the
/// paths the call touched that its agent does not hold: not given yet, its
/// text changed since, or no longer in its conversation.
fn deliver(
    mut calls: Query<
        (Entity, &ForAgent, &ToolRequest, &Published, &mut Outcome),
        (ReadyToSettle, ToolCalls, Without<Checked>),
    >,
    mut agents: Query<(&mut DeliveredRules, &Conversation, Option<&WorkDir>)>,
    rules: Res<Rules>,
    root: Res<ProjectRoot>,
    mut commands: Commands,
) {
    for (call, &ForAgent(agent), request, published, mut outcome) in &mut calls {
        commands.entity(call).insert(Checked);
        let Ok(Answered::ToolResult { result }) = &outcome.0 else {
            continue;
        };
        let Ok((mut delivered, conversation, dir)) = agents.get_mut(agent) else {
            continue;
        };
        if result.error().is_some() || rules.found.is_empty() {
            continue;
        }
        let touched = touched(published, dir.map_or(&root.0, |dir| &dir.0));
        let mut notes = Vec::new();
        for (name, rule) in &rules.found {
            let glob = touched
                .iter()
                .find_map(|(path, is_dir)| rule.glob_for(path, *is_dir));
            let Some(glob) = glob else {
                continue;
            };
            if delivered.holds(name, &rule.body, conversation) {
                continue;
            }
            let changed = delivered
                .0
                .get(name)
                .filter(|given| given.body != rule.body)
                .map_or("", |_| ", changed");
            let file = rule.file_name();
            notes.push(format!(
                "Rules for `{glob}` ({file}{changed}):\n{}",
                rule.body
            ));
            let given = Given {
                body: rule.body.clone(),
                call: request.call.id.clone(),
            };
            delivered.0.insert(name.clone(), given);
        }
        if notes.is_empty() {
            continue;
        }
        // After the result's text, so the rules end it.
        let notes = notes.join("\n\n");
        let mut content = result.output().as_content().to_vec();
        match content.last_mut() {
            Some(ToolResultContent::Text(text)) => {
                text.text = format!("{}\n\n{notes}", text.text.trim_end());
            }
            _ => content.push(ToolResultContent::text(notes)),
        }
        if let Ok(output) = ToolOutput::content(content) {
            *outcome = Outcome::from(result.clone().with_output(output));
        }
    }
}

/// The paths a call published it touched, relative to `base`, each with
/// whether it is a directory: the files it read or changed, and the
/// directory it listed with the entries it showed.
fn touched(published: &Published, base: &Path) -> Vec<(PathBuf, bool)> {
    let facts = &published.0;
    let mut paths: Vec<(PathBuf, bool)> = Vec::new();
    if let Ok(Some(read)) = facts.get::<FilesRead>() {
        paths.extend(read.paths().map(|path| (PathBuf::from(path), false)));
    }
    if let Ok(Some(changed)) = facts.get::<FilesChanged>() {
        paths.extend(changed.paths().map(|path| (PathBuf::from(path), false)));
    }
    if let Ok(Some(listed)) = facts.get::<FilesListed>() {
        let root = PathBuf::from(&listed.root);
        let entries = listed
            .entries
            .iter()
            .map(|entry| match entry.strip_suffix('/') {
                Some(dir) => (root.join(dir), true),
                None => (root.join(entry), false),
            });
        paths.extend(entries);
        paths.push((root, true));
    }
    let canonical = base.canonicalize().ok();
    let relative = |path: PathBuf| -> Option<PathBuf> {
        if path.is_relative() {
            return Some(path);
        }
        let below = path
            .strip_prefix(base)
            .ok()
            .or_else(|| path.strip_prefix(canonical.as_deref()?).ok())?;
        Some(below.to_path_buf())
    };
    paths
        .into_iter()
        .filter_map(|(path, is_dir)| Some((relative(path)?, is_dir)))
        .collect()
}

/// `rule`'s arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RuleArgs {
    /// The rule's name, as listed.
    pub name: String,
}

/// Answers a `rule` call with the rule's body; a glob rule read so counts
/// as delivered to the agent.
fn look_up(
    called: On<ToolCalled<RuleArgs>>,
    rules: Res<Rules>,
    mut agents: Query<&mut DeliveredRules>,
    mut decisions: Query<&mut Decision, ToolCalls>,
) {
    let name = called.args.name.trim();
    let result = match rules.found.get(name) {
        Some(rule) => {
            if !rule.globs.is_empty()
                && let Ok(mut delivered) = agents.get_mut(called.agent)
            {
                let given = Given {
                    body: rule.body.clone(),
                    call: called.tool_call.id.clone(),
                };
                delivered.0.insert(name.to_owned(), given);
            }
            ToolResult::success(rule.body.clone().into())
        }
        None => {
            let names: Vec<&str> = rules.found.keys().map(String::as_str).collect();
            let why = format!("No rule `{name}`. The rules: {}.", names.join(", "));
            ToolResult::failed(ToolExecutionError::other(why))
        }
    };
    if let Ok(mut decision) = decisions.get_mut(called.call) {
        decision.decide(Decider::plugin("glob-rules"), result);
    }
}

/// `/rules`: each rule, where it comes from, when it applies, and whether
/// it reached this agent; then the files skipped.
fn list_rules(
    In(args): In<CommandArgs>,
    rules: Res<Rules>,
    agents: Query<(&DeliveredRules, &Conversation)>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = agents.get(args.agent).ok();
    let given = |name: &str, body: &str| {
        agent.is_some_and(|(delivered, conversation)| delivered.holds(name, body, conversation))
    };
    let mut lines = vec![format!(
        "Rules ({}; * reached this agent):",
        rules.found.len()
    )];
    for (name, rule) in &rules.found {
        let mark = if rule.always || given(name, &rule.body) {
            "*"
        } else {
            " "
        };
        let source = match rule.source {
            Source::Personal => "personal",
            Source::Project => "project",
        };
        let when = if rule.always {
            "always".to_owned()
        } else if rule.globs.is_empty() {
            format!("on request: {}", rule.description.as_deref().unwrap_or(""))
        } else {
            format!("for {}", rule.globs.join(", "))
        };
        let file = rule.file.display();
        lines.push(format!("{mark} {name} ({source}, {file}): {when}"));
    }
    for (file, why) in &rules.broken {
        lines.push(format!("  skipped {}: {why}", file.display()));
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

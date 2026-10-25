//! SP-01: 01. Environment policy for agent commands (splits fresh held-out
//! spec `01-shell-env-policy.md`).
//!
//! [`ShellEnvPolicyPlugin`] builds the environment of every process the
//! agents start from a policy: `~/.config/rig/env-policy.toml` with the
//! project's `.rig/env-policy.toml` merged over it ([`EnvPolicy`]). Every
//! process gets the variables `allow` names (always with [`DEFAULT_ALLOW`])
//! that no `deny` glob matches, and the `set` pairs; every other variable
//! of this process is absent. The policy is read again as each tool call
//! comes in and set on the app's [`Processes`]; an agent with an
//! `[agent.<name>]` section gets a `shell` of its own with its extras. A
//! `shell` command that names a denied variable is refused before it
//! starts, a malformed file keeps the last good policy with one warning,
//! and `/env` lists what the focused agent's commands see.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};

use bevy_ecs::system::SystemParam;
use glob::Pattern;
use rig_basics::coding_tools::ShellOutput;
use rig_harness::prelude::*;
use rig_harness::rig_core::serve::ErasedHandler;
use rig_harness::rig_core::serve::adapters::ToolAdapter;
use rig_harness::rig_ecs::effects::Handler;
use rig_harness::rig_tools::Shell;
use rig_harness::rig_tools::process::Launcher;
use serde::Deserialize;

/// The project's policy file, from the [`ProjectRoot`].
pub const PROJECT_FILE: &str = ".rig/env-policy.toml";

/// The variables every command may see unless denied.
pub const DEFAULT_ALLOW: &[&str] = &[
    "PATH", "HOME", "USER", "SHELL", "LANG", "LC_*", "TERM", "TMPDIR",
];

/// Who decides the calls this plugin refuses.
const DECIDER: &str = "env-policy";

/// The policy for every agent's processes, and `/env`.
#[derive(Default)]
pub struct ShellEnvPolicyPlugin;

/// The plugin's stage systems, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnvPolicySystems {
    /// In [`PrepareSystems::Edit`]: reads the policy again and gives a
    /// `shell` call of an agent with a section of its own its handler.
    Prepare,
    /// In [`Stage::Gate`]: refuses a `shell` command naming a denied
    /// variable.
    Gate,
}

impl Plugin for ShellEnvPolicyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<Processes>()
            .init_resource::<UserPolicyFile>()
            .init_resource::<EnvPolicy>()
            .configure_sets(
                AgentPass,
                (
                    EnvPolicySystems::Prepare.in_set(PrepareSystems::Edit),
                    EnvPolicySystems::Gate.in_set(Stage::Gate),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    prepare.in_set(EnvPolicySystems::Prepare),
                    refuse_denied.in_set(EnvPolicySystems::Gate),
                ),
            )
            .add_systems(Startup, apply_at_start)
            .add_command(
                "env",
                "List the variables the focused agent's commands see",
                env_command,
            );
    }

    /// Keeps the launcher the other plugins set up, which the policy
    /// narrows.
    fn finish(&self, app: &mut App) {
        let base = app
            .world_mut()
            .get_resource_or_init::<Processes>()
            .launcher();
        app.insert_resource(BaseLauncher(base));
    }
}

/// Where the user's policy is read: `~/.config/rig/env-policy.toml`
/// unless set; `None` reads none. Insert one before adding the plugin to
/// change it.
#[derive(Resource, Clone, Debug)]
pub struct UserPolicyFile(pub Option<PathBuf>);

impl Default for UserPolicyFile {
    fn default() -> Self {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        Self(home.map(|home| home.join(".config/rig/env-policy.toml")))
    }
}

/// The launcher the app had once every plugin was built, before the
/// policy narrowed it: its wrapper, its directory and the variables it
/// already keeps from every command, secrets included.
#[derive(Resource, Clone, Debug)]
struct BaseLauncher(Launcher);

/// A policy file as written.
#[derive(Deserialize, Default, Debug)]
#[serde(default, deny_unknown_fields)]
struct PolicyTable {
    allow: Vec<String>,
    deny: Vec<String>,
    set: BTreeMap<String, String>,
    agent: BTreeMap<String, AgentTable>,
}

/// An `[agent.<name>]` section as written.
#[derive(Deserialize, Default, Debug)]
#[serde(default, deny_unknown_fields)]
struct AgentTable {
    allow: Vec<String>,
    set: BTreeMap<String, String>,
}

/// A policy file read and its globs compiled.
#[derive(Clone, Debug, Default)]
struct Policy {
    allow: Vec<Pattern>,
    deny: Vec<Pattern>,
    set: BTreeMap<String, String>,
    agent: BTreeMap<String, (Vec<Pattern>, BTreeMap<String, String>)>,
}

impl Policy {
    /// The policy `text` says, or why it says none.
    fn parse(text: &str) -> Result<Self, String> {
        let table: PolicyTable = toml::from_str(text).map_err(|error| error.to_string())?;
        let mut agent = BTreeMap::new();
        for (name, section) in table.agent {
            agent.insert(name, (globs(&section.allow)?, section.set));
        }
        Ok(Self {
            allow: globs(&table.allow)?,
            deny: globs(&table.deny)?,
            set: table.set,
            agent,
        })
    }
}

/// `patterns` compiled, or the first that is no glob.
fn globs(patterns: &[String]) -> Result<Vec<Pattern>, String> {
    patterns
        .iter()
        .map(|pattern| Pattern::new(pattern).map_err(|error| format!("`{pattern}`: {error}")))
        .collect()
}

/// What reading a policy file gave.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum Read {
    /// No file there.
    #[default]
    Missing,
    /// Its text.
    Text(String),
    /// Why it could not be read.
    Failed(String),
}

/// One policy file: where it is, how it is named to the user, what it
/// held when last read, and its last good policy.
#[derive(Debug, Default)]
struct Source {
    path: PathBuf,
    shown: String,
    read: Read,
    good: Option<Policy>,
}

impl Source {
    /// Reads the file again; whether the policy it gives changed. A file
    /// that does not read as a policy keeps the last good one, with one
    /// warning for each version of it.
    fn reload(&mut self, path: &Path, shown: &str, notices: &mut MessageWriter<Notice>) -> bool {
        if self.path != path {
            *self = Self {
                path: path.to_owned(),
                shown: shown.to_owned(),
                ..Self::default()
            };
        }
        let read = match std::fs::read_to_string(path) {
            Ok(text) => Read::Text(text),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Read::Missing,
            Err(error) => Read::Failed(error.to_string()),
        };
        if read == self.read {
            return false;
        }
        self.read = read.clone();
        let why = match read {
            Read::Missing => {
                self.good = None;
                return true;
            }
            Read::Text(text) => match Policy::parse(&text) {
                Ok(policy) => {
                    self.good = Some(policy);
                    return true;
                }
                Err(why) => why,
            },
            Read::Failed(why) => why,
        };
        let text = format!(
            "{} is not a valid environment policy; the last good one stays: {}",
            self.shown,
            why.lines().next().unwrap_or_default()
        );
        notices.write(Notice::warn(None, text));
        false
    }
}

/// The policy files as last read: the user's, then the project's, merged
/// over it.
#[derive(Resource, Debug, Default)]
pub struct EnvPolicy {
    user: Source,
    project: Source,
}

impl EnvPolicy {
    /// Whether some file has a section for the agent `name`.
    fn has_agent(&self, name: &str) -> bool {
        self.policies()
            .any(|(policy, _)| policy.agent.contains_key(name))
    }

    /// Each good policy, the user's first, with how its file is named.
    fn policies(&self) -> impl Iterator<Item = (&Policy, &str)> {
        [&self.user, &self.project]
            .into_iter()
            .filter_map(|source| Some((source.good.as_ref()?, source.shown.as_str())))
    }

    /// The policy for the agent `agent`, or for every agent with `None`.
    fn effective(&self, agent: Option<&str>) -> Effective {
        let mut effective = Effective {
            allow: DEFAULT_ALLOW
                .iter()
                .filter_map(|it| Pattern::new(it).ok())
                .collect(),
            ..Effective::default()
        };
        for (policy, shown) in self.policies() {
            effective.allow.extend(policy.allow.iter().cloned());
            let denied = policy
                .deny
                .iter()
                .map(|deny| (deny.clone(), shown.to_owned()));
            effective.deny.extend(denied);
            effective.set.extend(policy.set.clone());
        }
        let sections = self
            .policies()
            .filter_map(|(policy, _)| policy.agent.get(agent?));
        for (allow, set) in sections {
            effective.allow.extend(allow.iter().cloned());
            effective.set.extend(set.clone());
        }
        effective
    }
}

/// One agent's policy, every file merged.
#[derive(Debug, Default)]
struct Effective {
    allow: Vec<Pattern>,
    /// Each deny glob, with the file that has it.
    deny: Vec<(Pattern, String)>,
    set: BTreeMap<String, String>,
}

impl Effective {
    /// The file that denies `name`, if one does.
    fn denied_by(&self, name: &str) -> Option<&str> {
        let deny = self.deny.iter().find(|(deny, _)| deny.matches(name));
        deny.map(|(_, file)| file.as_str())
    }

    /// Whether a command may inherit `name`: allowed, not denied and not
    /// kept from every command already (a secret).
    fn keeps(&self, name: &OsString, base: &Launcher) -> bool {
        let allowed = |name: &str| self.allow.iter().any(|allow| allow.matches(name));
        !base.unset_env.contains(name)
            && name
                .to_str()
                .is_some_and(|name| allowed(name) && self.denied_by(name).is_none())
    }

    /// `base`, keeping from every command each variable of this process
    /// the policy does not keep, and setting its `set` pairs.
    fn launcher(&self, base: &Launcher) -> Launcher {
        let dropped = std::env::vars_os()
            .map(|(name, _)| name)
            .filter(|name| !self.keeps(name, base));
        let launcher = base.clone().without_env(dropped.collect::<Vec<_>>());
        self.set.iter().fold(launcher, |launcher, (name, value)| {
            launcher.with_env(name, value)
        })
    }

    /// The app's `current` launcher for an agent with a section of its
    /// own: what its section allows inherited again, and its `set` pairs.
    fn for_agent(&self, mut current: Launcher, base: &Launcher) -> Launcher {
        current.unset_env.retain(|name| !self.keeps(name, base));
        self.set.iter().fold(current, |launcher, (name, value)| {
            launcher.with_env(name, value)
        })
    }
}

/// The policy files, read again on use, and the launcher they give.
#[derive(SystemParam)]
struct PolicyFiles<'w> {
    policy: ResMut<'w, EnvPolicy>,
    user: Res<'w, UserPolicyFile>,
    root: Res<'w, ProjectRoot>,
    processes: Res<'w, Processes>,
    base: Option<Res<'w, BaseLauncher>>,
    notices: MessageWriter<'w, Notice>,
}

impl PolicyFiles<'_> {
    /// Reads the files again and, when the policy changed or `always`,
    /// starts every process from now on as it says.
    fn refresh(&mut self, always: bool) {
        let mut changed = match &self.user.0 {
            Some(path) => {
                let shown = path.display().to_string();
                self.policy.user.reload(path, &shown, &mut self.notices)
            }
            None => false,
        };
        let project = self.root.0.join(PROJECT_FILE);
        changed |= self
            .policy
            .project
            .reload(&project, PROJECT_FILE, &mut self.notices);
        if let Some(base) = &self.base
            && (changed || always)
        {
            self.processes
                .set(self.policy.effective(None).launcher(&base.0));
        }
    }

    /// How the agent `name`'s processes start: the app's launcher, with
    /// its section's extras when it has one.
    fn launcher_for(&self, name: Option<&str>) -> Launcher {
        let current = self.processes.launcher();
        match (name, &self.base) {
            (Some(name), Some(base)) if self.policy.has_agent(name) => self
                .policy
                .effective(Some(name))
                .for_agent(current, &base.0),
            _ => current,
        }
    }
}

/// The name an `[agent.<name>]` section matches: the agent's title, such as
/// a task's, else its `Name`.
fn agent_name<'a>(title: Option<&'a AgentTitle>, name: Option<&'a Name>) -> Option<&'a str> {
    title
        .map(|title| title.0.as_str())
        .or(name.map(Name::as_str))
}

/// Sets the policy before the first process starts.
fn apply_at_start(mut files: PolicyFiles) {
    files.refresh(true);
}

/// What an agent's own `shell` is built from, as the default `shell`
/// builds each call.
#[derive(SystemParam)]
struct ShellParts<'w, 's> {
    workspace: Res<'w, ProjectWorkspace>,
    dirs: Query<'w, 's, &'static WorkDir>,
    scratch: Option<Res<'w, Scratch>>,
    output: Option<Res<'w, ShellOutput>>,
    wake: Res<'w, Wake>,
}

/// Reads the policy again as tool calls come in, and gives each `shell`
/// call of an agent with a section of its own a shell started as that
/// section says.
fn prepare(
    mut calls: Query<
        (Entity, &ToolRequest, Option<&ForAgent>, &mut CallHandler),
        (Added<Admitted>, ToolCalls),
    >,
    agents: Query<(Option<&AgentTitle>, Option<&Name>)>,
    mut files: PolicyFiles,
    parts: ShellParts,
    mut commands: Commands,
) {
    if calls.is_empty() {
        return;
    }
    files.refresh(false);
    for (call, request, agent, mut handler) in &mut calls {
        let Some(&ForAgent(agent)) = agent else {
            continue;
        };
        let name = agents
            .get(agent)
            .ok()
            .and_then(|(title, name)| agent_name(title, name));
        let has_section = name.is_some_and(|name| files.policy.has_agent(name));
        if request.call.function.name.as_str() != Shell::NAME || !has_section {
            continue;
        }
        let launcher = files.launcher_for(name);
        let (output, live) = CallOutput::new(&parts.wake);
        commands.entity(call).insert(output);
        let (dir, confine) = match parts.dirs.get(agent) {
            Ok(dir) => (dir.0.clone(), true),
            Err(_) => (files.root.0.clone(), false),
        };
        let launcher = if confine || launcher.cwd.is_none() {
            launcher.in_dir(dir)
        } else {
            launcher
        };
        let shell = Shell {
            workspace: parts.workspace.0.clone(),
            launcher,
            live: Some(live),
            spill: parts.scratch.as_ref().and_then(|scratch| scratch.spill()),
            cut: parts
                .output
                .as_ref()
                .map_or(Shell::default().cut, |output| output.cut()),
            confine,
        };
        handler.0 = Handler(ErasedHandler::new(ToolAdapter::new(shell)));
    }
}

/// Refuses each `shell` command that names a variable a policy denies.
fn refuse_denied(
    mut calls: Query<(&ToolRequest, &mut Decision), (ReadyToStart, ToolCalls)>,
    policy: Res<EnvPolicy>,
) {
    let effective = policy.effective(None);
    for (request, mut decision) in &mut calls {
        if request.call.function.name.as_str() != Shell::NAME {
            continue;
        }
        let arguments = &request.call.function.arguments;
        let command = arguments.get("command").and_then(serde_json::Value::as_str);
        let names = referenced(command.unwrap_or_default());
        let denied = names
            .iter()
            .find_map(|name| Some((name, effective.denied_by(name)?)));
        if let Some((name, file)) = denied {
            let why = format!("`{name}` is denied by {file}; the command was not run.");
            decision.decide(Decider::plugin(DECIDER), Outcome::refused(why));
        }
    }
}

/// The variables `command` reads: `$NAME`, `${NAME}`, `${NAME:-x}` and
/// the like, and `printenv NAME`.
fn referenced(command: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut chars = command.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '$' {
            continue;
        }
        if chars.next_if_eq(&'{').is_some() {
            while chars.next_if(|c| matches!(c, '#' | '!')).is_some() {}
        }
        let mut name = String::new();
        while let Some(c) = chars.next_if(|c| c.is_ascii_alphanumeric() || *c == '_') {
            name.push(c);
        }
        if is_name(&name) {
            names.push(name);
        }
    }
    for part in command.split([';', '&', '|', '\n', '(', ')', '`']) {
        let mut words = part
            .split_whitespace()
            .skip_while(|word| *word != "printenv");
        if words.next().is_some() {
            let words = words.map(|word| word.trim_matches(['"', '\'']));
            names.extend(words.filter(|word| is_name(word)).map(str::to_owned));
        }
    }
    names
}

/// Whether `word` is a variable's name.
fn is_name(word: &str) -> bool {
    let mut chars = word.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `/env`: the variables the focused agent's commands see, the values of
/// those a policy sets.
fn env_command(
    In(args): In<CommandArgs>,
    agents: Query<(Option<&AgentTitle>, Option<&Name>)>,
    mut files: PolicyFiles,
) {
    files.refresh(false);
    let name = agents
        .get(args.agent)
        .ok()
        .and_then(|(title, name)| agent_name(title, name));
    let launcher = files.launcher_for(name);
    let set = files.policy.effective(name).set;
    let inherited = std::env::vars_os().map(|(name, _)| name);
    let kept = inherited.filter(|name| !launcher.unset_env.contains(name));
    let set_by_launcher = launcher.env.iter().map(|(name, _)| name.clone());
    let names: BTreeSet<String> = kept
        .chain(set_by_launcher)
        .map(|name| name.to_string_lossy().into_owned())
        .collect();
    let lines = names.iter().map(|name| match set.get(name) {
        Some(value) => format!("  {name}={value}"),
        None => format!("  {name}"),
    });
    let who = name.unwrap_or("the agent");
    let head = format!("What `{who}`'s commands see (values of `set` shown):");
    let text = std::iter::once(head)
        .chain(lines)
        .collect::<Vec<_>>()
        .join("\n");
    files.notices.write(Notice::info(args.agent, text));
}

#[cfg(test)]
mod tests;

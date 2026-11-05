//! `settings.toml`, the one file a user edits, as both the launcher and the
//! agent read it. Its `[plugins]` table holds only the user's changes to
//! rig's built-in plugins ([`BUILT_IN`]): `enable` and `disable` name
//! plugins, and `[[plugins.extra]]` tables add crates that are not rig's.
//! The agent is built with exactly the plugins on. Every edit rig makes
//! ([`edit`]) keeps the user's comments and layout.
//!
//! ```
//! use rig_harness_protocol::config::{SettingsFile, plugins};
//!
//! let home = SettingsFile::parse(
//!     "settings.toml".into(),
//!     "[plugins]\nenable = [\"rig_basics::budget::BudgetPlugin\"]\n",
//! )?;
//! let chosen = plugins(&home)?;
//! assert!(chosen.on.iter().any(|plugin| plugin == "rig_basics::budget::BudgetPlugin"));
//! # Ok::<(), rig_harness_protocol::config::Error>(())
//! ```

use std::collections::BTreeSet;
use std::fmt;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

pub use toml_edit;
use toml_edit::{Array, DocumentMut, Item, TableLike, Value};

/// One of rig's own plugins, as
/// [`built_in_plugins!`](crate::built_in_plugins) lists it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuiltIn {
    /// Its type's path, such as `rig_tui::TuiPlugin`, which names it.
    pub plugin: &'static str,
    /// The crate it is in.
    pub krate: &'static str,
    /// Whether it runs unless the user's settings turn it off.
    pub on: bool,
    /// The `target_os` values it builds on; empty for every target.
    pub targets: &'static [&'static str],
    /// What goes when it is left out, for a plugin whose absence takes away
    /// the way back from inside the agent.
    pub essential: Option<&'static str>,
    /// What it adds, which `rig plugin list` and `/plugins` show.
    pub about: &'static str,
}

impl BuiltIn {
    /// Whether it builds on the target this was compiled for.
    pub fn available(&self) -> bool {
        self.targets.is_empty() || self.targets.contains(&std::env::consts::OS)
    }
}

/// Calls the macro `$then` with every one of rig's own plugins, in the
/// order the agent adds them: whether it is on by default, its type, its
/// crate, the `target_os` values it builds on (none for every one), what
/// goes without it when that takes away the way back, and what it adds.
/// [`BUILT_IN`] is made from it, and so is a test's list of the defaults,
/// so there is one list.
#[macro_export]
macro_rules! built_in_plugins {
    ($then:ident) => {
        $then! {
            on rig_basics::prompt::DefaultPromptPlugin, "rig-basics", [], None,
                "rig's own system prompt: one fixed text, so the prompt cache keeps it.";
            on rig_basics::agents_md::AgentsMdPlugin, "rig-basics", [], None,
                "The AGENTS.md files of the working directory and the directories above it up to \
the repository root, and RIG_HOME/AGENTS.md, in the system prompt.";
            on rig_basics::skills::SkillsPlugin, "rig-basics", [], None,
                "Skills: each SKILL.md in .rig/skills and .agents/skills (up to the repository \
root), RIG_HOME/skills and ~/.agents/skills, listed in the system prompt for the \
model to read, and /skill:<name>.";
            on rig_basics::models::ModelsPlugin, "rig-basics", [], None,
                "/model, /effort and the model picker.";
            on rig_basics::login::LoginPlugin, "rig-basics", [], None,
                "/login and /logout: the ChatGPT plan, and providers' API keys.";
            on rig_basics::models::defaults::DefaultsPlugin, "rig-basics", [], None,
                "The model and reasoning a new session starts with: [model] in settings.toml, \
which the last choice rewrites.";
            on rig_basics::sessions::SessionsPlugin, "rig-basics", [], None,
                "/new, /resume and /name.";
            on rig_basics::sessions::agents::AgentsPlugin, "rig-basics", [], None,
                "/fork, /agents and rewind (Alt+Up to an earlier message of yours, then Enter): \
the agents of this session.";
            on rig_basics::compaction::CompactionPlugin, "rig-basics", [], None,
                "Summarizing a conversation near the context window, and /compact.";
            on rig_basics::telemetry::usage::UsagePlugin, "rig-basics", [], None,
                "What model calls cost, and /usage.";
            on rig_traffic::TrafficPlugin, "rig-traffic", [], None,
                "With --traffic-log or [traffic] log = true, provider HTTP in traffic.jsonl.";
            on rig_basics::activity::ActivityPlugin, "rig-basics", [], None,
                "What each agent is doing, for views.";
            on rig_basics::coding_tools::ReadTool, "rig-basics", [], None,
                "The read tool.";
            on rig_basics::coding_tools::EditTool, "rig-basics", [], None,
                "The edit tool.";
            on rig_basics::coding_tools::WriteTool, "rig-basics", [], None,
                "The write tool.";
            on rig_basics::coding_tools::ShellTool, "rig-basics", [], None,
                "The shell tool.";
            on rig_basics::coding_tools::AttachPlugin, "rig-basics", [], None,
                "Files named as @path in what the user types go with the message.";
            off rig_basics::redact::RedactPlugin, "rig-basics", [], None,
                "API keys, tokens and private keys become placeholders (\u{2039}SECRET_1\u{203a}) \
where they come in, so the model, logs and views never see them; a tool call \
naming one runs with the real value. Without it nothing is redacted.";
            on rig_basics::coding_tools::remote::RemoteHostsPlugin, "rig-basics", [], None,
                "Paths written HOST:PATH reach the hosts [remote.hosts] in settings.toml names \
(ssh, koh exec, ...); nothing without hosts.";
            off rig_basics::approval::ApprovalPlugin, "rig-basics", [], None,
                "One question for every effect a plugin wants approved; nothing asks by default. \
SubagentsPlugin adds it.";
            off rig_basics::preview::PreviewPlugin, "rig-basics", [], None,
                "What a tool call would change on disk, before it runs, for gates that ask \
(Preview).";
            on rig_basics::reload::ReloadPlugin, "rig-basics", [],
                Some("/reload, so a change applies only when rig starts again"),
                "/reload, which rebuilds the agent when the plugins settings.toml turns on or a \
plugin crate changed, and restarts it. Without it the agent runs as it started.";
            on rig_basics::reload::plugins::PluginsPlugin, "rig-basics", [],
                Some("/plugins itself"),
                "/plugins: every plugin, on and off, to turn on and off; applying rebuilds the \
agent and restarts it.";
            on rig_basics::BasicCommandsPlugin, "rig-basics", [], None,
                "/help and /quit.";
            on rig_basics::welcome::WelcomePlugin, "rig-basics", [], None,
                "The welcome at the top of the transcript: rig's mark, its version, the model and \
the project's directory. Never sent to the model.";
            off rig_inspect::InspectPlugin, "rig-inspect", [], None,
                "The inspect tool, with which the agent reads its own running world through Bevy \
Remote.";
            on rig_basics::print::PrintPlugin, "rig-basics", [], None,
                "-p and --json: answers one prompt, or what is piped in, and drives the agent with \
JSON lines on stdin and stdout; the front of every run that is not on an \
interactive terminal.";
            on rig_tui::TuiPlugin, "rig-tui", [],
                Some("the terminal view, leaving only -p and piped prompts"),
                "The terminal view, the front of a run on an interactive terminal. Without it the \
agent answers -p or what is piped in, and says so when started on a terminal.";
            off rig_basics::helpers::HelpersPlugin, "rig-basics", [], None,
                "The helper tool: hands tasks to other rig agents, each a `rig -p --json` child \
process with this session's plugins, one, several side by side or a chain, and \
returns their final answers.";
            off rig_basics::budget::BudgetPlugin, "rig-basics", [], None,
                "A budget for what the session spends, helpers included: [budget] usd and tokens \
in settings.toml, and /budget.";
            off rig_basics::subagents::SubagentsPlugin, "rig-basics", [], None,
                "Subagents: the task, message and wait tools, with which the agent starts other \
agents, messages them and waits on their reports.";
            off rig_steel::SteelPlugin, "rig-steel", [], None,
                "The run_steel tool: one Scheme program that runs several agents and tool calls \
and branches on their results.";
            off rig_basics::telemetry::effect_log::EffectLogPlugin, "rig-basics", [], None,
                "The effect log: every model call, tool call and message, as a record in the \
session's effects.jsonl.";
            off rig_basics::telemetry::diagnostics::DiagnosticsPlugin, "rig-basics", [], None,
                "The warnings and errors the agent logs, kept as data for the inspect tool.";
            off rig_mcp_client::McpPlugin, "rig-mcp-client", [], None,
                "MCP servers: those of RIG_HOME/mcp.json and the project's .rig/mcp.json, their \
tools as <server>__<tool>, and /mcp.";
            // Off: rig trusts a project's MCP servers as it trusts its other files.
            off rig_mcp_client::approval::McpApprovalPlugin, "rig-mcp-client", [], None,
                "With McpPlugin on, asks before a server of the project's .rig/mcp.json starts, \
and again once its entry changes: /mcp approve and /mcp deny.";
            off rig_acp::AcpPlugin, "rig-acp", [], None,
                "ACP: `rig acp` serves the Agent Client Protocol on stdin and stdout, so an editor \
such as Zed uses rig, each session in a process of its own, with the MCP servers \
the editor names.";
            off rig_claude_cloak::ClaudeCloakPlugin, "rig-claude-cloak", ["linux", "macos"], None,
                "Claude Cloak: `claude-cloak/*` models signed in with a Claude subscription \
(OAuth) instead of an API key, and /claude-cloak login and logout.";
        }
    };
}

/// [`BUILT_IN`] from [`built_in_plugins!`](crate::built_in_plugins).
macro_rules! built_in_table {
    ($($state:ident $first:ident $(:: $rest:ident)*, $krate:literal, [$($os:literal),*],
        $essential:expr, $about:literal;)*) => {
        /// Every one of rig's own plugins, in the order the agent adds them.
        pub const BUILT_IN: &[BuiltIn] = &[$(BuiltIn {
            plugin: concat!(stringify!($first) $(, "::", stringify!($rest))*),
            krate: $krate,
            on: built_in_table!(@on $state),
            targets: &[$($os),*],
            essential: $essential,
            about: $about,
        }),*];
    };
    (@on on) => { true };
    (@on off) => { false };
}

built_in_plugins!(built_in_table);

/// The built-in plugin `plugin` names.
pub fn built_in(plugin: &str) -> Option<&'static BuiltIn> {
    BUILT_IN.iter().find(|built_in| built_in.plugin == plugin)
}

/// What goes when the plugin `plugin` is left out, if its absence takes
/// away the way back from inside the agent: `rig plugin disable`, `rig
/// plugin list` and `/plugins` warn on it. `rig plugin enable` from a shell
/// always brings one back.
pub fn essential(plugin: &str) -> Option<&'static str> {
    built_in(plugin).and_then(|built_in| built_in.essential)
}

/// A plugin crate that is not rig's: one `[[plugins.extra]]` table of the
/// home's `settings.toml`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extra {
    /// The plugin type's path, such as `my_plugin::MyPlugin`.
    pub plugin: String,
    /// The crate it is in.
    pub krate: String,
    /// Where the crate comes from.
    pub source: Source,
}

/// Where an [`Extra`] crate comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// A local directory, made absolute against the directory of the
    /// settings file.
    Path(PathBuf),
    /// A git repository, optionally at a branch or revision.
    Git {
        /// The repository URL.
        url: String,
        /// The branch.
        branch: Option<String>,
        /// The revision.
        rev: Option<String>,
    },
    /// A crates.io version requirement.
    Version(String),
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Path(path) => write!(f, "path {}", path.display()),
            Self::Git { url, branch, rev } => {
                write!(f, "git {url}")?;
                if let Some(branch) = branch {
                    write!(f, ", branch {branch}")?;
                }
                if let Some(rev) = rev {
                    write!(f, ", rev {rev}")?;
                }
                Ok(())
            }
            Self::Version(version) => write!(f, "version {version}"),
        }
    }
}

/// Why the settings could not be read, chosen from or written. Each names
/// what it is about, so its text is what the user is told.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A file that could not be read or written, and why.
    Io {
        /// The file.
        path: PathBuf,
        /// What failed.
        message: String,
    },
    /// A file that is not TOML, at the line and column of its first error.
    NotToml {
        /// The file.
        path: PathBuf,
        /// The line and column, when known.
        at: Option<(usize, usize)>,
        /// What the parser said.
        message: String,
    },
    /// A table of a file that does not read as it must, and why.
    Table {
        /// The file.
        path: PathBuf,
        /// The table, such as `plugins`.
        table: String,
        /// What is wrong with it.
        problem: String,
    },
    /// A plugin turned on that does not build on this target.
    Unavailable {
        /// The plugin.
        plugin: String,
        /// The `target_os` values it builds on.
        targets: &'static [&'static str],
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "{}: {message}", path.display()),
            Self::NotToml { path, at, message } => {
                write!(f, "{}: not TOML", path.display())?;
                if let Some((line, column)) = at {
                    write!(f, " at line {line}, column {column}")?;
                }
                write!(f, ": {message}")
            }
            Self::Table {
                path,
                table,
                problem,
            } => write!(f, "{} [{table}]: {problem}", path.display()),
            Self::Unavailable { plugin, targets } => write!(
                f,
                "`{plugin}` does not build on {}; it builds on {} only",
                std::env::consts::OS,
                targets.join(" and ")
            ),
        }
    }
}

impl std::error::Error for Error {}

/// A settings file as read: where it is and what it holds, empty when it
/// does not exist.
#[derive(Clone, Debug)]
pub struct SettingsFile {
    /// Its path, which every problem with it names.
    pub path: PathBuf,
    /// Its contents.
    pub document: DocumentMut,
}

impl SettingsFile {
    /// The file at `path`; empty when it does not exist.
    ///
    /// # Errors
    ///
    /// When it cannot be read or is not TOML, naming the file, and the line
    /// and column of the first error.
    pub fn read(path: &Path) -> Result<Self, Error> {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::parse(path.to_path_buf(), &text),
            Err(failure) if failure.kind() == ErrorKind::NotFound => Ok(Self {
                path: path.to_path_buf(),
                document: DocumentMut::new(),
            }),
            Err(failure) => Err(Error::Io {
                path: path.to_path_buf(),
                message: format!("could not read: {failure}"),
            }),
        }
    }

    /// `text` as the file at `path`.
    ///
    /// # Errors
    ///
    /// When it is not TOML, naming `path`, and the line and column of the
    /// first error.
    pub fn parse(path: PathBuf, text: &str) -> Result<Self, Error> {
        match text.parse::<DocumentMut>() {
            Ok(document) => Ok(Self { path, document }),
            Err(failure) => {
                // toml's own text draws the line in several; a refusal is one.
                let at = (failure.span())
                    .and_then(|span| text.get(..span.start))
                    .map(|before| {
                        let line = before.matches('\n').count() + 1;
                        let column = before
                            .rsplit('\n')
                            .next()
                            .map_or(0, |it| it.chars().count())
                            + 1;
                        (line, column)
                    });
                let message = failure.message().trim_end().to_owned();
                Err(Error::NotToml { path, at, message })
            }
        }
    }

    /// The directory relative paths in it are relative to.
    fn directory(&self) -> &Path {
        self.path.parent().unwrap_or(Path::new("."))
    }
}

/// The plugins the agent is built with: the type paths of those on, rig's
/// in [`BUILT_IN`]'s order and then the extras in theirs, and every extra
/// crate of the file, on or off.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plugins {
    /// The plugins on, in the order the agent adds them.
    pub on: Vec<String>,
    /// Every `[[plugins.extra]]` table of the file.
    pub extra: Vec<Extra>,
}

impl Plugins {
    /// The extras that are on, which the agent is built with.
    pub fn extras_on(&self) -> impl Iterator<Item = &Extra> {
        (self.extra.iter()).filter(|extra| self.on.contains(&extra.plugin))
    }
}

/// The `[plugins]` table of one file, checked.
#[derive(Default)]
struct Changes {
    enable: Vec<String>,
    disable: Vec<String>,
    extra: Vec<Extra>,
}

/// The plugins the home's settings file `home` turns on, over
/// [`BUILT_IN`]'s defaults: its `disable`, then its `enable`.
///
/// # Errors
///
/// Naming the file: a `[plugins]` table that does not read, a plugin that
/// is not rig's or an extra, one named both on and off, or turning on a
/// plugin that does not build on this target.
pub fn plugins(home: &SettingsFile) -> Result<Plugins, Error> {
    let changes = changes(home).map_err(|problem| Error::Table {
        path: home.path.clone(),
        table: "plugins".to_owned(),
        problem,
    })?;
    let extras: Vec<&str> = (changes.extra.iter())
        .map(|extra| extra.plugin.as_str())
        .collect();
    for plugin in changes.enable.iter().chain(&changes.disable) {
        if built_in(plugin).is_none() && !extras.contains(&plugin.as_str()) {
            return Err(Error::Table {
                path: home.path.clone(),
                table: "plugins".to_owned(),
                problem: unknown(plugin),
            });
        }
    }
    let mut on: BTreeSet<&str> = (BUILT_IN.iter())
        .filter(|built_in| built_in.on)
        .map(|built_in| built_in.plugin)
        .chain(extras.iter().copied())
        .collect();
    for plugin in &changes.disable {
        on.remove(plugin.as_str());
    }
    on.extend(changes.enable.iter().map(String::as_str));
    for plugin in &on {
        if let Some(missing) = built_in(plugin).filter(|built_in| !built_in.available()) {
            return Err(Error::Unavailable {
                plugin: (*plugin).to_owned(),
                targets: missing.targets,
            });
        }
    }
    let on = (BUILT_IN.iter().map(|built_in| built_in.plugin))
        .chain(extras.iter().copied())
        .filter(|plugin| on.contains(plugin))
        .map(str::to_owned)
        .collect();
    Ok(Plugins {
        on,
        extra: changes.extra,
    })
}

/// The plugins `RIG_HOME/settings.toml` turns on ([`plugins`]).
///
/// # Errors
///
/// As [`plugins`], and when the file cannot be read or is not TOML.
pub fn chosen(home: &crate::Home) -> Result<Plugins, Error> {
    plugins(&SettingsFile::read(&home.settings())?)
}

/// Why `plugin` is refused when nothing provides it.
fn unknown(plugin: &str) -> String {
    format!("no plugin `{plugin}`; `rig plugin list` shows them")
}

/// The `[plugins]` table of `file`.
fn changes(file: &SettingsFile) -> Result<Changes, String> {
    let Some(item) = file.document.get("plugins") else {
        return Ok(Changes::default());
    };
    let table = item.as_table_like().ok_or("not a table")?;
    let mut changes = Changes::default();
    for (key, value) in table.iter() {
        match key {
            "enable" => changes.enable = names(key, value)?,
            "disable" => changes.disable = names(key, value)?,
            "extra" => {
                let tables = value
                    .as_array_of_tables()
                    .ok_or("`extra` is not a list of [[plugins.extra]] tables")?;
                for (index, table) in tables.iter().enumerate() {
                    let extra = self::extra(table, file.directory())
                        .map_err(|why| format!("[[plugins.extra]] {}: {why}", index + 1))?;
                    if changes
                        .extra
                        .iter()
                        .any(|other| other.plugin == extra.plugin)
                    {
                        return Err(format!("`{}` is listed twice", extra.plugin));
                    }
                    if (changes.extra.iter())
                        .any(|other| other.krate == extra.krate && other.source != extra.source)
                    {
                        return Err(format!(
                            "`{}` is listed before with another source",
                            extra.krate
                        ));
                    }
                    changes.extra.push(extra);
                }
            }
            other => {
                return Err(format!(
                    "unknown key `{other}`; it takes `enable`, `disable` and `extra`"
                ));
            }
        }
    }
    if let Some(both) = (changes.enable.iter()).find(|plugin| changes.disable.contains(plugin)) {
        return Err(format!("`{both}` is in both `enable` and `disable`"));
    }
    Ok(changes)
}

/// The plugin names of the list `key`.
fn names(key: &str, item: &Item) -> Result<Vec<String>, String> {
    let list = item
        .as_array()
        .ok_or_else(|| format!("`{key}` is not a list of plugin names"))?;
    list.iter()
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                format!(
                    "`{key}` holds `{}`, not a plugin name",
                    value.to_string().trim()
                )
            })
        })
        .collect()
}

/// The keys an extra plugin's table takes.
const EXTRA_KEYS: [&str; 7] = ["plugin", "crate", "path", "git", "branch", "rev", "version"];

/// One `[[plugins.extra]]` table; a relative `path` is relative to `base`.
fn extra(table: &dyn TableLike, base: &Path) -> Result<Extra, String> {
    let text = |key: &str| -> Result<Option<String>, String> {
        table
            .get(key)
            .map(|item| {
                item.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("`{key}` is not a string"))
            })
            .transpose()
    };
    let plugin = text("plugin")?.ok_or("`plugin` (the type path) is missing")?;
    if !is_type_path(&plugin) {
        return Err(format!("`{plugin}` is not a Rust type path"));
    }
    if built_in(&plugin).is_some() {
        return Err(format!(
            "`{plugin}` is one of rig's own plugins; name it in `enable` instead"
        ));
    }
    let krate = text("crate")?.ok_or("`crate` (the package name) is missing")?;
    if krate.is_empty()
        || !(krate.chars()).all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(format!("`{krate}` is not a crate name"));
    }
    if ["rig-harness", "rig-harness-agent", "bevy"].contains(&krate.as_str()) {
        return Err(format!("`{krate}` is part of the agent itself"));
    }
    let (branch, rev) = (text("branch")?, text("rev")?);
    let source = match (text("path")?, text("git")?, text("version")?) {
        (None, Some(url), None) => Source::Git { url, branch, rev },
        _ if branch.is_some() || rev.is_some() => {
            return Err("`branch` and `rev` only go with `git`".to_owned());
        }
        (Some(path), None, None) => {
            let path = base.join(path);
            Source::Path(std::path::absolute(&path).unwrap_or(path))
        }
        (None, None, Some(version)) => Source::Version(version),
        _ => return Err("set one of `path`, `git` and `version`".to_owned()),
    };
    if let Some((key, _)) = table.iter().find(|(key, _)| !EXTRA_KEYS.contains(key)) {
        return Err(format!("unknown key `{key}`"));
    }
    Ok(Extra {
        plugin,
        krate,
        source,
    })
}

/// Whether `text` is a Rust type path, such as `my_plugin::MyPlugin`.
fn is_type_path(text: &str) -> bool {
    text.split("::").all(|segment| {
        segment.chars().next().is_some_and(|c| !c.is_ascii_digit())
            && segment
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
    })
}

/// Turns `plugin` on or off in the `[plugins]` of `document`: it leaves
/// the list of the other state, and joins this one unless that is what it
/// is without the file. A name already in this list stays. Returns whether
/// the document changed.
pub fn set_on(document: &mut DocumentMut, plugin: &str, on: bool, by_default: bool) -> bool {
    let (into, out) = if on {
        ("enable", "disable")
    } else {
        ("disable", "enable")
    };
    let Some(table) = table(document, "plugins") else {
        return false;
    };
    let dropped = drop_name(table, out, plugin);
    if on == by_default {
        return dropped;
    }
    let list = table
        .entry(into)
        .or_insert_with(|| Item::Value(Value::Array(Array::new())));
    let Some(list) = list.as_array_mut() else {
        return dropped;
    };
    if list.iter().any(|value| value.as_str() == Some(plugin)) {
        return dropped;
    }
    list.push(plugin);
    true
}

/// Takes `plugin` out of the list `key` of `table`, and the key once it is
/// empty unless a comment goes with it. Returns whether it was there.
fn drop_name(table: &mut dyn TableLike, key: &str, plugin: &str) -> bool {
    let Some(list) = table.get_mut(key).and_then(Item::as_array_mut) else {
        return false;
    };
    let mut dropped = false;
    loop {
        let found = list.iter().position(|value| value.as_str() == Some(plugin));
        let Some(index) = found else { break };
        dropped = true;
        // A comment line above the name stays, above what comes next.
        let above = (list.remove(index).decor().prefix())
            .and_then(|prefix| prefix.as_str())
            .filter(|prefix| prefix.contains('#'))
            .map(str::to_owned);
        let Some(above) = above else { continue };
        match list.get_mut(index) {
            Some(next) => {
                let was = next
                    .decor()
                    .prefix()
                    .and_then(|it| it.as_str())
                    .unwrap_or_default();
                let joined = format!("{above}{}", was.trim_start_matches(' '));
                next.decor_mut().set_prefix(joined);
            }
            None => {
                let was = list.trailing().as_str().unwrap_or_default().to_owned();
                list.set_trailing(format!("{above}{was}"));
            }
        }
    }
    let commented = |text: Option<&str>| text.is_some_and(|text| text.contains('#'));
    let quiet = list.is_empty()
        && !commented(list.trailing().as_str())
        && !commented(list.decor().prefix().and_then(|it| it.as_str()))
        && !commented(list.decor().suffix().and_then(|it| it.as_str()));
    let key_quiet = table
        .key(key)
        .is_none_or(|key| !commented(key.leaf_decor().prefix().and_then(|it| it.as_str())));
    if quiet && key_quiet {
        table.remove(key);
    }
    dropped
}

/// The table `key` of `document`, added at its end when missing: after
/// every line of the file, its last comments included.
pub fn table<'a>(document: &'a mut DocumentMut, key: &str) -> Option<&'a mut dyn TableLike> {
    if !document.contains_key(key) {
        let after = document.trailing().as_str().unwrap_or_default().to_owned();
        document.set_trailing("");
        let mut table = toml_edit::Table::new();
        let gap = if after.is_empty() && !document.is_empty() {
            "\n"
        } else {
            ""
        };
        table.decor_mut().set_prefix(format!("{gap}{after}"));
        document.insert(key, Item::Table(table));
    }
    document.get_mut(key).and_then(Item::as_table_like_mut)
}

/// Sets `key` of `table` to the string `value`, keeping the comment on its
/// line. Returns whether it changed.
pub fn set_text(table: &mut dyn TableLike, key: &str, value: &str) -> bool {
    let mut new = Value::from(value);
    match table.get_mut(key) {
        Some(Item::Value(old)) if old.as_str() == Some(value) => false,
        Some(Item::Value(old)) => {
            *new.decor_mut() = old.decor().clone();
            *old = new;
            true
        }
        _ => {
            table.insert(key, Item::Value(new));
            true
        }
    }
}

/// Changes the settings file at `path` with `change`, which says whether it
/// changed anything, keeping every comment and line it does not touch. One
/// edit at a time, under a lock beside the file; the file is read again
/// under it and replaced whole, so a reader never sees half of it. A
/// symlinked file is written where it points. Nothing is written when
/// `change` fails or changes nothing. Returns whether it wrote.
///
/// # Errors
///
/// When the file does not read as TOML, `change` fails, or it cannot be
/// written, naming the file.
pub fn edit<T, E: From<Error>>(
    path: &Path,
    change: impl FnOnce(&mut SettingsFile) -> Result<(bool, T), E>,
) -> Result<(bool, T), E> {
    let named = |failure: std::io::Error| Error::Io {
        path: path.to_path_buf(),
        message: failure.to_string(),
    };
    let target = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let (directory, name) = match (target.parent(), target.file_name()) {
        (Some(directory), Some(name)) => (directory, name.to_string_lossy()),
        _ => {
            return Err(Error::Io {
                path: path.to_path_buf(),
                message: "not a file's path".to_owned(),
            }
            .into());
        }
    };
    std::fs::create_dir_all(directory).map_err(named)?;
    let lock = crate::lock_file(&directory.join(format!(".{name}.lock")), true).map_err(named)?;
    lock.lock().map_err(named)?;
    let mut file = SettingsFile::read(&target)?;
    file.path = path.to_path_buf();
    let (changed, value) = change(&mut file)?;
    if changed {
        let staged = directory.join(format!(".{name}.{}", std::process::id()));
        std::fs::write(&staged, file.document.to_string()).map_err(named)?;
        // A file the user made private stays so.
        if let Ok(was) = std::fs::metadata(&target) {
            std::fs::set_permissions(&staged, was.permissions()).map_err(named)?;
        }
        std::fs::rename(&staged, &target).map_err(|failure| {
            std::fs::remove_file(&staged).ok();
            named(failure)
        })?;
    }
    Ok((changed, value))
}

#[cfg(test)]
mod tests;

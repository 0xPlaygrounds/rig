//! Safer shell commands by rewriting (corpus spec `03-shell-rewrite.md`).
//! Before a `shell` call runs, its command goes through the ordered rules
//! of `RIG_HOME/shell-rewrite.toml`, each seeing the previous result. The
//! rewritten command is what runs, what every gate judges and what the
//! effect log records; the model's result starts with
//! "(ran as: `<rewritten>`)", and the conversation keeps the model's own
//! call.
//!
//! The rewrite is a Prepare system editing the call's `ToolRequest`, so
//! every Gate system (a guard) sees the rewritten command; a Review system
//! puts the "(ran as: …)" line on its result.
//!
//! ```toml
//! [[rule]]               # rm <files> -> trash <files>, when trash is on PATH
//! command = "rm"
//! replace = "trash"
//! needs_program = "trash"
//!
//! [[rule]]               # cargo test runs quietly on 4 jobs
//! command = "cargo test"
//! add = ["--quiet", "-j 4"]
//!
//! [[rule]]               # npm install -> pnpm install in a pnpm project
//! command = "npm install"
//! replace = "pnpm install"
//! needs_file = "pnpm-lock.yaml"
//!
//! [[rule]]               # every cargo command gets a timeout
//! command = "cargo"
//! prefix = "timeout 300"
//! ```

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use bevy_log::warn;
use corpus_shared::protected_paths::ProtectedPathsSystems;
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::Answered;
use rig_harness::rig_core::message::ToolResultContent;
use rig_harness::rig_core::tool::ToolOutput;
use rig_harness::rig_tools::Shell;
use serde::Deserialize;

/// The rules' file in `RIG_HOME`.
pub const FILE: &str = "shell-rewrite.toml";

/// One rewriting rule. It applies to each simple command of a command line
/// (the parts between unquoted `;`, `&&`, `||`, `|`, `&` and newlines)
/// that starts with `command`, while its conditions hold.
#[derive(Deserialize, Clone, Debug, Default, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct Rule {
    /// The leading words a simple command must start with, such as
    /// `cargo test`; empty matches every command.
    pub command: String,
    /// What replaces those words.
    pub replace: Option<String>,
    /// Arguments put after the command's words, each unless its first word
    /// is in the command already.
    pub add: Vec<String>,
    /// Put before the command unless its first word starts it already.
    pub prefix: Option<String>,
    /// Applies only when this program is on the command's `PATH`.
    pub needs_program: Option<String>,
    /// Applies only when this file exists in the command's directory.
    pub needs_file: Option<String>,
}

/// Where a command runs, for the rules' conditions.
#[derive(Clone, Debug, Default)]
pub struct Place {
    /// The command's working directory.
    pub dir: PathBuf,
    /// The command's `PATH`.
    pub path: Option<OsString>,
}

/// The rules of `RIG_HOME/shell-rewrite.toml`, in file order.
#[derive(Resource, Deserialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct ShellRewrites {
    /// The rules, `[[rule]]` in the file.
    #[serde(default, rename = "rule")]
    pub rules: Vec<Rule>,
}

impl ShellRewrites {
    /// The rules of a `shell-rewrite.toml` text.
    pub fn parse(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    /// The rules of the file at `path`; none when it does not exist.
    pub fn load(path: &Path) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::parse(&text).map_err(|error| error.to_string()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(error.to_string()),
        }
    }

    /// `command`, run at `place`, after every rule, in order.
    pub fn rewrite(&self, command: &str, place: &Place) -> String {
        self.rules
            .iter()
            .filter(|rule| rule.holds(place))
            .fold(command.to_owned(), |command, rule| rule.apply(&command))
    }
}

impl Rule {
    /// `command` with this rule applied to each of its simple commands.
    fn apply(&self, command: &str) -> String {
        pieces(command)
            .into_iter()
            .enumerate()
            .map(|(at, piece)| match at % 2 {
                0 => self.apply_one(piece),
                _ => piece.to_owned(),
            })
            .collect()
    }

    /// Whether the rule's conditions hold for a command run at `place`.
    fn holds(&self, place: &Place) -> bool {
        let on_path = |program| on_path(program, place.path.as_ref());
        self.needs_program.as_deref().is_none_or(on_path)
            && self
                .needs_file
                .as_deref()
                .is_none_or(|file| place.dir.join(file).exists())
    }

    /// The simple command `piece`, with its surrounding blanks, rewritten.
    fn apply_one(&self, piece: &str) -> String {
        let trimmed = piece.trim();
        let Some(rest) = after_words(trimmed, &self.command).filter(|_| !trimmed.is_empty()) else {
            return piece.to_owned();
        };
        let words: Vec<&str> = trimmed.split_whitespace().collect();
        let adds: Vec<&str> = self
            .add
            .iter()
            .map(String::as_str)
            .filter(|add| first_word(add).is_some_and(|flag| !words.contains(&flag)))
            .collect();
        let head = trimmed
            .get(..trimmed.len() - rest.len())
            .unwrap_or_default();
        let new_head = self.replace.as_deref().unwrap_or(head);
        let mut parts = vec![new_head];
        parts.extend(&adds);
        parts.push(rest.trim_start());
        let body: Vec<&str> = parts.into_iter().filter(|part| !part.is_empty()).collect();
        let mut body = body.join(" ");
        let prefix = self
            .prefix
            .as_deref()
            .filter(|prefix| first_word(prefix) != first_word(&body));
        if let Some(prefix) = prefix {
            body = format!("{prefix} {body}");
        }
        if self.replace.is_none() && adds.is_empty() && prefix.is_none() {
            return piece.to_owned();
        }
        let lead = piece.len() - piece.trim_start().len();
        let tail = piece.trim_end().len();
        format!(
            "{}{body}{}",
            piece.get(..lead).unwrap_or_default(),
            piece.get(tail..).unwrap_or_default()
        )
    }
}

/// `command` cut at its unquoted separators: simple commands at even
/// indices, the separators between them at odd ones.
fn pieces(command: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let (mut start, mut quote, mut escaped) = (0, None, false);
    let mut previous = None;
    let mut chars = command.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        let before = previous.replace(c);
        if escaped {
            escaped = false;
            continue;
        }
        let next = chars.peek().map(|&(_, next)| next);
        let separates = match (quote, c) {
            (Some('\''), '\'') | (Some('"'), '"') => {
                quote = None;
                false
            }
            (Some('\''), _) => false,
            (_, '\\') => {
                escaped = true;
                false
            }
            (Some(_), _) => false,
            (None, '\'' | '"') => {
                quote = Some(c);
                false
            }
            // `2>&1` and `&>` redirect.
            (None, '&') => !matches!(before, Some('>' | '<')) && next != Some('>'),
            (None, ';' | '|' | '\n') => true,
            _ => false,
        };
        if !separates {
            continue;
        }
        let mut end = at + c.len_utf8();
        if matches!(c, '&' | '|') && next == Some(c) {
            chars.next();
            previous = Some(c);
            end += c.len_utf8();
        }
        pieces.push(command.get(start..at).unwrap_or_default());
        pieces.push(command.get(at..end).unwrap_or_default());
        start = end;
    }
    pieces.push(command.get(start..).unwrap_or_default());
    pieces
}

/// What follows the leading `words` of `text`, from the blank after them;
/// `None` when `text` does not start with them.
fn after_words<'a>(text: &'a str, words: &str) -> Option<&'a str> {
    words.split_whitespace().try_fold(text, |text, word| {
        let rest = text.trim_start().strip_prefix(word)?;
        rest.chars()
            .next()
            .is_none_or(char::is_whitespace)
            .then_some(rest)
    })
}

fn first_word(text: &str) -> Option<&str> {
    text.split_whitespace().next()
}

/// Whether `program` is an executable file in a directory of `path`.
fn on_path(program: &str, path: Option<&OsString>) -> bool {
    path.is_some_and(|path| std::env::split_paths(path).any(|dir| executable(&dir.join(program))))
}

#[cfg(unix)]
fn executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.metadata()
        .is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn executable(path: &Path) -> bool {
    path.is_file()
}

/// On a `shell` call whose command a rule rewrote: the command it runs as,
/// until its result says so.
#[derive(Component, Clone, Debug)]
#[component(storage = "SparseSet")]
pub struct RanAs(pub String);

/// The plugin's stage systems, for others to order against: `Prepare`
/// edits each `shell` call's command, `Review` replaces the result of each
/// rewritten one.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShellRewriteSystems {
    /// Rewrites the command of each `shell` call as it is admitted, in
    /// [`PrepareSystems::Edit`].
    Prepare,
    /// Puts "(ran as: …)" first in each rewritten call's result, after
    /// protected-paths filtered it.
    Review,
}

/// Rewrites `shell` commands by the rules of `RIG_HOME/shell-rewrite.toml`
/// before they run.
#[derive(Default)]
pub struct ShellRewritePlugin;

impl Plugin for ShellRewritePlugin {
    fn build(&self, app: &mut App) {
        let path = RigHome::of(app.world()).root().join(FILE);
        let rewrites = ShellRewrites::load(&path).unwrap_or_else(|error| {
            warn!("{}: {error}; no shell command is rewritten", path.display());
            ShellRewrites::default()
        });
        app.insert_resource(rewrites)
            .init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .configure_sets(
                AgentPass,
                (
                    ShellRewriteSystems::Prepare.in_set(PrepareSystems::Edit),
                    ShellRewriteSystems::Review
                        .in_set(Stage::Review)
                        .after(ProtectedPathsSystems::Review),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    rewrite.in_set(ShellRewriteSystems::Prepare),
                    say_what_ran.in_set(ShellRewriteSystems::Review),
                ),
            );
    }
}

/// Rewrites each `shell` call's command as it is admitted. The
/// conversation keeps the model's call; the call's request is what runs,
/// what gates judge and what the effect log records.
fn rewrite(
    mut calls: Query<(Entity, &mut ToolRequest, &ForAgent), (Added<Admitted>, ToolCalls)>,
    dirs: Query<&WorkDir>,
    rules: Res<ShellRewrites>,
    processes: Res<Processes>,
    root: Res<ProjectRoot>,
    mut commands: Commands,
) {
    if rules.rules.is_empty() {
        return;
    }
    let launcher = processes.launcher();
    let base = launcher.cwd.clone().unwrap_or_else(|| root.0.clone());
    let path = launcher
        .env
        .iter()
        .rev()
        .find(|(name, _)| name == "PATH")
        .map(|(_, value)| value.clone())
        .or_else(|| std::env::var_os("PATH"));
    for (call, mut request, &ForAgent(agent)) in &mut calls {
        let arguments = &request.call.function.arguments;
        if request.call.function.name.as_str() != Shell::NAME {
            continue;
        }
        let Some(command) = arguments.get("command").and_then(|c| c.as_str()) else {
            continue;
        };
        let dir = arguments.get("cwd").and_then(|dir| dir.as_str());
        // An agent's own working directory is where its commands run.
        let base = dirs.get(agent).map_or(&base, |WorkDir(dir)| dir);
        let place = Place {
            dir: dir.map_or_else(|| base.clone(), |dir| base.join(dir)),
            path: path.clone(),
        };
        let rewritten = rules.rewrite(command, &place);
        if rewritten == command {
            continue;
        }
        let arguments = &mut request.call.function.arguments;
        arguments.insert("command".to_owned(), rewritten.clone().into());
        commands.entity(call).insert(RanAs(rewritten));
    }
}

/// Puts "(ran as: `<rewritten>`)" first in each rewritten call's result.
fn say_what_ran(
    mut calls: Query<(Entity, &RanAs, &mut Outcome, Has<Started>), (ReadyToSettle, ToolCalls)>,
    mut commands: Commands,
) {
    for (call, RanAs(ran_as), mut outcome, started) in &mut calls {
        // Once: Review sees the outcome each pass until it settles.
        commands.entity(call).remove::<RanAs>();
        let Ok(Answered::ToolResult { result }) = &outcome.0 else {
            continue;
        };
        if !started {
            continue;
        }
        let line = format!("(ran as: `{ran_as}`)");
        let output = match result.output().as_text() {
            Some(text) => ToolOutput::text(format!("{line}\n{text}")),
            None => {
                let content = std::iter::once(ToolResultContent::text(line))
                    .chain(result.output().as_content().iter().cloned())
                    .collect();
                let Ok(output) = ToolOutput::content(content) else {
                    continue;
                };
                output
            }
        };
        *outcome = Outcome::from(result.clone().with_output(output));
    }
}

#[cfg(test)]
mod tests;

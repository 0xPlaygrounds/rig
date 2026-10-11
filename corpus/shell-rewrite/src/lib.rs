//! Safer shell commands by rewriting (corpus spec `03-shell-rewrite.md`).
//! Before a `shell` call runs, its command goes through the ordered rules
//! of `RIG_HOME/shell-rewrite.toml`, each seeing the previous result. The
//! rewritten command is what runs and what the effect log records; the
//! model's result starts with "(ran as: `<rewritten>`)", and the
//! conversation keeps the model's own call.
//!
//! The rewrite is a rig-core layer ([`Intercept`]) that patches the call's
//! arguments. The kernel has no way to put a layer on a registered tool, so
//! the plugin turns the `shell` tool off and registers it again, served by
//! a `rig_tools::Shell` set up as `rig-coding-tools` sets it up.
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

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, PoisonError};

use bevy_log::warn;
use rig_core::effect::{EffectId, EffectKind, Outcome};
use rig_core::error::ErrorReport;
use rig_core::message::ToolResultContent;
use rig_core::serve::adapters::ToolAdapter;
use rig_core::serve::{Decision, ErasedHandler, Intercept, Verdict};
use rig_core::wasm_compat::WasmCompatSend;
use rig_harness::harness_protocol::env::AGENT_ONLY;
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::ToolOutput as Output;
use rig_tools::Shell;
use serde::Deserialize;

/// The tool whose calls are rewritten.
const SHELL: &str = "shell";
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
    /// Applies only when this program is on `PATH`.
    pub needs_program: Option<String>,
    /// Applies only when this file exists in the working directory.
    pub needs_file: Option<String>,
}

/// The rules, in file order. Inserted before the plugin is added, it takes
/// the place of `RIG_HOME/shell-rewrite.toml`.
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

    /// `command` after every rule, in order.
    pub fn rewrite(&self, command: &str) -> String {
        self.rules
            .iter()
            .fold(command.to_owned(), |command, rule| rule.apply(&command))
    }
}

impl Rule {
    /// `command` with this rule applied to each of its simple commands.
    fn apply(&self, command: &str) -> String {
        if !self.holds() {
            return command.to_owned();
        }
        pieces(command)
            .into_iter()
            .enumerate()
            .map(|(at, piece)| match at % 2 {
                0 => self.apply_one(piece),
                _ => piece.to_owned(),
            })
            .collect()
    }

    /// Whether the rule's conditions hold now.
    fn holds(&self) -> bool {
        self.needs_program.as_deref().is_none_or(on_path)
            && self
                .needs_file
                .as_deref()
                .is_none_or(|file| Path::new(file).exists())
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

/// Whether `program` is an executable file in a directory of `PATH`.
fn on_path(program: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|path| std::env::split_paths(&path).any(|dir| executable(&dir.join(program))))
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

/// The layer on the `shell` handler: patches each call's command and,
/// after it ran, says what ran in the model's result.
struct Rewrite {
    rules: ShellRewrites,
    /// The rewritten command of each dispatch the layer patched, until its
    /// answer.
    ran_as: Mutex<HashMap<EffectId, String>>,
}

impl Rewrite {
    fn decide(&self, id: EffectId, kind: &EffectKind) -> Decision {
        let EffectKind::ToolCall { name, args } = kind else {
            return Decision::Proceed;
        };
        let Ok(serde_json::Value::Object(mut args)) = serde_json::from_str(args) else {
            return Decision::Proceed;
        };
        let Some(command) = args.get("command").and_then(serde_json::Value::as_str) else {
            return Decision::Proceed;
        };
        let rewritten = self.rules.rewrite(command);
        if rewritten == command {
            return Decision::Proceed;
        }
        args.insert("command".to_owned(), rewritten.clone().into());
        let Ok(args) = serde_json::to_string(&args) else {
            return Decision::Proceed;
        };
        self.ran_as
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id, rewritten);
        Decision::Patch(EffectKind::ToolCall {
            name: name.clone(),
            args,
        })
    }

    fn report(&self, id: EffectId, outcome: &Result<Outcome, ErrorReport>) -> Verdict {
        let ran_as = self
            .ran_as
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&id);
        let (Some(ran_as), Ok(Outcome::ToolResult { result })) = (ran_as, outcome) else {
            return Verdict::Keep;
        };
        let line = format!("(ran as: `{ran_as}`)");
        let output = match result.output().as_text() {
            Some(text) => Output::text(format!("{line}\n{text}")),
            None => {
                let content = std::iter::once(ToolResultContent::text(line))
                    .chain(result.output().as_content().iter().cloned())
                    .collect();
                match Output::content(content) {
                    Ok(output) => output,
                    Err(_) => return Verdict::Keep,
                }
            }
        };
        let result = result.clone().with_output(output);
        Verdict::Replace(Ok(Outcome::ToolResult { result }))
    }
}

impl Intercept for Rewrite {
    fn name(&self) -> String {
        "shell-rewrite".to_owned()
    }

    fn before(
        &self,
        id: EffectId,
        kind: &EffectKind,
    ) -> impl Future<Output = Decision> + WasmCompatSend {
        std::future::ready(self.decide(id, kind))
    }

    fn after(
        &self,
        id: EffectId,
        _kind: &EffectKind,
        outcome: &Result<Outcome, ErrorReport>,
    ) -> impl Future<Output = Verdict> + WasmCompatSend {
        std::future::ready(self.report(id, outcome))
    }
}

/// Rewrites `shell` commands by [`ShellRewrites`] before they run.
#[derive(Default)]
pub struct ShellRewritePlugin;

impl Plugin for ShellRewritePlugin {
    fn build(&self, app: &mut App) {
        if app.world().contains_resource::<ShellRewrites>() {
            return;
        }
        let path = Home::from_env().root().join(FILE);
        let rewrites = ShellRewrites::load(&path).unwrap_or_else(|error| {
            warn!("{}: {error}; no shell command is rewritten", path.display());
            ShellRewrites::default()
        });
        app.insert_resource(rewrites);
    }

    /// Serves the registered `shell` tool again behind the rewriting layer,
    /// with its definition, rules and footprint.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let Some(rules) = world.get_resource::<ShellRewrites>().cloned() else {
            return;
        };
        let mut tools = world.query::<(Entity, &ToolDef, &ToolRules, &Footprint)>();
        let Some((tool, def, tool_rules, footprint)) = tools
            .iter(world)
            .find(|(_, def, ..)| def.0.name.as_str() == SHELL)
            .map(|(tool, def, rules, &footprint)| {
                (tool, def.0.clone(), rules.0.clone(), footprint)
            })
        else {
            return;
        };
        world.entity_mut(tool).insert(Disabled);
        let shell = Shell {
            unset_env: &AGENT_ONLY,
            spill: world
                .get_resource::<SessionPaths>()
                .map(SessionPaths::spill),
        };
        let rewrite = Rewrite {
            rules,
            ran_as: Mutex::default(),
        };
        let handler = ErasedHandler::new(ToolAdapter::new(shell)).layered(rewrite);
        let tool_rules: Vec<&str> = tool_rules.iter().map(String::as_str).collect();
        let options = ToolOptions {
            rules: &tool_rules,
            footprint,
        };
        world.spawn_tool(def, handler, options);
    }
}

#[cfg(test)]
mod tests;

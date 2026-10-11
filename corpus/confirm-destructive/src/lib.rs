//! Plugin corpus: 01. Confirm destructive shell commands (spec
//! `01-confirm-destructive.md`).
//!
//! The kernel starts an ordinary tool's call the moment the model asks for
//! it, so a `shell` command cannot be held back from outside. The plugin
//! therefore puts an open tool in place of `rig-coding-tools`' `shell`
//! (same name, description, rules and parameters): a command that matches
//! no [`DestructivePatterns`] runs at once with rig-tools' [`Shell`], one
//! that matches waits, [`AwaitingApproval`], for the user's answer in a
//! picker (`/confirm`), and is denied at once when nobody sits at the
//! terminal. "Allow for this session" keeps the pattern in the saved
//! [`SessionAllowances`].

use std::fs;
use std::io::ErrorKind;

use bevy_log::warn;
use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use regex::Regex;
use rig_core::tool::IntoToolOutput;
use rig_harness::prelude::*;
use rig_tools::{Shell, ShellArgs};
use rig_tui::{PickItem, PickRequest};
use serde::Deserialize;

/// The commands asked about unless `RIG_HOME/confirm.toml` adds more.
pub const BUILT_IN: &[&str] = &[
    r"\brm\s+(?:-\S*\s+)*-[a-zA-Z]*[rRf]",
    r"\brm\b[^;&|]*\s--(?:recursive|force)\b",
    r"\bgit\s+push\b[^;&|]*\s(?:--force\S*|-f\b)",
    r"\bgit\s+reset\b[^;&|]*\s--hard\b",
    r"\bgit\s+clean\b[^;&|]*\s-[a-zA-Z]*f",
    r"\bdd\s",
    r"\bmkfs\b",
    r"(?i)\bdrop\s+(?:table|database|schema)\b",
];

/// What the model reads for a call the user denied.
pub const DENIED: &str = "The user denied this command";

/// What the model reads for a matching call with nobody at the terminal.
pub const HEADLESS: &str = "needs approval; not available non-interactively";

/// The patterns a `shell` command is checked against.
#[derive(Resource, Clone, Debug)]
pub struct DestructivePatterns(pub Vec<Regex>);

impl DestructivePatterns {
    /// The patterns of `sources`; one that is not a valid regex is left out
    /// with a warning.
    pub fn new(sources: impl IntoIterator<Item = String>) -> Self {
        let compile = |source: String| {
            Regex::new(&source)
                .inspect_err(|error| warn!("confirm: pattern `{source}` left out: {error}"))
                .ok()
        };
        Self(sources.into_iter().filter_map(compile).collect())
    }

    /// The built-in patterns and those of `RIG_HOME/confirm.toml`:
    /// `patterns = ['\bterraform\s+destroy\b']`.
    fn from_home() -> Self {
        let path = Home::from_env().root().join("confirm.toml");
        let extra = match fs::read_to_string(&path) {
            Ok(text) => toml::from_str::<ConfirmFile>(&text)
                .inspect_err(|error| warn!("confirm: {} not read: {error}", path.display()))
                .map(|file| file.patterns)
                .unwrap_or_default(),
            Err(error) if error.kind() == ErrorKind::NotFound => Vec::new(),
            Err(error) => {
                warn!("confirm: {} not read: {error}", path.display());
                Vec::new()
            }
        };
        Self::new(BUILT_IN.iter().copied().map(str::to_owned).chain(extra))
    }

    /// The first pattern `command` matches.
    pub fn matching(&self, command: &str) -> Option<&Regex> {
        self.0.iter().find(|pattern| pattern.is_match(command))
    }
}

impl Default for DestructivePatterns {
    fn default() -> Self {
        Self::new(BUILT_IN.iter().copied().map(str::to_owned))
    }
}

/// `RIG_HOME/confirm.toml`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfirmFile {
    /// Regexes added to the built-in ones.
    #[serde(default)]
    patterns: Vec<String>,
}

/// The patterns the user allowed for the rest of the session, kept with
/// it across `/reload` and `/resume`.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved, Default)]
pub struct SessionAllowances(pub Vec<String>);

/// A `shell` call waiting for the user's answer, on the call's entity.
/// Interrupting the turn despawns the call, and the question with it.
#[derive(Component, Clone, Debug)]
pub struct AwaitingApproval {
    /// The calling agent.
    pub agent: Entity,
    /// The command.
    pub command: String,
    /// The pattern it matched.
    pub pattern: String,
    /// Whether the user chose "Deny with reason": the next message they
    /// type to the agent is the reason.
    pub reason_asked: bool,
}

impl AwaitingApproval {
    /// The picker asking about the call `call`, with "Deny" selected.
    fn picker(&self, call: Entity) -> PickRequest {
        let key = call.to_bits();
        let item = |label: &str, answer: &str| PickItem {
            label: label.to_owned(),
            command: format!("confirm {key} {answer}"),
        };
        PickRequest {
            agent: self.agent,
            title: format!("Allow `{}`?", self.command),
            items: vec![
                item("Allow once", "allow"),
                item("Allow for this session", "session"),
                item("Deny", "deny"),
                item("Deny with reason", "reason"),
            ],
            selected: 2,
        }
    }
}

/// The `shell` tool that runs a command once it is allowed.
#[derive(Resource, Clone)]
struct RealShell(Shell);

/// Asks before a `shell` command matching a [`DestructivePatterns`] runs.
#[derive(Default)]
pub struct ConfirmDestructivePlugin;

impl Plugin for ConfirmDestructivePlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<DestructivePatterns>() {
            app.insert_resource(DestructivePatterns::from_home());
        }
        // As `rig_coding_tools::ShellTool` builds it.
        let shell = Shell {
            unset_env: &rig_harness::harness_protocol::env::AGENT_ONLY,
            spill: app
                .world()
                .get_resource::<SessionPaths>()
                .map(SessionPaths::spill),
        };
        app.add_message::<PickRequest>()
            .init_resource::<SessionAllowances>()
            .insert_resource(RealShell(shell))
            .add_command(
                "confirm",
                "Answer a shell command waiting for approval; alone, asks again",
                confirm,
            )
            .add_observer(take_reason)
            .add_observer(ask_next);
    }

    /// Puts the gate in place of the `shell` tool, once every plugin added
    /// its tools; without a `shell` tool there is nothing to guard.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let mut tools = world.query::<(Entity, &ToolDef, &ToolRules)>();
        let shell = tools
            .iter(world)
            .find(|(_, def, _)| def.0.name.as_str() == <Shell as PortableTool>::NAME)
            .map(|(tool, def, rules)| (tool, def.0.description.clone(), rules.0.clone()));
        let Some((tool, description, rules)) = shell else {
            return;
        };
        world.entity_mut(tool).insert(Disabled);
        let rules: Vec<&str> = rules.iter().map(String::as_str).collect();
        let options = ToolOptions {
            rules: &rules,
            footprint: Footprint::Exclusive,
        };
        app.add_open_tool(<Shell as PortableTool>::NAME, &description, options, gate);
    }
}

/// Runs a `shell` call, holds it for the user, or denies it with nobody to
/// ask.
fn gate(
    called: On<ToolCalled<ShellArgs>>,
    patterns: Res<DestructivePatterns>,
    allowed: Res<SessionAllowances>,
    invoked: Option<Res<Invoked>>,
    shell: Res<RealShell>,
    wake: Res<Wake>,
    waiting: Query<(), With<AwaitingApproval>>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
) {
    let arguments = &called.run.call.function.arguments;
    let command = arguments
        .get("command")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let pattern = patterns
        .matching(command)
        .map(|pattern| pattern.as_str().to_owned())
        .filter(|pattern| !allowed.0.contains(pattern));
    let Some(pattern) = pattern else {
        run(&mut commands, called.call, &shell.0, arguments, &wake);
        return;
    };
    if !invoked.is_some_and(|invoked| invoked.interactive()) {
        let why = format!("`{command}` {HEADLESS}");
        let denied = ToolResult::failed(ToolExecutionError::refused(why));
        commands.entity(called.call).insert(ToolOutput(denied));
        return;
    }
    let awaiting = AwaitingApproval {
        agent: called.agent,
        command: command.to_owned(),
        pattern,
        reason_asked: false,
    };
    // One question at a time: the next is asked once this one is answered.
    if waiting.is_empty() {
        picks.write(awaiting.picker(called.call));
    }
    commands.entity(called.call).insert(awaiting);
}

/// Runs the call `call` with `arguments` on the real `shell` tool; its
/// output ends the call.
fn run(
    commands: &mut Commands,
    call: Entity,
    shell: &Shell,
    arguments: &serde_json::Map<String, serde_json::Value>,
    wake: &Wake,
) {
    let arguments = serde_json::Value::Object(arguments.clone());
    let (shell, args) = (
        shell.clone(),
        serde_json::from_value::<ShellArgs>(arguments),
    );
    let args = args.map_err(ToolExecutionError::from_error);
    let work = async move {
        let output = match args {
            Ok(args) => PortableTool::call(&shell, args)
                .await
                .map_err(|error| PortableTool::map_error(&shell, error)),
            Err(error) => Err(error),
        };
        output
            .and_then(IntoToolOutput::into_tool_output)
            .map_or_else(ToolResult::failed, ToolResult::success)
    };
    let pool = AsyncComputeTaskPool::get_or_init(TaskPool::default);
    commands
        .entity(call)
        .insert(Running::spawn_into::<ToolOutput, _>(pool, wake, work));
}

/// Ends the waiting call `call` as denied, for `reason` if given.
fn deny(commands: &mut Commands, call: Entity, reason: Option<&str>) {
    let why = match reason.map(str::trim).filter(|reason| !reason.is_empty()) {
        Some(reason) => format!("{DENIED}: {reason}"),
        None => format!("{DENIED}."),
    };
    let denied = ToolResult::failed(ToolExecutionError::refused(why));
    commands
        .entity(call)
        .remove::<AwaitingApproval>()
        .insert(ToolOutput(denied));
}

/// `/confirm <call> allow|session|deny [reason]|reason`, as the picker
/// runs it; `/confirm` alone asks about the first waiting call again, such
/// as after the picker was closed.
fn confirm(
    In(args): In<CommandArgs>,
    mut waiting: Query<(Entity, &ToolCallRun, &mut AwaitingApproval)>,
    mut allowed: ResMut<SessionAllowances>,
    shell: Res<RealShell>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    let mut words = args.args.splitn(3, char::is_whitespace);
    let (key, answer, reason) = (words.next(), words.next(), words.next());
    let call = key
        .filter(|key| !key.is_empty())
        .and_then(|key| key.parse().ok())
        .and_then(Entity::try_from_bits);
    let Some(call) = call else {
        match waiting.iter().next() {
            Some((call, _, awaiting)) => {
                picks.write(awaiting.picker(call));
            }
            None => {
                notices.write(Notice::info(args.agent, "No shell command waits."));
            }
        }
        return;
    };
    let Ok((call, run_of, mut awaiting)) = waiting.get_mut(call) else {
        notices.write(Notice::error(
            args.agent,
            "That shell command no longer waits.",
        ));
        return;
    };
    match answer.unwrap_or_default() {
        "allow" | "session" => {
            if answer == Some("session") && !allowed.0.contains(&awaiting.pattern) {
                allowed.0.push(awaiting.pattern.clone());
            }
            commands.entity(call).remove::<AwaitingApproval>();
            let arguments = &run_of.call.function.arguments;
            run(&mut commands, call, &shell.0, arguments, &wake);
        }
        "deny" => deny(&mut commands, call, reason),
        "reason" => {
            awaiting.reason_asked = true;
            notices.write(Notice::info(
                awaiting.agent,
                format!(
                    "Type why `{}` is denied: your next message is the reason.",
                    awaiting.command
                ),
            ));
        }
        other => {
            notices.write(Notice::error(
                args.agent,
                format!("/confirm: `{other}` is not allow, session, deny or reason."),
            ));
        }
    }
}

/// The user's next message after "Deny with reason" is the reason: it
/// denies the call and goes nowhere else.
fn take_reason(
    mut typed: On<Deliver>,
    waiting: Query<(Entity, &AwaitingApproval)>,
    mut commands: Commands,
) {
    if typed.origin.kind != OriginKind::User || typed.command().is_some() {
        return;
    }
    let agent = typed.entity;
    let asked = waiting
        .iter()
        .find(|(_, awaiting)| awaiting.agent == agent && awaiting.reason_asked);
    if let Some((call, _)) = asked {
        let reason = std::mem::take(&mut typed.text);
        deny(&mut commands, call, Some(&reason));
    }
}

/// Once a question is answered, or its call is gone, asks the next one.
fn ask_next(
    answered: On<Remove<AwaitingApproval>>,
    waiting: Query<(Entity, &AwaitingApproval)>,
    mut picks: MessageWriter<PickRequest>,
) {
    let next = waiting
        .iter()
        .find(|(call, awaiting)| *call != answered.entity && !awaiting.reason_asked);
    if let Some((call, awaiting)) = next {
        picks.write(awaiting.picker(call));
    }
}

#[cfg(test)]
mod tests;

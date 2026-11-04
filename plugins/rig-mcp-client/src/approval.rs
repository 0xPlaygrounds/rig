//! Asking before a project's MCP server runs, built on the public
//! [`McpHold`] as any policy plugin would be. A server of the project's
//! `.rig/mcp.json` is held until the user approves it as it is written,
//! with `/mcp approve <name>`; `/mcp deny <name>` holds it and stops the
//! question. The answers are kept for this user and project in
//! `mcp/approvals.json` of the project's plugin data, never in the project.
//! A server whose entry changed since is asked about again, and so is every
//! server of a file kept for another project, as two projects' data can
//! share a directory.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_ecs::system::SystemParam;
use rig_harness::plugin_data::PluginData;
use rig_harness::prelude::*;
use rig_harness::rig_memory::Files;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{McpCorePlugin, McpHold, McpServer, McpState, McpStatus, Scope};

#[cfg(test)]
mod tests;

/// The file in the project's plugin data.
const FILE: &str = "mcp/approvals.json";

/// Holds each server of the project's `.rig/mcp.json` until the user
/// approves it as written: `/mcp approve` and `/mcp deny`.
#[derive(Default)]
pub struct McpApprovalPlugin;

impl Plugin for McpApprovalPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<McpCorePlugin>() {
            app.add_plugins(McpCorePlugin);
        }
        app.add_systems(PreStartup, read_approvals)
            .add_observer(hold_unapproved)
            .add_command(
                "mcp-approve",
                "Start a project's MCP server as written: `/mcp approve <name>`",
                approve,
            )
            .add_command(
                "mcp-deny",
                "Stop asking about a project's MCP server: `/mcp deny <name>`",
                deny,
            );
    }
}

/// What the file holds: the project it is about, and the answers.
#[derive(Serialize, Deserialize, Default)]
struct Kept {
    project: String,
    servers: BTreeMap<String, Answer>,
}

/// One answer: whether `server`, as written then, may start.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
struct Answer {
    approved: bool,
    server: Value,
}

/// The answers of this project, by server name.
#[derive(Resource)]
struct Approvals {
    /// Where they are kept; in memory only without plugin data, or for a
    /// project whose path is not UTF-8, which the file could not tell from
    /// another.
    files: Option<Arc<dyn Files>>,
    /// The project, as its path.
    project: String,
    /// The lock a change holds while it reads the file again and writes it.
    lock: PathBuf,
    answers: BTreeMap<String, Answer>,
}

impl Approvals {
    /// The answers about `project` kept in `data`, if any, each change
    /// holding `lock`; a file that does not read, or is about another
    /// project, is no answers.
    fn read(data: Option<&PluginData>, project: &Path, lock: PathBuf) -> Self {
        let files = data.filter(|_| project.to_str().is_some());
        let mut approvals = Self {
            files: files.map(|data| data.project.clone()),
            project: project.display().to_string(),
            lock,
            answers: BTreeMap::new(),
        };
        approvals.answers = approvals.kept();
        approvals
    }

    /// The answers the file holds now.
    fn kept(&self) -> BTreeMap<String, Answer> {
        let kept = self
            .files
            .as_ref()
            .and_then(|files| files.read_file(FILE).ok())
            .and_then(|bytes| serde_json::from_slice::<Kept>(&bytes).ok())
            .filter(|kept| kept.project == self.project);
        kept.map(|kept| kept.servers).unwrap_or_default()
    }

    /// Whether the server `name`, written as `server`, was approved
    /// (`Some(true)`) or denied (`Some(false)`) as it is written now.
    fn answer(&self, name: &str, server: &Value) -> Option<bool> {
        let answer = self.answers.get(name)?;
        (answer.server == *server).then_some(answer.approved)
    }

    /// Keeps that the server `name`, written as `server`, is `approved` or
    /// denied, over the answers the file holds now, so another rig's
    /// answers since stay.
    ///
    /// # Errors
    ///
    /// When the answers cannot be written.
    fn set(&mut self, name: &str, server: Value, approved: bool) -> Result<(), String> {
        let answer = Answer { approved, server };
        let Some(files) = self.files.clone() else {
            self.answers.insert(name.to_owned(), answer);
            return Ok(());
        };
        let _lock = lock(&self.lock)?;
        self.answers = self.kept();
        self.answers.insert(name.to_owned(), answer);
        let kept = Kept {
            project: self.project.clone(),
            servers: self.answers.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&kept).map_err(|error| error.to_string())?;
        files
            .write_file(FILE, &bytes)
            .map_err(|error| format!("could not keep the approval: {error}"))
    }
}

/// The lock file at `path`, locked until it is dropped.
fn lock(path: &Path) -> Result<File, String> {
    let failed = |error: std::io::Error| format!("could not lock {}: {error}", path.display());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(failed)?;
    }
    let lock = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(path)
        .map_err(failed)?;
    lock.lock().map_err(failed)?;
    Ok(lock)
}

/// Reads this project's answers, before the files' servers spawn.
fn read_approvals(world: &mut World) {
    let project = world.get_resource_or_init::<Project>().root.clone();
    let lock = RigHome::of(world).root().join("mcp-approvals.lock");
    let approvals = Approvals::read(world.get_resource::<PluginData>(), &project, lock);
    world.insert_resource(approvals);
}

/// The hold of the project's server `name`, written as `written`, that the
/// user denied or did not answer about.
fn hold(name: &str, written: &Value, denied: bool) -> McpHold {
    let was = if denied { "denied" } else { "not approved" };
    McpHold(format!(
        "{was}; /mcp approve {name} starts it as written: {written}"
    ))
}

/// Holds a server of the project's file that can start as it spawns,
/// unless approved as written, saying so unless the user denied it.
fn hold_unapproved(
    added: On<Add<McpServer>>,
    servers: Query<(&McpServer, &McpStatus)>,
    approvals: Option<Res<Approvals>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((
        McpServer {
            name,
            scope: Scope::Project,
            written,
        },
        status,
    )) = servers.get(added.entity)
    else {
        return;
    };
    // A disabled or invalid entry never starts anyway.
    if matches!(status.state, McpState::Disabled | McpState::Invalid) {
        return;
    }
    let answer = approvals.and_then(|approvals| approvals.answer(name, written));
    if answer == Some(true) {
        return;
    }
    if answer.is_none() {
        notices.write(Notice::warn(
            None,
            format!(
                "The project's MCP server `{name}` is not approved, so it did not start: \
                 {written}\n/mcp approve {name} starts it and remembers it; /mcp deny {name} \
                 stops asking."
            ),
        ));
    }
    let held = hold(name, written, answer == Some(false));
    commands.entity(added.entity).insert(held);
}

/// What `/mcp approve` and `/mcp deny` read and change.
#[derive(SystemParam)]
struct Answering<'w, 's> {
    servers: Query<'w, 's, (Entity, &'static McpServer)>,
    approvals: Option<ResMut<'w, Approvals>>,
    notices: MessageWriter<'w, Notice>,
    commands: Commands<'w, 's>,
}

impl Answering<'_, '_> {
    /// Keeps the answer about the project's server `args` names, as it is
    /// written now, and starts or holds it.
    fn answer(&mut self, args: &CommandArgs, approved: bool) {
        let done = self.keep(&args.args, approved);
        match done {
            Ok(text) => self.notices.write(Notice::info(args.agent, text)),
            Err(why) => self.notices.write(Notice::error(args.agent, why)),
        };
    }

    fn keep(&mut self, name: &str, approved: bool) -> Result<String, String> {
        let verb = if approved { "approve" } else { "deny" };
        if name.is_empty() || name.contains(char::is_whitespace) {
            return Err(format!("Use /mcp {verb} <name>."));
        }
        let server = (self.servers.iter())
            .find(|(_, server)| server.scope == Scope::Project && server.name == name);
        let Some((entity, server)) = server else {
            return Err(format!(
                "The project's .rig/mcp.json has no server `{name}`."
            ));
        };
        let Some(approvals) = self.approvals.as_mut() else {
            return Err("The approvals are not read yet.".to_owned());
        };
        let written = server.written.clone();
        approvals.set(name, written.clone(), approved)?;
        let mut entity = self.commands.entity(entity);
        if approved {
            entity.remove::<McpHold>();
            Ok(format!("Approved `{name}` as written: {written}."))
        } else {
            entity.insert(hold(name, &written, true));
            Ok(format!(
                "Denied `{name}`: it does not run until /mcp approve {name}."
            ))
        }
    }
}

/// `/mcp approve <name>`.
fn approve(In(args): In<CommandArgs>, mut answering: Answering) {
    answering.answer(&args, true);
}

/// `/mcp deny <name>`.
fn deny(In(args): In<CommandArgs>, mut answering: Answering) {
    answering.answer(&args, false);
}

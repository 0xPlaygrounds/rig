//! Who may answer an ask, and how an ask a parent may not answer escalates
//! up the tree.
//!
//! Every ask names its session and agent ([`OpenAsk`], and the `ask.opened`
//! it came from in [`AskDetails`]). Its session's [`Policy`] says who
//! answers:
//! - `any_owner` (a root's default): any owner client, the user;
//! - `driver`: only the client holding the session's driver lease (any owner
//!   while nobody drives);
//! - `parent` (a child's default): the parent session, for a question
//!   always, for an approval only when the parent's own up-front rules allow
//!   that call without asking. Anything else escalates ([`Escalate`]) up the
//!   tree to the first ancestor that may, else to the root's answerers. The
//!   owner may answer a child's ask too: the user is above every session.
//!
//! Up-front rules ([`rules_answer`]) answer a call before anyone is asked;
//! the worker applies its session's `grant.approve` with it. An ask nobody
//! answers is denied after `policy.timeout_s` (`ask.resolved {by:"timeout"}`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use bevy_ecs::prelude::*;
use rig_protocol::event::{AskKind, AskOpened};
use rig_protocol::{AgentId, Answerers, AskId, Rule};

use super::tree::{
    ParentSession, SessionGrant, SessionPolicy, default_policy, policy_of, unrestricted,
};
use super::{Driver, Principal, Session};
use crate::channel::Caller;

/// An approval or question waiting for an answer.
#[derive(Component, Debug)]
pub struct OpenAsk {
    pub id: AskId,
    pub session: Entity,
    pub agent: AgentId,
    pub deadline: Instant,
    pub answerers: Answerers,
}

/// On an ask: the `ask.opened` it came from, and the tool call an approval
/// is for (the record's `data.call`), for routing on what it asks.
#[derive(Component, Debug, Clone, PartialEq)]
pub struct AskDetails {
    pub opened: AskOpened,
    pub call: Option<AskCall>,
}

impl AskDetails {
    /// The details of an `ask.opened` record's `data`, or `None` when it
    /// isn't one.
    pub fn from_data(data: &serde_json::Value) -> Option<AskDetails> {
        Some(AskDetails {
            opened: serde_json::from_value(data.clone()).ok()?,
            call: data
                .get("call")
                .and_then(|call| serde_json::from_value(call.clone()).ok()),
        })
    }
}

/// The tool call an approval asks about.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AskCall {
    pub tool: String,
    #[serde(default)]
    pub args: serde_json::Value,
}

/// On an ask, once routed: the session whose answerers hold it. For
/// `Parent` answerers, that session itself answers (its worker, through
/// `answer_agent`); otherwise its owner clients or driver do.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct AskHolder(pub Entity);

/// Bubbles `ask` up the tree to whoever may answer it.
#[derive(EntityEvent, Debug, Clone, Copy)]
#[entity_event(propagate = &'static ParentSession, auto_propagate)]
pub struct Escalate {
    #[event_target]
    pub session: Entity,
    pub ask: Entity,
}

/// Whether `caller` may answer the ask entity `ask`.
pub fn may_answer(world: &World, ask: Entity, caller: &Caller) -> bool {
    let Some(open) = world.get::<OpenAsk>(ask) else {
        return false;
    };
    let holder = world.get::<AskHolder>(ask).map_or(open.session, |h| h.0);
    match &caller.principal {
        Principal::Owner => match open.answerers {
            Answerers::AnyOwner | Answerers::Parent => true,
            Answerers::Driver => world
                .get::<Driver>(holder)
                .is_none_or(|driver| driver.0 == caller.client),
        },
        Principal::Session(id) => {
            open.answerers == Answerers::Parent
                && holder != open.session
                && world.get::<Session>(holder).is_some_and(|s| &s.id == id)
        }
    }
}

/// Routes a newly opened `ask`: to its own session's answerers, or, under a
/// `parent` policy, up the tree ([`Escalate`]) to the first ancestor that
/// may answer it. Sets [`AskHolder`] and the ask's `answerers`.
pub fn on_ask(world: &mut World, ask: Entity) {
    let Some(session) = world.get::<OpenAsk>(ask).map(|a| a.session) else {
        return;
    };
    let policy = policy_of(world, session);
    let parent = world.get::<ParentSession>(session).map(|p| p.0);
    match (policy.answerers, parent) {
        (Answerers::Parent, Some(parent)) => {
            world.trigger(Escalate {
                session: parent,
                ask,
            });
            world.flush();
        }
        // Its own ask, under a `parent` policy with no parent: its owner answers.
        (Answerers::Parent, None) => hold(world, ask, session, Answerers::AnyOwner),
        (answerers, _) => hold(world, ask, session, answerers),
    }
}

/// Marks `holder` as holding `ask` for `answerers`: `Parent` means the
/// holder session itself answers it.
fn hold(world: &mut World, ask: Entity, holder: Entity, answerers: Answerers) {
    if let Some(mut open) = world.get_mut::<OpenAsk>(ask) {
        open.answerers = answerers;
    }
    world.entity_mut(ask).insert(AskHolder(holder));
}

/// One hop of an [`Escalate`], which Bevy carries up [`ParentSession`]: the
/// session it reached holds the ask when it may answer it, and so does a
/// session that isn't itself under a `parent` policy (a root), for its
/// answerers; otherwise the ask goes on up.
pub fn route_escalation(
    mut escalate: On<Escalate>,
    sessions: Query<(
        &Session,
        Option<&SessionGrant>,
        Option<&SessionPolicy>,
        Has<ParentSession>,
    )>,
    asks: Query<(&AskDetails, Has<AskHolder>)>,
    mut commands: Commands,
) {
    let (at, ask) = (escalate.event_target(), escalate.ask);
    let (Ok((details, held)), Ok((session, grant, policy, has_parent))) =
        (asks.get(ask), sessions.get(at))
    else {
        escalate.propagate(false);
        return;
    };
    if held {
        escalate.propagate(false);
        return;
    }
    let grant = grant.map_or_else(unrestricted, |g| g.0.clone());
    let answerers = policy
        .map_or_else(|| default_policy(has_parent), |p| p.0.clone())
        .answerers;
    let holds = if parent_may_answer(details, &grant, &session.cwd) {
        Answerers::Parent
    } else if answerers == Answerers::Parent && has_parent {
        return; // on up the tree
    } else if answerers == Answerers::Parent {
        // A root under a `parent` policy has none: its owner answers.
        Answerers::AnyOwner
    } else {
        answerers
    };
    escalate.propagate(false);
    commands.queue(move |world: &mut World| hold(world, ask, at, holds));
}

/// Whether a session with `grant` working in `cwd` may answer the ask
/// `details` of one of its descendants: a question always; an approval
/// when its own up-front rules allow the call without asking.
fn parent_may_answer(details: &AskDetails, grant: &rig_protocol::Grant, cwd: &Path) -> bool {
    match details.opened.kind {
        AskKind::Question => true,
        AskKind::Approval => details.call.as_ref().is_some_and(|call| {
            rules_answer(&grant.approve, cwd, &call.tool, &call.args) == Some(RuleAnswer::Allow)
        }),
    }
}

/// Whether a session whose grant asks about `ask` must ask before a call of
/// `tool` runs (unless an up-front rule answers it, [`rules_answer`]):
/// `"shell"` is the shell tool, `"edits"` every tool that changes files
/// ([`EDIT_TOOLS`] and `move`), and any other entry a tool's own name. An
/// empty `ask`, every session's by default, never asks.
pub fn asks_for(ask: &[String], tool: &str) -> bool {
    ask.iter().any(|kind| match kind.as_str() {
        "shell" => tool == "shell",
        "edits" => EDIT_TOOLS.contains(&tool) || tool == "move",
        name => name == tool,
    })
}

/// The open asks of `session`'s descendants that wait on `session`'s own
/// answerers (the owner, or its driver): escalated past every parent that
/// could answer them, so `session` needs an approval or an answer as if it
/// had asked itself.
pub fn asks_waiting_on(world: &World, session: Entity) -> Vec<Entity> {
    let Some(mut asks) = world.try_query::<(Entity, &OpenAsk, &AskHolder)>() else {
        return Vec::new();
    };
    asks.query(world)
        .iter()
        .filter(|(_, open, holder)| {
            holder.0 == session && open.session != session && open.answerers != Answerers::Parent
        })
        .map(|(ask, _, _)| ask)
        .collect()
}

/// What an up-front rule says about a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleAnswer {
    Allow,
    Deny,
}

/// The tools whose `path` argument an `edits_under` rule covers.
pub const EDIT_TOOLS: &[&str] = &["edit", "write", "delete"];

/// What `rules` answer for a call of `tool` with `args` in `cwd`, or `None`
/// when no rule covers it and someone must be asked. A `deny` wins over any
/// allow. `shell_prefix` matches a whole word prefix of a `shell` command
/// with no shell control characters (so `cargo test` never allows
/// `cargo test; rm -rf ~`). `edits_under` matches an edit tool's `path`
/// (and both ends of a `move`) under the rule's path, lexically, refusing
/// any `..`.
pub fn rules_answer(
    rules: &[Rule],
    cwd: &Path,
    tool: &str,
    args: &serde_json::Value,
) -> Option<RuleAnswer> {
    if rules
        .iter()
        .any(|r| matches!(r, Rule::Deny { name } if name == tool))
    {
        return Some(RuleAnswer::Deny);
    }
    let allowed = rules.iter().any(|rule| match rule {
        Rule::Deny { .. } => false,
        Rule::Tool { name } => name == tool,
        Rule::ShellPrefix { prefix } => {
            tool == "shell"
                && args
                    .get("command")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|command| shell_prefix(prefix, command))
        }
        Rule::EditsUnder { path } => {
            let under = |key: &str| {
                args.get(key)
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|p| super::tree::within(&absolute(cwd, p), &cwd.join(path)))
            };
            (EDIT_TOOLS.contains(&tool) && under("path"))
                || (tool == "move" && under("from") && under("to"))
        }
    });
    allowed.then_some(RuleAnswer::Allow)
}

fn absolute(cwd: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

/// Whether `command` is `prefix` or `prefix` followed by more words, with
/// nothing that could chain, substitute or redirect another command.
fn shell_prefix(prefix: &str, command: &str) -> bool {
    const CONTROL: &[char] = &[
        ';', '&', '|', '`', '$', '<', '>', '(', ')', '\n', '\r', '\\', '{', '}', '*', '?', '~', '!',
    ];
    let command = command.trim();
    let prefix = prefix.trim();
    !prefix.is_empty()
        && !command.contains(CONTROL)
        && (command == prefix
            || command
                .strip_prefix(prefix)
                .is_some_and(|rest| rest.starts_with(' ')))
}

#[cfg(test)]
mod tests;

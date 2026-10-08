//! Approval: whether a tool call may run, decided per agent before the
//! tool sees it. Each agent's [`Policy`] decides first; a call the policy
//! leaves to the user waits with an [`AwaitingApproval`] on its call
//! entity until something answers with [`Approve`]: the terminal view, a
//! GUI or a remote client, whichever the app has. The gate is a rig-core
//! [`Intercept`] layered around the tool's handler on the one dispatch
//! path, so a refused call never reaches its tool, the model gets the
//! reason as the call's error, and the effect log records the refusal.

use std::sync::{Mutex, PoisonError};

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use futures::channel::oneshot;
use rig_core::effect::{EffectId, EffectKind, Outcome};
use rig_core::error::ErrorReport;
use rig_core::serve::{Decision, Intercept, Verdict};
use rig_core::wasm_compat::WasmCompatSend;
use serde::{Deserialize, Serialize};

use super::agent::{Agent, CallOf, Notice, ToolCallRun, TurnOf};
use super::effects::Denials;
use super::save::ReflectSaved;
use super::subagents::SubagentOf;
use super::tools::Footprint;

/// The layer's name, as the effect log records it.
const LAYER: &str = "approval";

/// What happens to a tool call when no rule of the [`Policy`] says.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[serde(rename_all = "kebab-case")]
pub enum ApprovalMode {
    /// Every call runs: the agent works on its own, as pi does.
    #[default]
    Auto,
    /// Calls that only read run; any other call waits for the user.
    Ask,
    /// Calls that only read run; any other call is refused.
    ReadOnly,
}

impl ApprovalMode {
    /// Every mode, in the order `/approvals` lists them.
    pub const ALL: [Self; 3] = [Self::Auto, Self::Ask, Self::ReadOnly];

    /// What `/approvals` and `policy.json` call it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Ask => "ask",
            Self::ReadOnly => "read-only",
        }
    }

    /// The mode named `name`.
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.name() == name)
    }

    /// What a call that no rule matched gets: reads always run.
    fn default_for(self, reads: bool) -> Permission {
        match (self, reads) {
            (Self::Auto, _) | (_, true) => Permission::Allow,
            (Self::Ask, false) => Permission::Ask,
            (Self::ReadOnly, false) => Permission::Deny,
        }
    }
}

/// What a rule, or the mode, says about a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[serde(rename_all = "kebab-case")]
pub enum Permission {
    /// It runs.
    Allow,
    /// It waits for the user.
    Ask,
    /// It is refused.
    Deny,
}

/// One rule of a [`Policy`]: what calls of the tools `tool` names, with a
/// subject `subject` matches, get. Both are patterns where `*` matches any
/// text, such as `mcp__github__*` or `git status*`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Reflect)]
pub struct Rule {
    /// The tool's name, or a pattern of names.
    pub tool: String,
    /// A pattern the call's subject must match: the path for a tool that
    /// reads or writes one file, the `command` of a shell call, otherwise
    /// the arguments as compact JSON. `None` matches every call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    /// What a matching call gets.
    pub permission: Permission,
}

impl Rule {
    /// Whether the rule applies to a call of `tool` about `subject`.
    fn matches(&self, tool: &str, subject: &str) -> bool {
        wildcard(&self.tool, tool)
            && self
                .subject
                .as_deref()
                .is_none_or(|pattern| wildcard(pattern, subject))
    }

    /// The rule as `/approvals` lists it.
    pub fn label(&self) -> String {
        let permission = match self.permission {
            Permission::Allow => "allow",
            Permission::Ask => "ask",
            Permission::Deny => "deny",
        };
        match &self.subject {
            Some(subject) => format!("{permission} {} {subject}", self.tool),
            None => format!("{permission} {}", self.tool),
        }
    }
}

/// Whether an agent's tool calls run, wait for the user or are refused:
/// the last of its rules that matches a call decides, and the mode decides
/// when none does. A subagent starts with the policy of the agent that
/// started it, a fork with its origin's, and any other agent with the
/// app's [`DefaultPolicy`].
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Serialize, Deserialize, Saved)]
pub struct Policy {
    /// What calls no rule matches get.
    #[serde(default)]
    pub mode: ApprovalMode,
    /// The rules; of those matching a call, the last decides.
    #[serde(default)]
    pub rules: Vec<Rule>,
}

impl Policy {
    /// What a call of `tool` with `args` gets, given what the tool's calls
    /// touch.
    pub fn decide(
        &self,
        tool: &str,
        footprint: Footprint,
        args: &serde_json::Map<String, serde_json::Value>,
    ) -> Permission {
        let subject = subject(footprint, args);
        self.rules
            .iter()
            .rev()
            .find(|rule| rule.matches(tool, &subject))
            .map_or_else(
                || self.mode.default_for(only_reads(footprint)),
                |rule| rule.permission,
            )
    }
}

/// The policy an agent starts with when nothing else gives it one. The
/// host fills it from `RIG_HOME/policy.json`.
#[derive(Resource, Clone, Debug, Default)]
pub struct DefaultPolicy(pub Policy);

/// Whether calls with `footprint` change nothing: those that read files,
/// and those whose work is checked on its own (a `task`, whose subagent
/// has the policy too).
fn only_reads(footprint: Footprint) -> bool {
    matches!(footprint, Footprint::Reads { .. } | Footprint::Independent)
}

/// What a rule's `subject` pattern is matched against.
pub fn subject(footprint: Footprint, args: &serde_json::Map<String, serde_json::Value>) -> String {
    let text = |arg: &str| args.get(arg).and_then(serde_json::Value::as_str);
    let named = match footprint {
        Footprint::Reads { arg } | Footprint::Writes { arg } => text(arg),
        Footprint::Exclusive | Footprint::Independent => text("command"),
    };
    named.map_or_else(
        || serde_json::Value::Object(args.clone()).to_string(),
        str::to_owned,
    )
}

/// Whether `text` matches `pattern`, in which `*` matches any text and
/// everything else matches itself.
fn wildcard(pattern: &str, text: &str) -> bool {
    let mut parts = pattern.split('*');
    let first = parts.next().unwrap_or_default();
    let Some(mut rest) = text.strip_prefix(first) else {
        return false;
    };
    let parts: Vec<&str> = parts.collect();
    let Some((last, middle)) = parts.split_last() else {
        // No `*`: the whole text.
        return rest.is_empty();
    };
    for part in middle {
        let Some(found) = rest.find(part) else {
            return false;
        };
        rest = rest.get(found + part.len()..).unwrap_or_default();
    }
    rest.ends_with(last)
}

/// The user's answer to a call that waits for one.
#[derive(Clone, Debug, PartialEq, Eq, Reflect)]
pub enum ApprovalAnswer {
    /// Run this call.
    Allow,
    /// Run this call, and every later call of its tool by this agent.
    AllowAlways,
    /// Refuse it, telling the model `reason` when there is one.
    Deny {
        /// What to tell the model, such as what to do instead.
        reason: String,
    },
}

/// A tool call waiting for the user, on its call entity: what the views
/// show. Answer it with [`Approve`]; stopping the turn refuses it.
#[derive(Component)]
pub struct AwaitingApproval {
    /// The tool.
    pub tool: String,
    /// What the call is about: a path, a command or its arguments.
    pub subject: String,
    /// Where the answer goes; taken by the first answer.
    answer: Option<oneshot::Sender<ApprovalAnswer>>,
}

/// Answer the tool call `entity`, which has an [`AwaitingApproval`].
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Approve {
    /// The call.
    pub entity: Entity,
    /// The answer.
    pub answer: ApprovalAnswer,
}

/// What a tool call's gate lets through, settled when the call starts.
enum Gate {
    Allow,
    Deny(String),
    Ask(oneshot::Receiver<ApprovalAnswer>),
}

/// The approval layer of one tool call: a rig-core [`Intercept`] around
/// its tool's handler that lets the call through, refuses it, or waits
/// for the user's answer first.
pub(crate) struct Approval {
    gate: Mutex<Option<Gate>>,
    denials: Denials,
}

impl Approval {
    fn new(gate: Gate, denials: Denials) -> Self {
        Self {
            gate: Mutex::new(Some(gate)),
            denials,
        }
    }
}

impl Intercept for Approval {
    fn name(&self) -> String {
        LAYER.to_owned()
    }

    fn before(
        &self,
        id: EffectId,
        _kind: &EffectKind,
    ) -> impl Future<Output = Decision> + WasmCompatSend {
        let gate = self
            .gate
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        let denials = self.denials.clone();
        async move {
            let why = match gate {
                None | Some(Gate::Allow) => return Decision::Proceed,
                Some(Gate::Deny(why)) => why,
                Some(Gate::Ask(answer)) => match answer.await {
                    Ok(ApprovalAnswer::Allow | ApprovalAnswer::AllowAlways) => {
                        return Decision::Proceed;
                    }
                    Ok(ApprovalAnswer::Deny { reason }) => denied_by_user(&reason),
                    Err(_) => "The call was stopped before the user answered.".to_owned(),
                },
            };
            denials.note(id, why.clone());
            Decision::deny(why)
        }
    }

    async fn after(
        &self,
        _id: EffectId,
        _kind: &EffectKind,
        _outcome: &Result<Outcome, ErrorReport>,
    ) -> Verdict {
        Verdict::Keep
    }
}

/// What the model is told when the user refuses a call.
fn denied_by_user(reason: &str) -> String {
    let reason = reason.trim();
    if reason.is_empty() {
        "The user denied this call. Do not send it again; ask the user how to go on.".to_owned()
    } else {
        format!(
            "The user denied this call and said: {reason}\nDo what they say instead of sending \
             the call again."
        )
    }
}

/// How a tool call starts, as the agent's policy decides: its handler's
/// layer, and the [`AwaitingApproval`] to put on its call entity when it
/// waits for the user. `delegates` is true for a `task` call, which never
/// waits: its subagent asks for its own calls.
pub(crate) fn gate(
    policy: Option<&Policy>,
    run: &ToolCallRun,
    footprint: Footprint,
    delegates: bool,
    denials: Denials,
) -> (Approval, Option<AwaitingApproval>, Option<String>) {
    let tool = run.call.function.name.as_str();
    let args = &run.call.function.arguments;
    let permission = policy.map_or(Permission::Allow, |policy| {
        policy.decide(tool, footprint, args)
    });
    let (gate, waiting, refused) = match permission {
        Permission::Allow => (Gate::Allow, None, None),
        Permission::Ask if delegates => (Gate::Allow, None, None),
        Permission::Ask => {
            let (sender, receiver) = oneshot::channel();
            let waiting = AwaitingApproval {
                tool: tool.to_owned(),
                subject: subject(footprint, args),
                answer: Some(sender),
            };
            (Gate::Ask(receiver), Some(waiting), None)
        }
        Permission::Deny => {
            let why = format!(
                "The approval policy does not allow `{tool}` here ({}). Do not send it again; go \
                 on without it, or ask the user to allow it.",
                policy.map_or("", |policy| match policy.mode {
                    ApprovalMode::ReadOnly => "this agent is read-only",
                    ApprovalMode::Auto | ApprovalMode::Ask => "a rule denies it",
                })
            );
            (
                Gate::Deny(why),
                None,
                Some(format!("Denied `{tool}` by policy.")),
            )
        }
    };
    (Approval::new(gate, denials), waiting, refused)
}

/// Sends the answer to a call that waits for one. Allowing a tool always
/// adds a rule to its agent's policy; a refusal is told to the user too.
pub(crate) fn on_approve(
    approve: On<Approve>,
    mut calls: Query<(&mut AwaitingApproval, &CallOf)>,
    turns: Query<&TurnOf>,
    mut policies: Query<&mut Policy>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let call = approve.entity;
    let Ok((mut waiting, &CallOf(turn))) = calls.get_mut(call) else {
        return;
    };
    let Some(answer) = waiting.answer.take() else {
        return;
    };
    let tool = waiting.tool.clone();
    commands.entity(call).remove::<AwaitingApproval>();
    let agent = turns.get(turn).ok().map(|&TurnOf(agent)| agent);
    match &approve.answer {
        ApprovalAnswer::Allow => {}
        ApprovalAnswer::AllowAlways => {
            if let Some(mut policy) = agent.and_then(|agent| policies.get_mut(agent).ok()) {
                policy.rules.push(Rule {
                    tool: tool.clone(),
                    subject: None,
                    permission: Permission::Allow,
                });
            }
            notices.write(Notice::info(
                agent,
                format!("`{tool}` runs without asking from now on (/approvals lists the rules)."),
            ));
        }
        ApprovalAnswer::Deny { .. } => {
            notices.write(Notice::info(agent, format!("Denied `{tool}`.")));
        }
    }
    // The call may have been stopped meanwhile; then nobody waits.
    answer.send(approve.answer.clone()).ok();
}

/// Gives a new agent its starting policy, unless it came with one: a
/// subagent's is the agent's that started it, any other the app's
/// default.
pub(crate) fn give_policy(
    add: On<Add<Agent>>,
    agents: Query<(Has<Policy>, Option<&SubagentOf>)>,
    policies: Query<&Policy>,
    default: Option<Res<DefaultPolicy>>,
    mut commands: Commands,
) {
    let Ok((has, parent)) = agents.get(add.entity) else {
        return;
    };
    if has {
        return;
    }
    let policy = parent
        .and_then(|&SubagentOf(parent)| policies.get(parent).ok())
        .cloned()
        .or_else(|| default.map(|default| default.0.clone()))
        .unwrap_or_default();
    commands.entity(add.entity).insert(policy);
}

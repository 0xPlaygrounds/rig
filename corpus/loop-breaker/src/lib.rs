//! Plugin corpus: 08. Loop breaker (spec `08-loop-breaker.md`).
//!
//! Each turn counts its tool calls by fingerprint, the tool's name and its
//! arguments with their keys sorted. A call repeated [`LoopLimits::refuse_at`]
//! times in a turn is answered with an error in place of its result; one
//! repeated [`LoopLimits::stop_at`] times, or a turn past
//! [`LoopLimits::max_calls_per_turn`] calls, ends the turn before its next
//! model request, with a notice. `/loops off` turns it off for the session.
//!
//! The kernel starts an ordinary tool's call the moment the model's reply
//! is taken (`ToolStarter::spawn_calls` spawns the task before any observer
//! of the call's entity runs), so a refused call is not held back: its task
//! is cancelled and its result replaced as soon as its `Running` is
//! inserted. An open tool's call is not refused at all.

use std::collections::HashMap;

use rig_core::tool::{ToolExecutionError, ToolResult};
use rig_ecs::agent::Halt;
use rig_ecs::prelude::*;
use serde_json::{Map, Value};

#[cfg(test)]
mod tests;

/// How much repetition a turn may have. Insert it before adding the plugin
/// to change it.
#[derive(Resource, Clone, Copy, Debug)]
pub struct LoopLimits {
    /// The identical call that is refused, and every later one: the third.
    pub refuse_at: u32,
    /// The identical call that ends the turn: the fifth.
    pub stop_at: u32,
    /// The tool calls a turn may make before it is ended.
    pub max_calls_per_turn: u32,
}

impl Default for LoopLimits {
    fn default() -> Self {
        Self {
            refuse_at: 3,
            stop_at: 5,
            max_calls_per_turn: 100,
        }
    }
}

/// Whether loops are broken this session; `/loops off` turns it off. Kept
/// across `/reload` and `/resume`.
#[derive(Resource, Reflect, Clone, Copy, Debug)]
#[reflect(Resource, Saved)]
pub struct LoopBreaking {
    /// On unless turned off.
    pub enabled: bool,
}

impl Default for LoopBreaking {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// A turn's tool calls so far, on the turn's entity: it goes with the turn.
#[derive(Component, Default, Debug)]
pub struct TurnCalls {
    /// Every tool call of the turn.
    pub total: u32,
    /// The calls of each fingerprint.
    pub seen: HashMap<String, u32>,
    /// Why the turn is ended before its next model request.
    pub stop: Option<String>,
}

/// A tool call's result in place of the tool's, decided when the call was
/// made.
#[derive(Component, Clone, Debug)]
struct Refused(String);

/// Refuses a tool call repeated in a turn, and ends a turn that loops.
#[derive(Default)]
pub struct LoopBreakerPlugin;

impl Plugin for LoopBreakerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LoopLimits>()
            .init_resource::<LoopBreaking>()
            .register_required_components::<TurnOf, TurnCalls>()
            .add_observer(judge)
            .add_observer(refuse)
            .add_observer(end_stopped)
            .add_command(
                "loops",
                "Show the loop breaker, or turn it `on` or `off` for the session",
                loops,
            );
    }
}

/// Counts each tool call of a turn as its entity is spawned, in the reply's
/// order, and decides which are refused and which end the turn.
fn judge(
    call: On<Add<ToolCallRun>>,
    calls: Query<(&ToolCallRun, &CallOf)>,
    mut turns: Query<&mut TurnCalls>,
    limits: Res<LoopLimits>,
    breaking: Res<LoopBreaking>,
    mut commands: Commands,
) {
    if !breaking.enabled {
        return;
    }
    // A plugin's call outside a turn has no `CallOf`.
    let Ok((run, CallOf(turn))) = calls.get(call.entity) else {
        return;
    };
    let Ok(mut counts) = turns.get_mut(*turn) else {
        return;
    };
    let name = run.call.function.name.as_str();
    let arguments = &run.call.function.arguments;
    counts.total += 1;
    let seen = counts.seen.entry(fingerprint(name, arguments)).or_default();
    *seen += 1;
    let seen = *seen;
    let stop = if counts.total > limits.max_calls_per_turn {
        Some(format!(
            "stopped: the agent made more than {} tool calls in one turn",
            limits.max_calls_per_turn
        ))
    } else if seen >= limits.stop_at {
        Some(format!(
            "stopped: the agent was repeating `{}`",
            describe(name, arguments)
        ))
    } else {
        None
    };
    let refusal = match (stop, &counts.stop) {
        (_, Some(stopping)) => stopping.clone(),
        (Some(stop), None) => {
            counts.stop = Some(stop.clone());
            stop
        }
        (None, None) if seen >= limits.refuse_at => {
            let before = match seen - 1 {
                2 => "twice".to_owned(),
                times => format!("{times} times"),
            };
            format!(
                "You already called `{name}` with these arguments {before} with the same \
                 result; change approach."
            )
        }
        (None, None) => return,
    };
    commands.entity(call.entity).insert(Refused(refusal));
}

/// Answers a refused call once it starts, which cancels the tool's task.
fn refuse(
    started: On<Add<Running>>,
    refused: Query<&Refused, With<ToolCallRun>>,
    mut commands: Commands,
) {
    if let Ok(Refused(why)) = refused.get(started.entity) {
        let result = ToolResult::failed(ToolExecutionError::refused(why.clone()));
        commands
            .entity(started.entity)
            .insert_if_new(ToolOutput(result));
    }
}

/// Ends a turn that is to stop in place of its next model request, once
/// every call of the reply has its result in the conversation. Despawning
/// the turn before the request is sent sends none.
fn end_stopped(
    prepare: On<PrepareRequest>,
    turns: Query<&TurnCalls>,
    mut agents: Query<&mut Conversation>,
    mut commit: Commit,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok(TurnCalls {
        stop: Some(why), ..
    }) = turns.get(prepare.entity)
    else {
        return;
    };
    // The conversation ends in tool results the model owes an answer to: a
    // restore must not send them again.
    if let Ok(mut conversation) = agents.get_mut(prepare.agent) {
        commit.halt(prepare.agent, &mut conversation, Halt::Kept);
    }
    notices.write(Notice::error(prepare.agent, why.clone()));
    commands.entity(prepare.entity).despawn();
}

/// `/loops`, `/loops on`, `/loops off`.
fn loops(
    In(args): In<CommandArgs>,
    mut breaking: ResMut<LoopBreaking>,
    limits: Res<LoopLimits>,
    mut notices: MessageWriter<Notice>,
) {
    let text = match args.args.as_str() {
        "on" => {
            breaking.enabled = true;
            "Loop breaker on.".to_owned()
        }
        "off" => {
            breaking.enabled = false;
            "Loop breaker off for this session.".to_owned()
        }
        "" if breaking.enabled => format!(
            "Loop breaker on: the {} identical call of a turn is refused, the {} ends the \
             turn, and so does a turn past {} tool calls.",
            ordinal(limits.refuse_at),
            ordinal(limits.stop_at),
            limits.max_calls_per_turn
        ),
        "" => "Loop breaker off.".to_owned(),
        _ => {
            notices.write(Notice::error(args.agent, "/loops takes `on` or `off`."));
            return;
        }
    };
    notices.write(Notice::info(args.agent, text));
}

/// A call's identity: its tool and its arguments with their keys sorted,
/// whatever order the model wrote them in.
fn fingerprint(name: &str, arguments: &Map<String, Value>) -> String {
    format!("{name} {}", sorted(&Value::Object(arguments.clone())))
}

/// `value` with every object's keys in order.
fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let map = entries
                .into_iter()
                .map(|(key, value)| (key.clone(), sorted(value)));
            Value::Object(map.collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

/// A call for a person: `read src/lib.rs`, the tool and its argument
/// values, shortened.
fn describe(name: &str, arguments: &Map<String, Value>) -> String {
    const MAX: usize = 60;
    let mut text = name.to_owned();
    for value in arguments.values() {
        text.push(' ');
        match value {
            Value::String(text_value) => text.push_str(text_value),
            other => text.push_str(&other.to_string()),
        }
    }
    match text.char_indices().nth(MAX) {
        Some((cut, _)) => format!("{}…", text.get(..cut).unwrap_or_default()),
        None => text,
    }
}

/// `3` as "3rd", for the status line of `/loops`.
fn ordinal(n: u32) -> String {
    let suffix = match (n % 10, n % 100) {
        (_, 11..=13) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    };
    format!("{n}{suffix}")
}

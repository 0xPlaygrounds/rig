//! Plugin corpus: 08. Loop breaker (spec `08-loop-breaker.md`).
//!
//! A Gate system counts each turn's tool calls by fingerprint, the tool's
//! name and its arguments, as they are about to run, in the reply's order.
//! The [`LoopLimits::refuse_at`]th identical call of a turn, and every later
//! one, is refused; the [`LoopLimits::stop_at`]th, or a turn past
//! [`LoopLimits::max_calls_per_turn`] calls, stops the turn with a notice.
//! `/loops off` turns it off for the session.

use rig_harness::prelude::*;
use serde::Deserialize;
use serde_json::{Map, Value};

#[cfg(test)]
mod tests;

/// How much repetition a turn may have: the `[loops]` table of the user's
/// settings.
#[derive(Resource, Deserialize, Clone, Copy, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct LoopLimits {
    /// The identical call that is refused, and every later one: the third.
    pub refuse_at: u32,
    /// The identical call that stops the turn: the fifth.
    pub stop_at: u32,
    /// The tool calls a turn may make before it is stopped.
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
    /// Each distinct call, its tool and arguments, and how often it was
    /// made. Arguments compare as JSON values, whatever their keys' order.
    pub seen: Vec<(String, Map<String, Value>, u32)>,
}

impl TurnCalls {
    /// Counts a call of `name` with `arguments`: how often it was made in
    /// the turn, this one included.
    fn count(&mut self, name: &str, arguments: &Map<String, Value>) -> u32 {
        self.total += 1;
        let same = |(seen, args, _): &&mut (String, Map<String, Value>, u32)| {
            seen == name && args == arguments
        };
        match self.seen.iter_mut().find(same) {
            Some((_, _, times)) => {
                *times += 1;
                *times
            }
            None => {
                self.seen.push((name.to_owned(), arguments.clone(), 1));
                1
            }
        }
    }
}

/// The loop breaker's Gate system, which refuses calls and stops turns,
/// for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoopBreakerSystems;

/// On a tool call the plugin counted: it is counted once.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Counted;

/// Refuses a tool call repeated in a turn, and stops a turn that loops.
#[derive(Default)]
pub struct LoopBreakerPlugin;

impl Plugin for LoopBreakerPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<LoopLimits>("loops")
            .init_resource::<LoopBreaking>()
            .register_required_components::<TurnOf, TurnCalls>()
            .configure_sets(AgentPass, LoopBreakerSystems.in_set(Stage::Gate))
            .add_systems(AgentPass, judge.in_set(LoopBreakerSystems))
            .add_command(
                "loops",
                "Show the loop breaker, or turn it `on` or `off` for the session",
                loops,
            );
    }
}

/// Counts each tool call of a running turn when it is about to run, in the
/// reply's order; refuses the repeated ones, and stops a turn that loops.
fn judge(
    mut turns: Query<(&TurnOf, &Calls, &mut TurnCalls), Without<TurnOutcome>>,
    mut calls: Query<(&ToolRequest, &mut Decision), (ReadyToStart, ToolCalls, Without<Counted>)>,
    limits: Res<LoopLimits>,
    breaking: Res<LoopBreaking>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if !breaking.enabled {
        return;
    }
    for (&TurnOf(agent), of, mut counts) in &mut turns {
        for call in of.iter() {
            let Ok((request, mut decision)) = calls.get_mut(call) else {
                continue;
            };
            commands.entity(call).insert(Counted);
            let function = &request.call.function;
            let (name, arguments) = (function.name.as_str(), &function.arguments);
            let times = counts.count(name, arguments);
            let stop = if counts.total > limits.max_calls_per_turn {
                Some(format!(
                    "the agent made more than {} tool calls in one turn",
                    limits.max_calls_per_turn
                ))
            } else if times >= limits.stop_at {
                Some(format!(
                    "the agent was repeating `{}`",
                    describe(name, arguments)
                ))
            } else {
                None
            };
            if let Some(why) = stop {
                // The call and the rest of the reply never start; they are
                // answered as stopped, and the turn ends.
                notices.write(Notice::error(agent, format!("stopped: {why}")));
                commands.trigger(StopTurn::plugin(agent, "loop-breaker", &why).alone());
                break;
            }
            if times >= limits.refuse_at {
                let before = match times - 1 {
                    2 => "twice".to_owned(),
                    times => format!("{times} times"),
                };
                let why = format!(
                    "You already called `{name}` with these arguments {before} with the same \
                     result; change approach."
                );
                decision.decide(Decider::plugin("loop-breaker"), Outcome::refused(why));
            }
        }
    }
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
            "Loop breaker on: a call repeated {} times in a turn is refused, {} times stops \
             the turn, and so do more than {} calls.",
            limits.refuse_at, limits.stop_at, limits.max_calls_per_turn
        ),
        "" => "Loop breaker off.".to_owned(),
        _ => {
            notices.write(Notice::error(args.agent, "/loops takes `on` or `off`."));
            return;
        }
    };
    notices.write(Notice::info(args.agent, text));
}

/// A call for a person: `read src/lib.rs`, the tool and its argument
/// values, shortened.
fn describe(name: &str, arguments: &Map<String, Value>) -> String {
    const MAX: usize = 60;
    let mut text = name.to_owned();
    for value in arguments.values() {
        text.push(' ');
        match value {
            Value::String(value) => text.push_str(value),
            other => text.push_str(&other.to_string()),
        }
    }
    match text.char_indices().nth(MAX) {
        Some((cut, _)) => format!("{}…", text.get(..cut).unwrap_or_default()),
        None => text,
    }
}

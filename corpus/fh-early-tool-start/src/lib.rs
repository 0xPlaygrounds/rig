//! FH-24: start read-only tool calls while the reply that asks for them is
//! still streaming.
//!
//! As a turn's reply streams, each tool call it ends ([`StreamDelta`]'s
//! `End`) of a tool whose [`Footprint`] only reads, with arguments that
//! parsed, starts at once as a call of its own, [`StartedEarly`]: every
//! Prepare and Gate system sees it before it starts. One that a gate holds
//! or decides, or whose own footprint writes, is withdrawn before it
//! starts, with the early calls after it; so is every one after a call
//! that may not start early, such as a `write`, which may change what a
//! later read sees. Once started, its outcome is held from Review until
//! the reply ends.
//!
//! When the reply ends, its own calls go through the gates as usual. In
//! [`GateSystems::After`], each one the gates let through that an early
//! call answers (same id, same request as Prepare left it) is decided with
//! the early call's outcome, held until it has one, and the early call is
//! withdrawn: the conversation gets the reply's calls, in the reply's
//! order, as if they ran after it. A failed, cut off or interrupted
//! reply, or one that did not ask for the call after all, withdraws it.
//!
//! `/early off` and `/early on` switch it, `/early stats` says the tool
//! time saved per turn. The running turn's status line counts its early
//! calls, and a reply's call that waits for its early call says so.

use std::collections::{BTreeMap, HashSet};
use std::time::SystemTime;

use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectId;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// Who decides a reply's call with its early call's outcome.
const NAME: &str = "early-tools";

/// The hold on an early call's outcome until its reply ended.
const WAITS_FOR_REPLY: &str = "started early: waits for the reply to end";

/// The hold on a reply's call until its early call answered.
const RUNS_EARLY: &str = "⚡ started early";

/// The `[early_tools]` settings.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct EarlyTools {
    /// Start calls that only read while the reply streams; `/early off`
    /// clears it.
    pub enabled: bool,
}

impl Default for EarlyTools {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// On a tool call this plugin started before its reply ended.
#[derive(Component, Debug)]
pub struct StartedEarly {
    /// The model call that streamed it.
    pub model: Entity,
    /// That call's id, the parent of its reply's calls.
    pub reply: EffectId,
    /// The reply's own call it answers, once the reply ended.
    pub twin: Option<Entity>,
    /// When the ended reply asked for it.
    pub asked: Option<SystemTime>,
    /// When it was answered.
    pub finished: Option<SystemTime>,
}

/// On a model call whose reply starts no more calls early.
#[derive(Component, Debug, Default)]
#[component(storage = "SparseSet")]
struct Closed;

/// Tool time saved by early starts, per turn index, kept with the session.
#[derive(Component, Reflect, Default, Debug)]
#[reflect(Component, Saved)]
pub struct EarlyStats {
    /// Milliseconds of tool time that ran before the reply ended, by turn.
    pub saved_ms: BTreeMap<u64, u64>,
    /// How many calls ran early.
    pub calls: u64,
}

/// On a running turn: how many of its calls started early.
#[derive(Component, Debug, Default)]
pub struct EarlyCount(pub u32);

/// The plugin's stage systems, after every gate.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EarlySystems;

/// The plugin.
#[derive(Default)]
pub struct EarlyToolStartPlugin;

impl Plugin for EarlyToolStartPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<EarlyTools>("early_tools")
            .register_required_components::<Agent, EarlyStats>()
            .configure_sets(AgentPass, EarlySystems.in_set(GateSystems::After))
            .add_systems(
                AgentPass,
                (withdraw_unstarted, adopt, sweep)
                    .chain()
                    .in_set(EarlySystems),
            )
            .add_systems(PostUpdate, start_early)
            .add_observer(hold_outcome)
            .add_observer(answered)
            .add_command("early", "Early tool starts: /early on|off|stats", early);
        #[cfg(feature = "tui")]
        app.add_systems(
            PostUpdate,
            status::show_count.in_set(rig_tui::StatusSystems::Show),
        );
    }
}

/// Starts each tool call a still streaming turn reply ended, when its tool
/// only reads and its arguments parsed; the first one that may not ends
/// early starts for that reply. It reads what streamed after the loop,
/// since a delta moves no stage, and asks for the frame the calls start in.
fn start_early(
    mut deltas: MessageReader<StreamDelta>,
    replies: Query<
        (&Effect, &CallOf, &ForAgent),
        (
            ModelCalls,
            With<TurnCall>,
            Without<Outcome>,
            Without<Settled>,
            Without<Closed>,
        ),
    >,
    tools: Query<(&ToolDef, &Footprint)>,
    settings: Res<EarlyTools>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let mut closed = HashSet::new();
    for delta in deltas.read() {
        let StreamEvent::End {
            content: AssistantContent::ToolCall(call),
            ..
        } = &delta.event
        else {
            continue;
        };
        let Ok((effect, &CallOf(turn), &ForAgent(agent))) = replies.get(delta.effect) else {
            continue;
        };
        if closed.contains(&delta.effect) {
            continue;
        }
        let name = call.function.name.as_str();
        let reads = tools
            .iter()
            .any(|(def, footprint)| def.0.name.as_str() == name && footprint.0.reads_only());
        if !(settings.enabled && reads && call.function.invalid_arguments.is_none()) {
            commands.entity(delta.effect).try_insert(Closed);
            closed.insert(delta.effect);
            continue;
        }
        commands.spawn((
            ToolRequest { call: call.clone() },
            Effect::with_parent_id(None),
            CallOf(turn),
            ForAgent(agent),
            StartedEarly {
                model: delta.effect,
                reply: effect.id(),
                twin: None,
                asked: None,
                finished: None,
            },
        ));
        wake.wake();
    }
}

/// Withdraws each early call about to start that a gate held (it would
/// wait on someone) or decided, or whose own footprint writes, with the
/// early calls after it, and starts no more for its reply: the reply's
/// call then goes through the gates after the reply.
fn withdraw_unstarted(
    unstarted: Query<
        (
            &StartedEarly,
            &Effect,
            &Decision,
            Option<&Footprint>,
            Has<Holds>,
        ),
        (With<Admitted>, Without<Started>, Without<Settled>),
    >,
    early: Query<(Entity, &StartedEarly, &Effect), Without<Settled>>,
    mut commands: Commands,
) {
    for (from, effect, decision, footprint, held) in &unstarted {
        let reads = footprint.is_some_and(|footprint| footprint.0.reads_only());
        if reads && !held && !decision.is_decided() {
            continue;
        }
        commands.entity(from.model).try_insert(Closed);
        for (call, of, later) in &early {
            if of.model == from.model && later.id() >= effect.id() {
                commands.entity(call).try_despawn();
            }
        }
    }
}

/// Answers each call of an ended reply that the gates let through with its
/// early call's outcome and what it published, holding it until there is
/// one; a call the gates decided withdraws its early call.
fn adopt(
    mut asked: Query<
        (Entity, &ToolRequest, &Effect, &mut Decision, Option<&Holds>),
        (
            ToolCalls,
            With<Admitted>,
            Without<StartedEarly>,
            Without<Waiting>,
            Without<Started>,
            Without<Outcome>,
            Without<Settled>,
        ),
    >,
    mut early: Query<
        (
            Entity,
            &mut StartedEarly,
            &ToolRequest,
            (Option<&Outcome>, Option<&Published>),
            Option<&Started>,
            &CallOf,
            &ForAgent,
        ),
        Without<Settled>,
    >,
    ours: Query<(), With<WaitsForEarly>>,
    turns: Query<&Turn>,
    mut stats: Query<&mut EarlyStats>,
    mut commands: Commands,
) {
    for (twin, request, effect, mut decision, holds) in &mut asked {
        let found = early.iter_mut().find(|(_, early, early_request, ..)| {
            Some(early.reply) == effect.parent
                && early.twin.is_none_or(|matched| matched == twin)
                && early_request.call == request.call
        });
        let Some((
            call,
            mut started_early,
            _,
            (outcome, published),
            started,
            &CallOf(turn),
            &ForAgent(agent),
        )) = found
        else {
            continue;
        };
        started_early.twin = Some(twin);
        let asked_at = *started_early.asked.get_or_insert_with(SystemTime::now);
        if decision.is_decided() {
            commands.entity(call).try_despawn();
            continue;
        }
        let mut held = holds.into_iter().flat_map(Holds::iter);
        if held.any(|hold| !ours.contains(hold)) {
            continue;
        }
        match outcome {
            Some(outcome) => {
                decision.decide(Decider::plugin(NAME), outcome.clone());
                // With the facts it published, such as the files it read.
                if let Some(published) = published {
                    commands.entity(twin).try_insert(published.clone());
                }
                // The hold on the reply's call goes with it.
                commands.entity(call).try_despawn();
                let ran = started
                    .zip(started_early.finished)
                    .map(|(started, finished)| {
                        finished.min(asked_at).duration_since(started.at).ok()
                    });
                if let (Some(Some(saved)), Ok(turn), Ok(mut stats)) =
                    (ran, turns.get(turn), stats.get_mut(agent))
                {
                    let millis = u64::try_from(saved.as_millis()).unwrap_or(u64::MAX);
                    *stats.saved_ms.entry(turn.index).or_default() += millis;
                    stats.calls += 1;
                }
            }
            None if holds.is_none() => {
                commands.spawn((
                    WaitsForEarly,
                    Name::new(RUNS_EARLY),
                    HoldOf(twin),
                    ChildOf(call),
                ));
            }
            None => {}
        }
    }
}

/// The hold on a reply's call until its early call answered, a child of
/// the early call, so it goes with it.
#[derive(Component)]
struct WaitsForEarly;

/// Withdraws the early calls of a reply that ended (or failed, was cut
/// off or interrupted) whose reply did not ask for them, or whose reply's
/// call was answered otherwise.
fn sweep(
    early: Query<(Entity, &StartedEarly), Without<Settled>>,
    streaming: Query<(), (ModelCalls, Without<Settled>)>,
    pending: Query<(), (Without<Outcome>, Without<Started>, Without<Settled>)>,
    mut commands: Commands,
) {
    for (call, early) in &early {
        let waits = early.twin.is_some_and(|twin| pending.contains(twin));
        if !streaming.contains(early.model) && !waits {
            commands.entity(call).try_despawn();
        }
    }
}

/// Holds the outcome of an early call as it starts, so Review sees it only
/// as the reply's call's, and counts it on the turn.
fn hold_outcome(
    started: On<Add<Started>>,
    early: Query<&CallOf, With<StartedEarly>>,
    mut counts: Query<&mut EarlyCount>,
    mut commands: Commands,
) {
    let Ok(&CallOf(turn)) = early.get(started.entity) else {
        return;
    };
    commands.spawn((Name::new(WAITS_FOR_REPLY), HoldOf(started.entity)));
    match counts.get_mut(turn) {
        Ok(mut count) => count.0 += 1,
        Err(_) => {
            commands.entity(turn).try_insert(EarlyCount(1));
        }
    }
}

/// Notes when an early call was answered.
fn answered(answered: On<Insert<Outcome>>, mut early: Query<&mut StartedEarly>) {
    if let Ok(mut early) = early.get_mut(answered.entity) {
        early.finished.get_or_insert_with(SystemTime::now);
    }
}

#[cfg(feature = "tui")]
mod status {
    use rig_harness::prelude::*;
    use rig_tui::{Side, StatusItem, StatusItems, Tone};

    use super::EarlyCount;

    /// The running turn's count of early calls: after the status.
    const EARLY: StatusItem = StatusItem::new("fh-early-tool-start", Side::Left, 56, 9);

    /// Shows the running turn's count of early calls.
    pub(super) fn show_count(
        mut turns: Query<(&EarlyCount, &mut StatusItems), Changed<EarlyCount>>,
    ) {
        for (EarlyCount(count), mut items) in &mut turns {
            items.show(EARLY.says(format!("⚡{count} early"), Tone::Dim));
        }
    }
}

/// `/early on|off|stats`.
fn early(
    In(args): In<CommandArgs>,
    mut settings: ResMut<EarlyTools>,
    stats: Query<&EarlyStats>,
    mut notices: MessageWriter<Notice>,
) {
    let text = match args.args.trim() {
        "on" => {
            settings.enabled = true;
            "Early tool starts are on.".to_owned()
        }
        "off" => {
            settings.enabled = false;
            "Early tool starts are off.".to_owned()
        }
        "stats" | "" => {
            let state = if settings.enabled { "on" } else { "off" };
            let Ok(stats) = stats.get(args.agent) else {
                return;
            };
            let total: u64 = stats.saved_ms.values().sum();
            let turns: Vec<String> = stats
                .saved_ms
                .iter()
                .rev()
                .take(5)
                .map(|(turn, ms)| format!("turn {turn}: {ms} ms"))
                .collect();
            format!(
                "Early tool starts are {state}: {} calls, {total} ms of tool time saved. {}",
                stats.calls,
                turns.join(", ")
            )
        }
        other => {
            notices.write(Notice::error(
                args.agent,
                format!("/early takes on, off or stats, not `{other}`"),
            ));
            return;
        }
    };
    notices.write(Notice::info(args.agent, text));
}

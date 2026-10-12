//! FH-24: start read-only tool calls while the reply that asks for them is
//! still streaming.
//!
//! Every turn's model call gets an outermost [`Adapter`] whose stream tap
//! sees each tool call the moment the provider ends it and hands it to the
//! app through a [`Running::stream`]. A call of a tool safe to start early
//! ([`EarlyTools::tools`], or one a plugin marks [`StartsEarly`]) whose
//! arguments parsed is spawned at once as an ordinary tool call of the
//! turn, [`StartedEarly`], so every Prepare and Gate system sees it before
//! it starts. Once it started, a hold keeps its outcome from settling until
//! the reply ends. When the reply ends and the kernel spawns the reply's
//! own calls, each early call takes the place of its twin (`Supersedes`),
//! so its result goes into the conversation where the model asked for it.
//! A failed, cut off or interrupted reply, or a reply that did not ask for
//! the call after all, withdraws it before it settled.
//!
//! The first call that is not started early, such as a `write`, ends early
//! starts for that reply, so the early calls are always the first calls of
//! the reply and nothing runs before a call the model wrote ahead of it. A
//! call a gate holds or decides before it starts is withdrawn too, with
//! its question, and runs after the reply as usual.
//!
//! `/early off` and `/early on` switch it, `/early stats` says the tool
//! time saved per turn. The running turn's status line counts its early
//! calls.

use std::collections::BTreeMap;
use std::time::SystemTime;

use bevy_tasks::{IoTaskPool, TaskPool};
use futures::channel::mpsc;
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectId;
use rig_harness::rig_core::streaming::{Item, Relayed, StreamEvent};
use rig_harness::rig_ecs::effects::Handler;
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The adapter's and the stream rewrite's name.
const NAME: &str = "early-tools";

/// What the hold on an early call that finished before its reply says.
const HOLD: &str = "started early: waits for the reply to end";

/// The running turn's count of early calls: after the status.
const EARLY: StatusItem = StatusItem::at(Side::Left, 56, 9);

/// The `[early_tools]` settings: whether early starts are on and the
/// tools whose calls may start early.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct EarlyTools {
    /// Start safe calls while the reply streams; `/early off` clears it.
    pub enabled: bool,
    /// The names of the tools safe to start early, besides those marked
    /// [`StartsEarly`].
    pub tools: Vec<String>,
}

impl Default for EarlyTools {
    fn default() -> Self {
        let tools = ["read", "search", "list", "ls", "glob", "grep"];
        Self {
            enabled: true,
            tools: tools.map(str::to_owned).to_vec(),
        }
    }
}

/// On a tool's entity: its calls only read, so they may start while the
/// reply that asks for them still streams.
#[derive(Component, Debug, Default)]
pub struct StartsEarly;

/// On a tool call this plugin started before its reply ended.
#[derive(Component, Debug)]
pub struct StartedEarly {
    /// The call as the model streamed it, before any Prepare edit.
    pub asked: ToolCall,
    /// When the ended reply asked for it, which makes it the reply's call.
    pub adopted: Option<SystemTime>,
    /// When it was answered.
    pub finished: Option<SystemTime>,
    /// The feed of the reply it came from.
    feed: Entity,
    /// Whether it started before the reply ended.
    ran_early: bool,
}

/// Tool time saved by early starts, per turn index, kept with the session.
#[derive(Component, Reflect, Default, Debug)]
#[reflect(Component, Saved)]
pub struct EarlyStats {
    /// Milliseconds of tool time that ran before the reply ended, by turn.
    pub saved_ms: BTreeMap<u64, u64>,
    /// How many calls started early.
    pub calls: u64,
}

/// On a running turn: how many of its calls started early.
#[derive(Component, Debug, Default)]
pub struct EarlyCount(pub u32);

/// The plugin's Prepare systems, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EarlySystems;

/// The plugin.
#[derive(Default)]
pub struct EarlyToolStartPlugin;

impl Plugin for EarlyToolStartPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<EarlyTools>("early_tools")
            .register_required_components::<Agent, EarlyStats>()
            .configure_sets(AgentPass, EarlySystems.in_set(PrepareSystems::Edit))
            .add_systems(AgentPass, (tap, sweep).in_set(EarlySystems))
            .add_systems(PostUpdate, show_count.in_set(StatusSystems))
            .add_observer(start_early)
            .add_observer(hold_until_asked)
            .add_observer(withdraw_held)
            .add_observer(answered)
            .add_observer(adopt)
            .add_observer(count_saved)
            .add_command("early", "Early tool starts: /early on|off|stats", early);
    }
}

/// A tool call the reply's stream ended.
struct Completed(ToolCall);

/// A turn's model call whose stream is tapped: its early calls, in order.
#[derive(Component)]
struct Feed {
    /// The model call.
    call: Entity,
    /// Its effect id, the parent of the calls of its reply.
    reply: EffectId,
    agent: Entity,
    turn: Entity,
    /// Whether the next call may still start early.
    open: bool,
    started: Vec<Entity>,
}

/// The hold on an early call that keeps its outcome until the reply ended.
#[derive(Component)]
struct ReplyHold;

/// Taps each new turn call's stream for the tool calls it ends.
fn tap(
    mut calls: Query<
        (Entity, &Effect, &CallOf, &ForAgent, &mut Adapters),
        (Added<Admitted>, ModelCalls, With<TurnCall>),
    >,
    settings: Res<EarlyTools>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    if !settings.enabled {
        return;
    }
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    for (call, effect, &CallOf(turn), &ForAgent(agent), mut adapters) in &mut calls {
        let (sender, completed) = mpsc::unbounded();
        let adapter = Adapter::new(NAME, move |mut connection: Connection| {
            let sender = sender.clone();
            let tap = move |_| {
                let sender = sender.clone();
                move |relayed: Relayed| {
                    if let Relayed::Item(Item::Event(StreamEvent::End {
                        content: AssistantContent::ToolCall(call),
                        ..
                    })) = &relayed
                    {
                        sender.unbounded_send(Completed(call.clone())).ok();
                    }
                    vec![relayed]
                }
            };
            connection.handler = Handler(connection.handler.0.map_stream(NAME, tap));
            connection
        });
        adapters.0.push(adapter.outermost());
        let feed = Feed {
            call,
            reply: effect.id(),
            agent,
            turn,
            open: true,
            started: Vec::new(),
        };
        commands.spawn((
            Name::new("early tool starts"),
            feed,
            Running::stream(pool, &wake, completed),
        ));
    }
}

/// Starts a completed call of a still streaming reply when it is safe;
/// the first one that is not ends early starts for the reply.
fn start_early(
    done: On<Insert<Done<Completed>>>,
    mut feeds: Query<(&mut Feed, &Done<Completed>)>,
    streaming: Query<(), (ModelCalls, Without<Outcome>, Without<Settled>)>,
    tools: Query<(&ToolDef, Has<StartsEarly>)>,
    settings: Res<EarlyTools>,
    mut commands: Commands,
) {
    let Ok((mut feed, Done(Completed(call)))) = feeds.get_mut(done.entity) else {
        return;
    };
    if !feed.open {
        return;
    }
    let name = call.function.name.as_str();
    let marked = || {
        tools
            .iter()
            .any(|(def, marked)| marked && def.0.name.as_str() == name)
    };
    let safe = settings.tools.iter().any(|tool| tool == name) || marked();
    let valid = call.function.invalid_arguments.is_none();
    if !(settings.enabled && safe && valid && streaming.contains(feed.call)) {
        feed.open = false;
        return;
    }
    let early = StartedEarly {
        asked: call.clone(),
        adopted: None,
        finished: None,
        feed: done.entity,
        ran_early: false,
    };
    let spawned = commands
        .spawn((
            ToolRequest { call: call.clone() },
            Effect::child_of(Some(feed.reply)),
            CallOf(feed.turn),
            ForAgent(feed.agent),
            early,
        ))
        .id();
    feed.started.push(spawned);
}

/// Holds the outcome of an early call that started before its reply ended,
/// and counts it on the turn.
fn hold_until_asked(
    started: On<Add<Started>>,
    mut early: Query<(&mut StartedEarly, &CallOf)>,
    mut counts: Query<&mut EarlyCount>,
    mut commands: Commands,
) {
    let Ok((mut early, &CallOf(turn))) = early.get_mut(started.entity) else {
        return;
    };
    if early.adopted.is_some() {
        return;
    }
    early.ran_early = true;
    commands.spawn((ReplyHold, Name::new(HOLD), HoldOf(started.entity)));
    match counts.get_mut(turn) {
        Ok(mut count) => count.0 += 1,
        Err(_) => {
            commands.entity(turn).try_insert(EarlyCount(1));
        }
    }
}

/// Withdraws an early call held before it started, such as for a
/// question: it would wait on someone, so it runs after the reply.
fn withdraw_held(
    held: On<Add<Holds>>,
    early: Query<&StartedEarly, Without<Started>>,
    mut feeds: Query<&mut Feed>,
    mut commands: Commands,
) {
    if let Ok(early) = early.get(held.entity)
        && early.adopted.is_none()
        && let Ok(mut feed) = feeds.get_mut(early.feed)
    {
        withdraw(&mut commands, &mut feed, held.entity);
    }
}

/// Notes when an early call was answered; withdraws one answered before it
/// started, such as refused by a gate, which then judges the reply's call.
fn answered(
    answered: On<Insert<Outcome>>,
    mut early: Query<(&mut StartedEarly, Has<Started>)>,
    mut feeds: Query<&mut Feed>,
    mut commands: Commands,
) {
    let Ok((mut early, started)) = early.get_mut(answered.entity) else {
        return;
    };
    if started {
        early.finished.get_or_insert_with(SystemTime::now);
    } else if early.adopted.is_none()
        && let Ok(mut feed) = feeds.get_mut(early.feed)
    {
        withdraw(&mut commands, &mut feed, answered.entity);
    }
}

/// Withdraws `call` and the early calls after it, and starts no more for
/// the reply: the early calls stay the first calls of the reply.
fn withdraw(commands: &mut Commands, feed: &mut Feed, call: Entity) {
    feed.open = false;
    let Some(at) = feed.started.iter().position(|&early| early == call) else {
        return;
    };
    for early in feed.started.drain(at..) {
        commands.entity(early).try_despawn();
    }
}

/// Makes the early call the reply's call once the ended reply asks for
/// it, with the same arguments: it supersedes the kernel's twin, and its
/// outcome may settle.
fn adopt(
    asked: On<Add<ToolRequest>>,
    calls: Query<(&ToolRequest, &Effect), Without<StartedEarly>>,
    feeds: Query<&Feed>,
    mut early: Query<(&mut StartedEarly, Option<&Holds>)>,
    ours: Query<(), With<ReplyHold>>,
    mut commands: Commands,
) {
    let Ok((request, effect)) = calls.get(asked.entity) else {
        return;
    };
    let Some(feed) = effect
        .parent
        .and_then(|parent| feeds.iter().find(|feed| feed.reply == parent))
    else {
        return;
    };
    for &candidate in &feed.started {
        let Ok((mut twin, holds)) = early.get_mut(candidate) else {
            continue;
        };
        if twin.adopted.is_some() || twin.asked != request.call {
            continue;
        }
        twin.adopted = Some(SystemTime::now());
        commands
            .entity(candidate)
            .try_insert(Supersedes(asked.entity));
        for hold in holds.into_iter().flat_map(Holds::iter) {
            if ours.contains(hold) {
                commands.entity(hold).try_despawn();
            }
        }
        return;
    }
}

/// Once a tapped model call settled, its reply's calls were spawned and
/// adopted: withdraws the early calls it did not ask for, all of them for
/// a reply that failed, was cut off or interrupted, and ends the tap.
fn sweep(
    feeds: Query<(Entity, &Feed)>,
    calls: Query<Has<Settled>, ModelCalls>,
    early: Query<&StartedEarly>,
    mut commands: Commands,
) {
    for (entity, feed) in &feeds {
        if !calls.get(feed.call).unwrap_or(true) {
            continue;
        }
        let unasked = |call: &Entity| early.get(*call).is_ok_and(|early| early.adopted.is_none());
        if let Some(at) = feed.started.iter().position(unasked) {
            for &call in feed.started.iter().skip(at) {
                commands.entity(call).try_despawn();
            }
        }
        commands.entity(entity).try_despawn();
    }
}

/// Adds the tool time an adopted early call ran before its reply ended to
/// its agent's [`EarlyStats`].
fn count_saved(
    settled: On<Add<Settled>>,
    early: Query<(&StartedEarly, &Started, &CallOf, &ForAgent)>,
    turns: Query<&Turn>,
    mut stats: Query<&mut EarlyStats>,
) {
    let Ok((early, started, &CallOf(turn), &ForAgent(agent))) = early.get(settled.entity) else {
        return;
    };
    let (Some(ended), Some(finished), true) = (early.adopted, early.finished, early.ran_early)
    else {
        return;
    };
    let (Ok(turn), Ok(mut stats)) = (turns.get(turn), stats.get_mut(agent)) else {
        return;
    };
    let saved = finished
        .min(ended)
        .duration_since(started.at)
        .unwrap_or_default();
    let millis = u64::try_from(saved.as_millis()).unwrap_or(u64::MAX);
    *stats.saved_ms.entry(turn.index).or_default() += millis;
    stats.calls += 1;
}

/// Shows the running turn's count of early calls.
fn show_count(mut turns: Query<(&EarlyCount, &mut StatusItems), Changed<EarlyCount>>) {
    for (EarlyCount(count), mut items) in &mut turns {
        items.show(EARLY.says(format!("⚡{count} early"), Tone::Dim));
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

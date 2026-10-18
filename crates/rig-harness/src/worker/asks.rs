//! Questions through the channel: the worker is an [`Answerer`] for every
//! agent, so plugins ask as they would with a terminal. Each open [`Ask`]
//! becomes `ask.opened`; the session's up-front rules answer an approval
//! before anyone is asked (`ask.resolved` by `rule`); an `Answer` command
//! decides it; and one nobody answers is denied once its deadline passes
//! (`ask.resolved` by `timeout`).

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bevy_ecs::prelude::*;
use rig_core::effect::Decider;
use rig_ecs::agent::{AgentId, Agent};
use rig_ecs::stage::{Decision, ForAgent, HoldOf, Outcome, Settled};
use rig_ecs::tools::ToolRequest;
use rig_protocol::event::{AskKind, AskOpened, AskOption, AskResolved};
use rig_protocol::methods::{self, WorkerEvent};
use rig_protocol::{AgentId as WireAgent, Answerers, AskId, Event};

use super::{WorkerConfig, WorkerOutbox};
use crate::ask::{Answer, Ask};
use crate::clock::{Due, WakeAt};
use crate::server::policy::{RuleAnswer, rules_answer};

/// How long an ask stays open when nobody answers it.
pub const ASK_TIMEOUT: Duration = Duration::from_secs(600);

/// On an [`Ask`] the channel was told of: its id.
#[derive(Component, Clone, Debug)]
pub(crate) struct Opened(pub(crate) AskId);

/// On an opened ask whose resolution the channel was told already, such
/// as one a command answered.
#[derive(Component)]
pub(crate) struct Told;

/// The open asks by id.
#[derive(Resource, Default)]
pub(crate) struct AskIds {
    open: HashMap<AskId, Entity>,
    next: u64,
}

fn publish(outbox: &mut WorkerOutbox, event: Event) {
    outbox.events.push(WorkerEvent {
        kind: event.kind().to_owned(),
        data: event.encode(),
        source: None,
    });
}

/// Tells the channel of each new question, after the session's rules had
/// their say on an approval.
#[allow(clippy::too_many_arguments)]
pub(crate) fn open_asks(
    mut asks: Query<
        (Entity, &Ask, Option<&ForAgent>, Option<&HoldOf>, &mut Decision),
        (Without<Opened>, Without<Outcome>, Without<Settled>),
    >,
    calls: Query<&ToolRequest>,
    agents: Query<(&AgentId, Option<&Name>), With<Agent>>,
    config: Res<WorkerConfig>,
    mut ids: ResMut<AskIds>,
    mut outbox: ResMut<WorkerOutbox>,
    mut commands: Commands,
) {
    for (entity, ask, agent, hold, mut decision) in &mut asks {
        if decision.is_decided() {
            continue;
        }
        // An approval holds the call it is about and offers to allow it; a
        // question a tool asks, such as `ask_user`, holds its call too.
        let approves = choice_like(ask, &["allow"]).is_some();
        let call = hold
            .and_then(|&HoldOf(effect)| calls.get(effect).ok())
            .filter(|_| approves);
        ids.next += 1;
        let id = AskId(format!("a-{}", ids.next));
        if let Some(request) = call {
            let tool = request.call.function.name.as_str();
            let args = serde_json::Value::Object(request.call.function.arguments.clone());
            let rules = config.grant.as_ref().map(|g| g.approve.as_slice());
            if let Some(answer) = rules.and_then(|rules| rules_answer(rules, &config.cwd, tool, &args))
            {
                let (choice, outcome) = match answer {
                    RuleAnswer::Allow => {
                        let allow = choice_like(ask, &["allow", "yes"]);
                        let outcome = allow
                            .clone()
                            .map(|id| Answer::choice(id).outcome())
                            .unwrap_or_else(|| Outcome::refused("No choice allows it."));
                        (allow, outcome)
                    }
                    RuleAnswer::Deny => (
                        choice_like(ask, &["deny", "no"]),
                        Outcome::refused("The session's rules deny this."),
                    ),
                };
                decision.decide(Decider::plugin("rig-worker"), outcome);
                publish(
                    &mut outbox,
                    AskResolved {
                        ask_id: id.clone(),
                        by: "rule".into(),
                        option_id: choice,
                        text: None,
                    }
                    .into(),
                );
                commands.entity(entity).insert((Opened(id), Told));
                continue;
            }
        }
        let deadline = SystemTime::now() + config.ask_timeout;
        let (agent_id, agent_name) = agent
            .and_then(|&ForAgent(agent)| agents.get(agent).ok())
            .map(|(id, name)| {
                let name = name.map_or_else(|| id.0.clone(), |name| name.to_string());
                (id.0.clone(), name)
            })
            .unwrap_or_default();
        let opened = AskOpened {
            ask_id: id.clone(),
            agent_id: WireAgent(agent_id),
            agent_name,
            kind: match call {
                Some(_) => AskKind::Approval,
                None => AskKind::Question,
            },
            title: ask.title.clone(),
            body: ask.body.clone(),
            options: ask
                .choices
                .iter()
                .map(|choice| AskOption {
                    option_id: choice.id.clone(),
                    label: choice.label.clone(),
                    kind: option_kind(&choice.id).into(),
                })
                .collect(),
            deadline_ms: deadline
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)),
            answerers: Answerers::AnyOwner,
        };
        let mut data = Event::from(opened).encode();
        if let (Some(request), Some(map)) = (call, data.as_object_mut()) {
            map.insert(
                "call".into(),
                serde_json::json!({
                    "tool": request.call.function.name.as_str(),
                    "args": request.call.function.arguments,
                }),
            );
        }
        outbox.events.push(WorkerEvent {
            kind: "ask.opened".into(),
            data,
            source: None,
        });
        ids.open.insert(id.clone(), entity);
        commands
            .entity(entity)
            .insert((Opened(id), WakeAt(deadline)));
    }
}

/// The first choice whose id is one of `ids`.
fn choice_like(ask: &Ask, ids: &[&str]) -> Option<String> {
    ask.choices
        .iter()
        .find(|choice| ids.contains(&choice.id.as_str()))
        .map(|choice| choice.id.clone())
}

/// ACP's kind of a choice, from its id.
fn option_kind(id: &str) -> &'static str {
    match id {
        "allow" | "yes" => "allow_once",
        "deny" | "no" => "reject_once",
        _ => "other",
    }
}

/// Answers the ask `answer` names, as the command says: its option, its
/// text, or with neither, denied (`rigd`'s own deadline). An ask no longer
/// open is left alone: `rigd` refuses a second answer itself.
pub(crate) fn answer(world: &mut World, answer: &methods::Answer) {
    let Some(entity) = world.resource::<AskIds>().open.get(&answer.ask_id).copied() else {
        bevy_log::warn!("no open ask {:?} to answer", answer.ask_id.0);
        return;
    };
    let outcome = match (&answer.option_id, &answer.text) {
        (Some(option), text) => {
            let chosen = Answer::choice(option.clone());
            match text {
                Some(text) => chosen.with_text(text.clone()).outcome(),
                None => chosen.outcome(),
            }
        }
        (None, Some(text)) => Answer::text(text.clone()).outcome(),
        (None, None) => Outcome::refused("Nobody answered in time, so it was denied."),
    };
    let Ok(mut ask) = world.get_entity_mut(entity) else {
        return;
    };
    if let Some(mut decision) = ask.get_mut::<Decision>()
        && decision.decide(Decider::Person, outcome)
    {
        ask.insert(Told);
    }
}

/// Denies each opened ask whose deadline passed.
pub(crate) fn expire(
    due: On<Add<Due>>,
    mut asks: Query<(&Opened, &mut Decision), (Without<Outcome>, Without<Told>)>,
    mut outbox: ResMut<WorkerOutbox>,
    mut commands: Commands,
) {
    let Ok((Opened(id), mut decision)) = asks.get_mut(due.entity) else {
        return;
    };
    if decision.decide(
        Decider::plugin("rig-worker"),
        Outcome::refused("Nobody answered in time, so it was denied."),
    ) {
        commands.entity(due.entity).insert(Told);
        publish(
            &mut outbox,
            AskResolved {
                ask_id: id.clone(),
                by: "timeout".into(),
                option_id: None,
                text: None,
            }
            .into(),
        );
    }
}

/// Forgets each opened ask once it is answered, telling the channel of
/// one that closed some other way, such as its turn stopped.
pub(crate) fn close_asks(
    closed: Query<(Entity, &Opened, Has<Told>), Or<(Added<Outcome>, Added<Settled>)>>,
    mut ids: ResMut<AskIds>,
    mut outbox: ResMut<WorkerOutbox>,
    mut gone: RemovedComponents<Opened>,
) {
    for (_, Opened(id), by_channel) in &closed {
        if ids.open.remove(id).is_some() && !by_channel {
            publish(
                &mut outbox,
                AskResolved {
                    ask_id: id.clone(),
                    by: "worker".into(),
                    option_id: None,
                    text: None,
                }
                .into(),
            );
        }
    }
    for entity in gone.read() {
        let stale: Vec<AskId> = ids
            .open
            .iter()
            .filter(|(_, open)| **open == entity)
            .map(|(id, _)| id.clone())
            .collect();
        for id in stale {
            ids.open.remove(&id);
            publish(
                &mut outbox,
                AskResolved {
                    ask_id: id,
                    by: "worker".into(),
                    option_id: None,
                    text: None,
                }
                .into(),
            );
        }
    }
}

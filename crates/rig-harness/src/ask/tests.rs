use bevy_app::TaskPoolPlugin;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::effect::{Answered as EffectOutcome, Decider, Disposition, EffectKind};
use rig_ecs::agent::{Agent, Conversation};
use rig_ecs::inbox::{Deliver, DeliveryMode, Incoming};
use rig_ecs::session_log::SessionStore;
use rig_ecs::stage::{EffectRecorded, ForAgent, Outcome, Started};
use rig_memory::MemoryStore;

use super::{Answer, Answerer, Answerers, Ask, AskPlugin, Choice};
use crate::commands::CommandPlugin;
use crate::pick::{Chosen, Pick, PickPlugin};

/// The kernel's records, frame after frame.
#[derive(Resource, Default)]
struct Records(Vec<rig_core::effect::EffectRecord>);

/// A kernel app with the question plugin, slash commands and its first
/// agent.
fn app() -> (App, Entity) {
    let mut app = App::new();
    let harness = (AskPlugin, CommandPlugin, PickPlugin);
    app.add_plugins((TaskPoolPlugin::default(), rig_ecs::AgentPlugin, harness))
        .insert_resource(SessionStore::new(MemoryStore::default()))
        .init_resource::<Records>()
        .add_systems(
            Last,
            |mut recorded: MessageReader<EffectRecorded>, mut records: ResMut<Records>| {
                records
                    .0
                    .extend(recorded.read().map(|recorded| recorded.0.clone()));
            },
        );
    app.update();
    let mut agents = app.world_mut().query_filtered::<Entity, With<Agent>>();
    let agent = agents
        .iter(app.world())
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    (app, agent)
}

#[test]
fn a_question_is_an_open_effect_recorded_with_its_answer() {
    let (mut app, agent) = app();
    let ask = Ask::yes_no("Keep it?")
        .body("3 files")
        .choice("later", "Ask me later");
    let question = app.world_mut().spawn((ask.clone(), ForAgent(agent))).id();
    app.update();
    assert!(app.world().entity(question).contains::<Started>());
    let answer = Answer::choice("later");
    app.world_mut()
        .entity_mut(question)
        .insert(answer.outcome());
    app.update();
    let records = &app.world().resource::<Records>().0;
    let record = records
        .iter()
        .find(|record| matches!(&record.kind, EffectKind::Custom { kind, .. } if &**kind == "ask"));
    assert!(record.is_some(), "no record of the question: {records:?}");
    let Some(record) = record else {
        return;
    };
    assert_eq!(record.disposition, Disposition::Ran);
    assert_eq!(Answer::of(&Outcome(record.outcome.clone())), Some(answer));
    let EffectKind::Custom { payload, .. } = &record.kind else {
        return;
    };
    assert_eq!(
        serde_json::from_value::<Ask>(payload.clone()).ok(),
        Some(ask)
    );
    assert_eq!(Answer::of(&Outcome::refused("no")), None);
}

#[test]
fn the_users_next_line_answers_a_question_that_takes_text() {
    let (mut app, agent) = app();
    // One beside the work, though older, leaves the user's lines to the
    // agent.
    let inline = Ask::new("Note it?").initial("cache").inline();
    let beside = app.world_mut().spawn((inline, ForAgent(agent))).id();
    let question = app
        .world_mut()
        .spawn((Ask::new("Why deny it?").free_text(), ForAgent(agent)))
        .id();
    app.update();
    app.world_mut().trigger(Deliver::new(
        agent,
        "it deletes the cache",
        DeliveryMode::Steer,
    ));
    app.world_mut()
        .trigger(Deliver::new(agent, "/help", DeliveryMode::Steer));
    app.update();
    let outcome = app.world().get::<Outcome>(question);
    assert_eq!(
        outcome.and_then(Answer::of),
        Some(Answer::text("it deletes the cache"))
    );
    assert!(app.world().get::<Outcome>(beside).is_none());
    // The answer never reached the model's conversation, and no turn started.
    let conversation = app.world().get::<Conversation>(agent);
    assert!(conversation.is_some_and(|conversation| conversation.messages().is_empty()));
    let mut messages = app.world_mut().query::<(&Incoming, &Outcome)>();
    let answered: Vec<_> = messages
        .iter(app.world())
        .filter(|(_, outcome)| {
            matches!(&outcome.0, Ok(EffectOutcome::Custom { payload }) if payload == "answer")
        })
        .map(|(incoming, _)| incoming.text().to_owned())
        .collect();
    assert_eq!(answered, ["it deletes the cache"]);
}

#[test]
fn a_questions_record_names_who_answered_it() {
    let (mut app, agent) = app();
    let world = app.world_mut();
    let asked = world.spawn((Ask::yes_no("Run?"), ForAgent(agent))).id();
    let auto = world.spawn((Ask::yes_no("Write?"), ForAgent(agent))).id();
    app.update();
    let world = app.world_mut();
    world.trigger(Pick::person(asked, Some(Chosen::item(0))));
    world.trigger(Pick {
        entity: auto,
        chosen: Some(Chosen::item(1)),
        by: Decider::plugin("auto-approve"),
    });
    app.update();
    let records = &app.world().resource::<Records>().0;
    let answered_by = |title: &str| {
        let record = records.iter().find(|record| match &record.kind {
            EffectKind::Custom { payload, .. } => payload["title"] == title,
            _ => false,
        });
        record.map(|record| record.decided_by.clone())
    };
    assert_eq!(answered_by("Run?"), Some(Some(Decider::Person)));
    assert_eq!(
        answered_by("Write?"),
        Some(Some(Decider::plugin("auto-approve"))),
        "{records:?}"
    );
}

#[test]
fn a_front_answers_for_its_agent_or_for_all() {
    let mut app = App::new();
    let one = app.world_mut().spawn_empty().id();
    let other = app.world_mut().spawn_empty().id();
    let can = |app: &mut App, agent| {
        app.world_mut()
            .run_system_cached_with(
                |In(agent): In<Option<Entity>>, answerers: Answerers| answerers.can_answer(agent),
                agent,
            )
            .unwrap_or(false)
    };
    assert!(!can(&mut app, Some(one)));
    app.world_mut().spawn(Answerer { agent: Some(one) });
    assert!(can(&mut app, Some(one)) && !can(&mut app, Some(other)) && !can(&mut app, None));
    app.world_mut().spawn(Answerer::default());
    assert!(can(&mut app, Some(other)) && can(&mut app, None));
}

/// A choice that starts ticked and says more round-trips through serde, and
/// a choice saved before either existed still reads.
#[test]
fn a_choice_keeps_its_tick_and_more_through_serde() {
    let choice = Choice::new("1", "write a.rs")
        .checked()
        .more("{\"path\":\"a.rs\"}");
    let saved = serde_json::to_value(&choice).ok();
    let read = saved.and_then(|value| serde_json::from_value::<Choice>(value).ok());
    assert_eq!(read, Some(choice));
    let old = serde_json::json!({ "id": "yes", "label": "Yes" });
    assert_eq!(
        serde_json::from_value::<Choice>(old).ok(),
        Some(Choice::new("yes", "Yes"))
    );
}

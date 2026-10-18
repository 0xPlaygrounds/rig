use bevy_app::TaskPoolPlugin;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::effect::EffectKind;
use rig_ecs::agent::Agent;
use rig_ecs::session_log::SessionStore;
use rig_ecs::stage::{
    AgentPass, EffectRequest, ForAgent, Outcome, PluginEffects, ReadyToStart, Stage, Started,
};
use rig_memory::MemoryStore;

use super::{ALLOW, ApprovalPlugin, Approved, NeedsApproval};
use rig_harness::ask::{Answer, Answerer, Ask, AskPlugin};

/// A Gate system of the plugin `by`, which wants every plugin effect
/// approved.
fn wants(by: &'static str) -> impl FnMut(Query<Entity, (ReadyToStart, PluginEffects)>, Commands) {
    move |effects, mut commands| {
        for effect in &effects {
            commands
                .entity(effect)
                .queue(NeedsApproval::add(by, format!("{by} wants it.")));
        }
    }
}

/// A kernel app with questions, approval and two plugins that want every
/// plugin effect approved, a front that answers when `answering`, and a
/// `deploy` effect of its first agent.
fn start(answering: bool) -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins((
        TaskPoolPlugin::default(),
        rig_ecs::AgentPlugin,
        AskPlugin,
        ApprovalPlugin,
    ))
    .insert_resource(SessionStore::new(MemoryStore::default()))
    .add_systems(
        AgentPass,
        (wants("guard"), wants("audit")).in_set(Stage::Gate),
    );
    if answering {
        app.world_mut().spawn(Answerer::default());
    }
    app.update();
    let mut agents = app.world_mut().query_filtered::<Entity, With<Agent>>();
    let agent = agents
        .iter(app.world())
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    let request = EffectRequest {
        kind: EffectKind::Custom {
            kind: "deploy".into(),
            payload: serde_json::Value::Null,
        },
        handler: None,
    };
    let effect = app.world_mut().spawn((request, ForAgent(agent))).id();
    app.update();
    (app, effect)
}

/// The open questions.
fn asks(app: &mut App) -> Vec<(Entity, Ask)> {
    let mut asks = app
        .world_mut()
        .query_filtered::<(Entity, &Ask), Without<Outcome>>();
    asks.iter(app.world())
        .map(|(entity, ask)| (entity, ask.clone()))
        .collect()
}

/// The refusal an effect was answered with, if it was.
fn refusal(app: &App, effect: Entity) -> Option<String> {
    let outcome = app.world().get::<Outcome>(effect)?;
    outcome
        .0
        .as_ref()
        .err()
        .map(|report| report.message.clone())
}

#[test]
fn two_plugins_reasons_make_one_question_and_allowing_it_with_a_note_starts_the_effect() {
    let (mut app, effect) = start(true);
    let open = asks(&mut app);
    assert_eq!(open.len(), 1, "one question: {open:?}");
    let Some((question, ask)) = open.first() else {
        return;
    };
    assert_eq!(ask.title, "Allow `deploy`?");
    assert!(
        ask.body.contains("guard wants it.\naudit wants it."),
        "{}",
        ask.body
    );
    assert!(!app.world().entity(effect).contains::<Started>(), "held");
    // A note on the choice still allows it.
    let allow = Answer::choice(ALLOW).with_text("just this once").outcome();
    app.world_mut().entity_mut(*question).insert(allow);
    app.update();
    let world = app.world();
    assert!(world.entity(effect).contains::<Approved>());
    assert!(world.entity(effect).contains::<Started>());
    assert!(asks(&mut app).is_empty(), "not asked again");
}

#[test]
fn a_typed_line_denies_with_its_reason_and_with_nobody_to_ask_it_is_refused_at_once() {
    let (mut app, effect) = start(true);
    let open = asks(&mut app);
    assert!(!open.is_empty(), "a question");
    let Some((question, _)) = open.first() else {
        return;
    };
    let typed = Answer::text("not on a Friday").outcome();
    app.world_mut().entity_mut(*question).insert(typed);
    app.update();
    assert_eq!(
        refusal(&app, effect).as_deref(),
        Some("The user denied this: not on a Friday")
    );

    let (mut app, effect) = start(false);
    assert!(asks(&mut app).is_empty());
    assert_eq!(
        refusal(&app, effect).as_deref(),
        Some("Needs the user's approval, and nobody can answer: guard wants it. audit wants it.")
    );
}

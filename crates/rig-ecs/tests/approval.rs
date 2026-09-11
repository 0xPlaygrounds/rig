//! Approval decisions govern dispatch while unrelated runs keep progressing.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test assertions"
)]

#[path = "../examples/support/mod.rs"]
mod examples;
mod run_support;

use bevy_app::{App, Update};
use bevy_ecs::prelude::*;
use rig_core::{
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    message::AssistantContent,
    observe::Emitter,
    serve::{Dispatch, Reply, Serve},
    tool::{ToolOutput, ToolResult},
};
use rig_ecs::{
    agent::{Failed, RunResult, ToolCallSlot},
    approval::{
        ApprovalChoice, ApprovalError, ApprovalRequest, ApprovalRequired, DecisionOutcome, decide,
    },
    bus::{
        BusSet, EffectOutcome, Handlers, Held, PendingEffect, RigSchedule, acquire_hold,
        release_hold, run_to_quiescence,
    },
    commands::{Agent, Prompt, install},
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Tool(Arc<AtomicUsize>);
impl Serve for Tool {
    type Family = rig_core::effect::family::Tool;
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("tool"),
            family: FamilyDescriptor::Tool {
                name: "write".into(),
                description: "count a write".into(),
                parameters: serde_json::json!({}),
                embedding: None,
            },
            layers: vec![],
        }
    }
    async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
        self.0.fetch_add(1, Ordering::SeqCst);
        Reply::Outcome(Ok(Outcome::ToolResult {
            result: ToolResult::success(ToolOutput::text("written")),
        }))
    }
}

fn require(
    calls: Query<Entity, (Added<PendingEffect>, With<ToolCallSlot>)>,
    mut commands: Commands,
) {
    for call in &calls {
        commands
            .entity(call)
            .insert(ApprovalRequired(Emitter::named("app/reviewer")));
    }
}

fn setup() -> (App, Entity, Arc<AtomicUsize>) {
    let mut app = App::new();
    install(app.world_mut(), Default::default()).unwrap();
    app.add_systems(Update, run_to_quiescence)
        .add_systems(RigSchedule, require.in_set(BusSet::Gate));
    let calls = Arc::new(AtomicUsize::new(0));
    let model = Handlers::register_in(
        app.world_mut(),
        "model",
        examples::Scripted::new(vec![
            vec![examples::call("write", serde_json::json!({"value": 1}))],
            vec![AssistantContent::text("finished")],
        ]),
    )
    .unwrap();
    let tool = Handlers::register_in(app.world_mut(), "tool", Tool(calls.clone())).unwrap();
    let agent = Agent::new(model)
        .tools([tool])
        .max_turns(2)
        .spawn(app.world_mut())
        .unwrap();
    let run = Prompt::new(agent, "write").spawn(app.world_mut()).unwrap();
    (app, run, calls)
}

fn pending(app: &mut App) -> ApprovalRequest {
    run_support::tick_until(app, "pending approval", |world| {
        world
            .query::<&ApprovalRequest>()
            .iter(world)
            .any(|r| r.is_pending())
    });
    app.world_mut()
        .query::<&ApprovalRequest>()
        .iter(app.world())
        .next()
        .unwrap()
        .clone()
}

#[test]
fn another_run_finishes_while_approval_waits_and_other_holds_survive() {
    let (mut app, run, calls) = setup();
    let request = pending(&mut app);
    assert_eq!(request.run(), run);
    assert_eq!(request.name(), "write");
    assert_eq!(request.args(), "{\"value\":1}");
    let (model, _) = run_support::Capturing::new("other", "independent");
    let model = Handlers::register_in(app.world_mut(), "other", model).unwrap();
    let agent = Agent::new(model).spawn(app.world_mut()).unwrap();
    let other = Prompt::new(agent, "hello").spawn(app.world_mut()).unwrap();
    run_support::tick_until(&mut app, "unrelated result", |w| {
        w.get::<RunResult>(other).is_some()
    });
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(app.world().get::<RunResult>(run).is_none());
    let effect = request.ticket().effect();
    acquire_hold(app.world_mut(), effect, Emitter::named("app/other-policy"));
    assert_eq!(
        decide(app.world_mut(), request.ticket(), ApprovalChoice::Approve),
        Ok(DecisionOutcome::Applied)
    );
    assert_eq!(
        decide(app.world_mut(), request.ticket(), ApprovalChoice::Approve),
        Ok(DecisionOutcome::AlreadyApplied)
    );
    assert_eq!(
        decide(
            app.world_mut(),
            request.ticket(),
            ApprovalChoice::Deny("late".into())
        ),
        Err(ApprovalError::AlreadyDecided)
    );
    app.update();
    assert!(app.world().get::<Held>(effect).is_some());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    release_hold(app.world_mut(), effect, "app/other-policy");
    run_support::tick_until(&mut app, "approved run result", |w| {
        w.get::<RunResult>(run).is_some()
    });
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn changed_arguments_require_a_new_ticket_and_old_input_never_releases_it() {
    let (mut app, run, calls) = setup();
    let first = pending(&mut app);
    let effect = first.ticket().effect();
    if let EffectKind::ToolCall { args, .. } = &mut app
        .world_mut()
        .get_mut::<PendingEffect>(effect)
        .unwrap()
        .kind
    {
        *args = "{\"value\":2}".into();
    }
    assert_eq!(
        decide(app.world_mut(), first.ticket(), ApprovalChoice::Approve),
        Err(ApprovalError::Stale)
    );
    app.update();
    let next = app.world().get::<ApprovalRequest>(effect).unwrap().clone();
    assert_ne!(first.ticket(), next.ticket());
    assert_eq!(
        decide(app.world_mut(), first.ticket(), ApprovalChoice::Approve),
        Err(ApprovalError::Stale)
    );
    assert!(app.world().get::<Held>(effect).is_some());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    decide(app.world_mut(), next.ticket(), ApprovalChoice::Approve).unwrap();
    run_support::tick_until(&mut app, "revised approved result", |w| {
        w.get::<RunResult>(run).is_some()
    });
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn denial_and_cancel_do_not_dispatch_the_tool() {
    for choice in [
        ApprovalChoice::Deny("no".into()),
        ApprovalChoice::Cancel("stop".into()),
    ] {
        let (mut app, run, calls) = setup();
        let request = pending(&mut app);
        decide(app.world_mut(), request.ticket(), choice).unwrap();
        run_support::tick_until(&mut app, "denied or cancelled ending", |w| {
            w.get::<RunResult>(run).is_some() || w.get::<Failed>(run).is_some()
        });
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn removed_and_already_answered_targets_reject_input() {
    let (mut app, _, _) = setup();
    let request = pending(&mut app);
    let effect = request.ticket().effect();
    app.world_mut().entity_mut(effect).insert(EffectOutcome(Err(
        rig_core::error::ErrorReport::new(rig_core::error::ErrorKind::Denied, "another policy"),
    )));
    assert_eq!(
        decide(app.world_mut(), request.ticket(), ApprovalChoice::Approve),
        Err(ApprovalError::TooLate)
    );
    app.world_mut().despawn(effect);
    assert_eq!(
        decide(app.world_mut(), request.ticket(), ApprovalChoice::Approve),
        Err(ApprovalError::Missing(effect))
    );
}

#[test]
fn changing_an_approved_proposal_invalidates_even_repeated_input() {
    let (mut app, _, calls) = setup();
    let request = pending(&mut app);
    decide(app.world_mut(), request.ticket(), ApprovalChoice::Approve).unwrap();
    let effect = request.ticket().effect();
    app.world_mut()
        .get_mut::<PendingEffect>(effect)
        .unwrap()
        .key = HandlerKey::from("replacement");
    assert_eq!(
        decide(app.world_mut(), request.ticket(), ApprovalChoice::Approve),
        Err(ApprovalError::Stale)
    );
    app.update();
    assert!(app.world().get::<Held>(effect).is_some());
    assert!(
        app.world()
            .get::<ApprovalRequest>(effect)
            .unwrap()
            .is_pending()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn hold_loss_never_becomes_implicit_approval() {
    let (mut app, _, calls) = setup();
    let request = pending(&mut app);
    release_hold(app.world_mut(), request.ticket().effect(), "app/reviewer");
    assert_eq!(
        decide(app.world_mut(), request.ticket(), ApprovalChoice::Approve),
        Err(ApprovalError::HoldLost)
    );
    app.update();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        app.world().get::<ApprovalError>(request.ticket().effect()),
        Some(&ApprovalError::HoldLost)
    );
}

#[test]
fn recording_a_decision_does_not_publish_a_second_component_insertion() {
    #[derive(Resource, Default)]
    struct Inserts(usize);
    let (mut app, run, calls) = setup();
    app.init_resource::<Inserts>();
    app.add_observer(
        |_: On<Insert, ApprovalRequest>, mut inserts: ResMut<Inserts>| {
            inserts.0 += 1;
        },
    );
    let request = pending(&mut app);
    assert_eq!(app.world().resource::<Inserts>().0, 1);
    decide(app.world_mut(), request.ticket(), ApprovalChoice::Approve).unwrap();
    assert_eq!(app.world().resource::<Inserts>().0, 1);
    run_support::tick_until(&mut app, "approved observer run", |w| {
        w.get::<RunResult>(run).is_some()
    });
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn owner_and_run_changes_invalidate_displayed_tickets() {
    for change_owner in [true, false] {
        let (mut app, _, calls) = setup();
        let request = pending(&mut app);
        let effect = request.ticket().effect();
        if change_owner {
            app.world_mut()
                .entity_mut(effect)
                .insert(ApprovalRequired(Emitter::named("app/new-reviewer")));
        } else {
            let owner = app
                .world()
                .get::<rig_ecs::agent::RunOf>(request.run())
                .unwrap()
                .0;
            let other = app
                .world_mut()
                .spawn((rig_ecs::agent::Run, rig_ecs::agent::RunOf(owner)))
                .id();
            app.world_mut().entity_mut(effect).insert(ChildOf(other));
        }
        assert_eq!(
            decide(app.world_mut(), request.ticket(), ApprovalChoice::Approve),
            Err(ApprovalError::Stale)
        );
        app.update();
        let next = app.world().get::<ApprovalRequest>(effect).unwrap();
        assert_ne!(request.ticket(), next.ticket());
        assert!(next.is_pending());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn request_observers_can_supply_decisions_without_a_second_input_runtime() {
    let (mut app, run, calls) = setup();
    app.add_observer(
        |event: On<Add, ApprovalRequest>,
         requests: Query<&ApprovalRequest>,
         mut commands: Commands| {
            let ticket = requests.get(event.entity).unwrap().ticket();
            commands.queue(move |world: &mut World| {
                assert_eq!(
                    decide(world, ticket, ApprovalChoice::Approve),
                    Ok(DecisionOutcome::Applied)
                );
            });
        },
    );
    run_support::tick_until(&mut app, "observer approved result", |w| {
        w.get::<RunResult>(run).is_some()
    });
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn cloned_ui_state_cannot_authorize_a_different_world() {
    let (mut first, _, _) = setup();
    let request = pending(&mut first);
    let (mut second, _, _) = setup();
    let second_request = pending(&mut second);
    let target = second_request.ticket().effect();
    // Copying even the whole component must not make its foreign ticket valid.
    second
        .world_mut()
        .entity_mut(target)
        .insert(request.clone());
    assert_eq!(
        decide(
            second.world_mut(),
            request.ticket(),
            ApprovalChoice::Approve
        ),
        Err(ApprovalError::Stale)
    );
    second.update();
    let current = second.world().get::<ApprovalRequest>(target).unwrap();
    assert_ne!(current.ticket(), request.ticket());
    assert!(current.is_pending());
}

#[test]
fn copying_an_approved_snapshot_does_not_approve_another_effect() {
    let (mut app, _, calls) = setup();
    let request = pending(&mut app);
    let source = request.ticket().effect();
    acquire_hold(app.world_mut(), source, Emitter::named("app/other-policy"));
    decide(app.world_mut(), request.ticket(), ApprovalChoice::Approve).unwrap();
    let approved = app.world().get::<ApprovalRequest>(source).unwrap().clone();
    let effect = app.world().get::<PendingEffect>(source).unwrap().clone();
    let parent = app.world().get::<ChildOf>(source).unwrap().parent();
    let duplicate = app
        .world_mut()
        .spawn((
            effect,
            ChildOf(parent),
            approved,
            ApprovalRequired(Emitter::named("app/reviewer")),
        ))
        .id();
    app.update();
    let new_request = app.world().get::<ApprovalRequest>(duplicate).unwrap();
    assert_eq!(new_request.ticket().effect(), duplicate);
    assert_ne!(new_request.ticket(), request.ticket());
    assert!(new_request.is_pending());
    assert!(app.world().get::<Held>(duplicate).is_some());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn invalid_cancellation_never_consumes_or_acknowledges_a_decision() {
    for missing_runtime in [false, true] {
        let (mut app, run, calls) = setup();
        let request = pending(&mut app);
        if missing_runtime {
            app.world_mut()
                .remove_resource::<rig_ecs::agent::RunCounter>();
        } else {
            app.world_mut()
                .entity_mut(run)
                .remove::<rig_ecs::agent::RunOf>();
        }
        for _ in 0..2 {
            let result = decide(
                app.world_mut(),
                request.ticket(),
                ApprovalChoice::Cancel("stop".into()),
            );
            assert_eq!(
                result,
                Err(if missing_runtime {
                    ApprovalError::TooLate
                } else {
                    ApprovalError::InvalidTarget(request.ticket().effect())
                })
            );
            assert!(
                app.world()
                    .get::<ApprovalRequest>(request.ticket().effect())
                    .unwrap()
                    .is_pending()
            );
        }
        assert!(app.world().get::<rig_ecs::agent::Cancelled>(run).is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}

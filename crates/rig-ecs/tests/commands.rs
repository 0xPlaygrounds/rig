//! Public consumer coverage for immediate and deferred construction.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test assertions"
)]

mod run_support;

use bevy_app::{App, Update};
use bevy_ecs::{prelude::*, system::RunSystemOnce};
use rig_core::{effect::EffectFamily, serve::ServingPolicy};
use rig_ecs::{
    agent::{self, MessageParts, Run, RunResult},
    bus::{Handlers, run_to_quiescence},
    commands::{Agent, CommandFailures, OperationError, Prompt, RigCommands, install},
};
use run_support::{Capturing, NeverCalled};

fn app() -> App {
    let mut app = App::new();
    install(app.world_mut(), ServingPolicy::default()).expect("install");
    app.add_systems(Update, run_to_quiescence);
    app
}

#[derive(Component)]
struct ApplicationRequest;

#[derive(Resource, Default)]
struct Submitted(Vec<Entity>);

// No named lifetimes, SystemState callbacks, boxing, or generic arguments.
fn submit(commands: &mut Commands, model: Entity, tool: Entity) -> Entity {
    let agent = commands.spawn_agent(Agent::new(model).preamble("Be brief.").tools([tool]));
    let run = commands.prompt(Prompt::new(agent, "hello"));
    commands.entity(run).insert(ApplicationRequest);
    run
}

#[test]
fn one_system_registers_and_submits_with_commands_before_handlers() {
    let mut app = app();
    app.init_resource::<Submitted>();
    #[derive(Resource, Default)]
    struct Answers(std::collections::HashMap<Entity, String>);
    fn collect_answers(
        runs: Query<rig_ecs::inspect::RunView, With<ApplicationRequest>>,
        submitted: Res<Submitted>,
        mut answers: ResMut<Answers>,
    ) {
        for id in &submitted.0 {
            if let Ok(run) = runs.get(*id) {
                assert!(run.failure.is_none());
                if let Some(answer) = run.answer() {
                    answers.0.insert(*id, answer.to_owned());
                }
            }
        }
    }
    app.init_resource::<Answers>();
    app.add_systems(Update, collect_answers.after(run_to_quiescence));
    let (model, requests) = Capturing::new("model", "answer");
    // Deliberately put the Commands buffer before Handlers' own buffer.
    app.world_mut()
        .run_system_once(
            move |mut commands: Commands,
                  mut handlers: Handlers,
                  mut submitted: ResMut<Submitted>| {
                let model = handlers.register("model", model.clone()).expect("register");
                let tool = handlers
                    .register(
                        "lookup",
                        NeverCalled {
                            name: "lookup".into(),
                        },
                    )
                    .expect("tool");
                submitted.0.push(submit(&mut commands, model, tool));
                submitted.0.push(submit(&mut commands, model, tool));
            },
        )
        .expect("system");
    assert_eq!(app.world().resource::<CommandFailures>().iter().count(), 0);
    let runs = app.world().resource::<Submitted>().0.clone();
    for run in &runs {
        assert!(app.world().get::<ApplicationRequest>(*run).is_some());
        assert!(app.world().get::<Run>(*run).is_some());
    }
    run_support::tick_until(&mut app, "two submitted runs", |world| {
        world.resource::<Answers>().0.len() == runs.len()
    });
    let answers = &app.world().resource::<Answers>().0;
    assert!(
        runs.iter()
            .all(|run| answers.get(run).map(String::as_str) == Some("answer"))
    );
    let requests = requests.lock().expect("captured requests");
    assert_eq!(requests.len(), 2);
    for request in requests.iter() {
        assert_eq!(request.tools.len(), 1);
        assert_eq!(request.tools[0].name, "lookup");
    }
}

#[test]
fn named_options_match_independently_constructed_graph_requests() {
    for streaming in [false, true] {
        let mut app = app();
        let (model, requests) = Capturing::new("model", "answer");
        let model = Handlers::register_in(app.world_mut(), "model", model).expect("register");
        let tool = Handlers::register_in(
            app.world_mut(),
            "tool",
            NeverCalled {
                name: "tool".into(),
            },
        )
        .expect("tool");
        let named = Agent::new(model)
            .owner("named")
            .preamble("Be brief.")
            .max_turns(3)
            .max_tokens(100)
            .temperature(0.5)
            .tools([tool])
            .spawn(app.world_mut())
            .expect("agent");
        // This oracle intentionally uses no construction helper under test.
        let direct = app
            .world_mut()
            .spawn((
                agent::Owner("direct".into()),
                agent::UsesModel(model),
                agent::Preamble(Some("Be brief.".into())),
                agent::Temperature(Some(0.5)),
                agent::MaxTokens(Some(100)),
                agent::AdditionalParams(None),
                agent::ToolChoiceSpec(None),
                agent::Output::default(),
                agent::DefaultMaxTurns(Some(3)),
                agent::MaxTurns(3),
                agent::InvalidCalls::default(),
            ))
            .id();
        app.world_mut()
            .spawn((agent::Grant(tool), agent::Order(0), ChildOf(direct)));
        let history = vec![MessageParts::User {
            content: vec![rig_core::message::UserContent::text("earlier")],
        }];
        let mut submission = Prompt::new(named, "hello")
            .history(history.clone())
            .max_turns(2);
        if streaming {
            submission = submission.streaming();
        }
        let first = submission.spawn(app.world_mut()).expect("prompt");
        // Independently construct the run AND utterances; do not call spawn_run,
        // spawn_utterance, next_order, or the checked initializer on this side.
        let second = app
            .world_mut()
            .spawn((
                agent::Run,
                agent::RunOf(direct),
                agent::RunSeq(1),
                agent::RunStreaming(streaming),
                agent::Cursor::default(),
                agent::OutputRetries::default(),
                agent::InvalidRetries::default(),
                agent::OutputToolName::default(),
                agent::Usage::default(),
                rig_ecs::bus::Scope("direct/run#1".into()),
                agent::MaxTurns(2),
                agent::Assembling,
            ))
            .id();
        app.world_mut().resource_mut::<agent::RunCounter>().0 = 2;
        for (order, parts) in history
            .into_iter()
            .chain(std::iter::once(MessageParts::User {
                content: vec![rig_core::message::UserContent::text("hello")],
            }))
            .enumerate()
        {
            app.world_mut().spawn((
                agent::Utterance,
                parts.role(),
                agent::Parts(parts),
                agent::Order(order as u64),
                ChildOf(second),
            ));
        }
        assert_eq!(
            app.world().get::<agent::MaxTurns>(first),
            app.world().get::<agent::MaxTurns>(second)
        );
        run_support::tick_until(&mut app, "parity", |world| {
            world.get::<RunResult>(first).is_some() && world.get::<RunResult>(second).is_some()
        });
        let modes: Vec<_> = app
            .world_mut()
            .query::<&rig_ecs::bus::PendingEffect>()
            .iter(app.world())
            .filter_map(|effect| {
                if let rig_core::effect::EffectKind::Completion { stream, .. } = effect.kind {
                    Some(stream)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(modes, vec![streaming, streaming]);
        let requests = requests.lock().expect("requests");
        assert_eq!(requests.len(), 2);
        assert_eq!(
            serde_json::to_value(&requests[0]).unwrap(),
            serde_json::to_value(&requests[1]).unwrap()
        );
    }
}

#[test]
fn immediate_failure_does_not_create_partial_graphs() {
    let mut app = app();
    let tool = Handlers::register_in(
        app.world_mut(),
        "tool",
        NeverCalled {
            name: "tool".into(),
        },
    )
    .expect("tool");
    let before = app.world().entities().count_spawned();
    assert_eq!(
        Agent::new(tool).spawn(app.world_mut()),
        Err(OperationError::HandlerFamily {
            entity: tool,
            expected: EffectFamily::Completion
        })
    );
    assert_eq!(app.world().entities().count_spawned(), before);
    assert_eq!(
        Prompt::new(tool, "wrong role").spawn(app.world_mut()),
        Err(OperationError::NotAgent(tool))
    );
    let missing = app.world_mut().spawn_empty().id();
    app.world_mut().despawn(missing);
    assert_eq!(
        Prompt::new(missing, "stale").spawn(app.world_mut()),
        Err(OperationError::MissingEntity(missing))
    );
}

#[test]
fn deferred_failures_are_retained_and_consumable() {
    let mut app = app();
    let missing = app.world_mut().spawn_empty().id();
    app.world_mut().despawn(missing);
    app.world_mut()
        .run_system_once(move |mut commands: Commands| {
            commands.prompt(Prompt::new(missing, "stale"));
        })
        .expect("system");
    let failures: Vec<_> = app
        .world_mut()
        .resource_mut::<CommandFailures>()
        .drain()
        .collect();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].error, OperationError::MissingEntity(missing));
    assert!(app.world().get::<Run>(failures[0].entity).is_none());
    assert_eq!(app.world().resource::<CommandFailures>().iter().count(), 0);
}

#[test]
fn installation_and_missing_installation_are_explicit() {
    let mut world = World::new();
    let target = world.spawn_empty().id();
    assert_eq!(
        Agent::new(target).spawn(&mut world),
        Err(OperationError::NotInstalled)
    );
    install(&mut world, ServingPolicy::default()).expect("install");
    assert_eq!(
        install(&mut world, ServingPolicy::default()),
        Err(OperationError::AlreadyInstalled)
    );
}

#[test]
fn pending_memory_registration_loads_before_completion_in_either_parameter_order() {
    use rig_core::effect::{EffectKind, FamilyDescriptor, MemoryOp};
    use rig_ecs::bus::PendingEffect;

    fn start(commands: &mut Commands, handlers: &mut Handlers) {
        let (model, _) = Capturing::new("model", "answer");
        let model = handlers.register("model", model).expect("model");
        // The host can answer this through an ordinary ECS system.
        let memory = handlers
            .register_open("memory", FamilyDescriptor::Memory {})
            .expect("memory");
        let agent = commands.spawn_agent(Agent::new(model).memory(memory, "chat"));
        commands.prompt(Prompt::new(agent, "hello"));
    }

    for commands_first in [true, false] {
        let mut app = app();
        if commands_first {
            app.world_mut()
                .run_system_once(|mut commands: Commands, mut handlers: Handlers| {
                    start(&mut commands, &mut handlers);
                })
                .expect("commands first");
        } else {
            app.world_mut()
                .run_system_once(|mut handlers: Handlers, mut commands: Commands| {
                    start(&mut commands, &mut handlers);
                })
                .expect("handlers first");
        }
        assert_eq!(app.world().resource::<CommandFailures>().iter().count(), 0);
        let world = app.world_mut();
        assert_eq!(
            world
                .query_filtered::<(), With<agent::LoadingMemory>>()
                .iter(world)
                .count(),
            1
        );
        let effects: Vec<_> = world.query::<&PendingEffect>().iter(world).collect();
        assert_eq!(effects.len(), 1);
        assert!(matches!(
            effects[0].kind,
            EffectKind::Memory {
                op: MemoryOp::Load { .. }
            }
        ));
    }
}

#[test]
fn logical_deregistration_is_visible_before_its_commands_apply() {
    let mut app = app();
    app.world_mut()
        .run_system_once(|mut commands: Commands, mut handlers: Handlers| {
            let (model, _) = Capturing::new("model", "answer");
            let model = handlers.register("model", model).expect("model");
            handlers.deregister(&"model".into());
            commands.spawn_agent(Agent::new(model));
        })
        .expect("system");
    assert!(matches!(
        app.world()
            .resource::<CommandFailures>()
            .iter()
            .next()
            .map(|f| &f.error),
        Some(OperationError::HandlerFamily {
            expected: EffectFamily::Completion,
            ..
        })
    ));
}

#[test]
fn removing_a_materialized_handler_cancels_its_pending_binding() {
    let mut app = app();
    app.world_mut()
        .run_system_once(|mut commands: Commands, mut handlers: Handlers| {
            let (model, _) = Capturing::new("model", "answer");
            let model = handlers.register("model", model).expect("model");
            let agent = commands.spawn_agent(Agent::new(model));
            commands.entity(model).despawn();
            commands.prompt(Prompt::new(agent, "removed model"));
        })
        .expect("system without a deferred panic");
    let failures: Vec<_> = app
        .world_mut()
        .resource_mut::<CommandFailures>()
        .drain()
        .collect();
    assert_eq!(failures.len(), 1);
    let (replacement, _) = Capturing::new("model", "replacement");
    let entity = Handlers::register_in(app.world_mut(), "model", replacement)
        .expect("fresh registration after removal");
    assert!(app.world().get::<rig_ecs::bus::Bound>(entity).is_some());
}

#[test]
fn binding_observers_can_remove_a_reserved_agent_without_a_constructor_panic() {
    #[derive(Resource, Default)]
    struct Destination(Option<Entity>);
    let mut app = app();
    app.init_resource::<Destination>();
    app.add_observer(
        |_: On<Add, rig_ecs::bus::Bound>, target: Res<Destination>, mut commands: Commands| {
            if let Some(entity) = target.0 {
                commands.entity(entity).despawn();
            }
        },
    );
    app.world_mut()
        .run_system_once(
            |mut commands: Commands, mut handlers: Handlers, mut target: ResMut<Destination>| {
                let (model, _) = Capturing::new("model", "answer");
                let model = handlers.register("model", model).expect("model");
                target.0 = Some(commands.spawn_agent(Agent::new(model)));
            },
        )
        .expect("system");
    let entity = app
        .world()
        .resource::<Destination>()
        .0
        .expect("destination");
    let failures: Vec<_> = app
        .world_mut()
        .resource_mut::<CommandFailures>()
        .drain()
        .collect();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].error, OperationError::MissingEntity(entity));
}

#[test]
fn deferred_grants_are_ordered_idempotent_and_change_future_requests() {
    let mut app = app();
    let (agent, tool, first, requests) = app
        .world_mut()
        .run_system_once(|mut commands: Commands, mut handlers: Handlers| {
            let (model, requests) = Capturing::new("model", "answer");
            let model = handlers.register("model", model).unwrap();
            let tool = handlers
                .register(
                    "tool",
                    NeverCalled {
                        name: "tool".into(),
                    },
                )
                .unwrap();
            let agent = commands.spawn_agent(Agent::new(model));
            commands.grant_tool(agent, tool);
            commands.grant_tool(agent, tool);
            let run = commands.prompt(Prompt::new(agent, "first"));
            (agent, tool, run, requests)
        })
        .unwrap();
    run_support::tick_until(&mut app, "granted prompt", |w| {
        w.get::<RunResult>(first).is_some()
    });
    assert_eq!(requests.lock().unwrap()[0].tools.len(), 1);
    assert_eq!(app.world().resource::<CommandFailures>().iter().count(), 0);
    assert_eq!(
        rig_ecs::lifecycle::revoke_tool(app.world_mut(), agent, tool),
        Ok(1)
    );
    assert_eq!(
        rig_ecs::lifecycle::revoke_tool(app.world_mut(), agent, tool),
        Ok(0)
    );
    let next = Prompt::new(agent, "next").spawn(app.world_mut()).unwrap();
    run_support::tick_until(&mut app, "revoked prompt", |w| {
        w.get::<RunResult>(next).is_some()
    });
    assert!(requests.lock().unwrap()[1].tools.is_empty());
}

#[test]
fn granting_wrong_families_and_stale_agents_does_not_allocate_links() {
    let mut app = app();
    let (model, _) = Capturing::new("model", "answer");
    let model = Handlers::register_in(app.world_mut(), "model", model).unwrap();
    let agent = Agent::new(model).spawn(app.world_mut()).unwrap();
    let count = app.world().entities().count_spawned();
    assert!(matches!(
        rig_ecs::lifecycle::grant_tool(app.world_mut(), agent, model),
        Err(OperationError::HandlerFamily { .. })
    ));
    assert_eq!(app.world().entities().count_spawned(), count);
    app.world_mut().despawn(agent);
    assert_eq!(
        rig_ecs::lifecycle::grant_tool(app.world_mut(), agent, model),
        Err(OperationError::MissingEntity(agent))
    );
}

#[test]
fn static_grant_does_not_reuse_a_retrieval_only_link() {
    let mut app = app();
    let (model, requests) = Capturing::new("model", "answer");
    let model = Handlers::register_in(app.world_mut(), "model", model).unwrap();
    let tool = Handlers::register_in(
        app.world_mut(),
        "tool",
        NeverCalled {
            name: "tool".into(),
        },
    )
    .unwrap();
    let agent = Agent::new(model).spawn(app.world_mut()).unwrap();
    let retrieval = app
        .world_mut()
        .spawn((
            agent::Grant(tool),
            agent::Retrievable,
            agent::Order(0),
            ChildOf(agent),
        ))
        .id();
    let link = rig_ecs::lifecycle::grant_tool(app.world_mut(), agent, tool).unwrap();
    assert_ne!(link, retrieval);
    assert_eq!(
        rig_ecs::lifecycle::grant_tool(app.world_mut(), agent, tool),
        Ok(link)
    );
    assert!(app.world().get::<agent::Retrievable>(retrieval).is_some());
    let run = Prompt::new(agent, "use static grant")
        .spawn(app.world_mut())
        .unwrap();
    run_support::tick_until(&mut app, "static grant", |w| {
        w.get::<RunResult>(run).is_some()
    });
    assert_eq!(requests.lock().unwrap()[0].tools.len(), 1);
}

#[test]
fn deferred_lifecycle_failure_identifies_and_preserves_the_existing_target() {
    let mut app = app();
    let (model, _) = Capturing::new("model", "answer");
    let model = Handlers::register_in(app.world_mut(), "model", model).unwrap();
    let agent = Agent::new(model).spawn(app.world_mut()).unwrap();
    app.world_mut()
        .run_system_once(move |mut commands: Commands| {
            commands.grant_tool(agent, model);
        })
        .unwrap();
    let failures: Vec<_> = app
        .world_mut()
        .resource_mut::<CommandFailures>()
        .drain()
        .collect();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].entity, agent);
    assert!(matches!(
        failures[0].error,
        OperationError::HandlerFamily { .. }
    ));
    assert!(app.world().get::<agent::UsesModel>(agent).is_some());
    let run = Prompt::new(agent, "still usable")
        .spawn(app.world_mut())
        .unwrap();
    run_support::tick_until(&mut app, "preserved agent", |w| {
        w.get::<RunResult>(run).is_some()
    });
}

#[test]
fn handler_inspection_exposes_applied_bindings_and_commands_accept_pending_ids() {
    let mut app = app();
    let key = rig_core::effect::HandlerKey::from("model");
    app.world_mut()
        .run_system_once(|mut commands: Commands, mut handlers: Handlers| {
            let (model, _) = Capturing::new("model", "answer");
            let model = handlers.register("model", model).unwrap();
            assert!(handlers.descriptor(&"model".into()).is_none());
            assert!(handlers.keys().is_empty());
            assert!(handlers.descriptors().is_empty());
            commands.spawn_agent(Agent::new(model));
        })
        .unwrap();
    let before = Handlers::with(app.world_mut(), |handlers| {
        let before = handlers.descriptor(&key).unwrap();
        let (replacement, _) = Capturing::new("replacement-model", "new answer");
        handlers.register(key.clone(), replacement).unwrap();
        assert_eq!(handlers.descriptor(&key), Some(before.clone()));
        assert_eq!(handlers.descriptors(), vec![before.clone()]);
        before
    })
    .unwrap();
    Handlers::with(app.world_mut(), |handlers| {
        let after = handlers.descriptor(&key).unwrap();
        assert_ne!(after.family, before.family);
        assert_eq!(after.key, key);
        assert_eq!(handlers.keys(), vec![key.clone()]);
        assert_eq!(handlers.descriptors(), vec![after]);
        assert!(handlers.deregister(&key));
        assert!(handlers.descriptor(&key).is_some());
        assert_eq!(handlers.keys(), vec![key.clone()]);
    })
    .unwrap();
    Handlers::with(app.world_mut(), |handlers| {
        assert!(handlers.descriptor(&key).is_none());
        assert!(handlers.keys().is_empty());
        assert!(handlers.descriptors().is_empty());
    })
    .unwrap();
    assert_eq!(app.world().resource::<CommandFailures>().iter().count(), 0);
}

fn remove_added<T: Component>(app: &mut App) {
    app.add_observer(|added: On<Add, T>, mut commands: Commands| {
        commands.entity(added.event().entity).despawn();
    });
}

#[test]
fn agent_observers_can_remove_construction_targets_without_panics_or_orphan_grants() {
    for deferred in [false, true] {
        for remembering in [false, true] {
            let mut app = app();
            let (model, _) = Capturing::new("model", "answer");
            let model = Handlers::register_in(app.world_mut(), "model", model).unwrap();
            let tool = Handlers::register_in(
                app.world_mut(),
                "tool",
                NeverCalled {
                    name: "tool".into(),
                },
            )
            .unwrap();
            let memory = Handlers::with(app.world_mut(), |handlers| {
                handlers.register_open("memory", rig_core::effect::FamilyDescriptor::Memory {})
            })
            .unwrap()
            .unwrap();
            remove_added::<agent::Owner>(&mut app);
            let mut config = Agent::new(model).tools([tool]);
            if remembering {
                config = config.memory(memory, "conversation");
            }
            let before = app.world().entities().count_spawned();
            if deferred {
                let id = app
                    .world_mut()
                    .run_system_once(move |mut commands: Commands| {
                        commands.spawn_agent(config.clone())
                    })
                    .unwrap();
                let failures: Vec<_> = app
                    .world_mut()
                    .resource_mut::<CommandFailures>()
                    .drain()
                    .collect();
                assert_eq!(failures.len(), 1);
                assert_eq!(failures[0].entity, id);
                assert_eq!(failures[0].error, OperationError::MissingEntity(id));
            } else {
                assert!(matches!(
                    config.spawn(app.world_mut()),
                    Err(OperationError::MissingEntity(_))
                ));
            }
            assert_eq!(app.world().entities().count_spawned(), before);
            assert_eq!(
                app.world_mut()
                    .query::<&agent::Grant>()
                    .iter(app.world())
                    .count(),
                0
            );
        }
    }
}

#[test]
fn prompt_observers_can_remove_runs_at_each_construction_stage() {
    for phase in ["run", "limit", "assembling", "utterance"] {
        for deferred in [false, true] {
            let mut app = app();
            let (model, _) = Capturing::new("model", "answer");
            let model = Handlers::register_in(app.world_mut(), "model", model).unwrap();
            let agent = Agent::new(model).spawn(app.world_mut()).unwrap();
            match phase {
                "run" => remove_added::<agent::Run>(&mut app),
                "limit" => remove_added::<agent::MaxTurns>(&mut app),
                "assembling" => remove_added::<agent::Assembling>(&mut app),
                _ => {
                    app.add_observer(
                        |added: On<Add, agent::Utterance>,
                         parents: Query<&ChildOf>,
                         mut commands: Commands| {
                            let parent = parents.get(added.event().entity).unwrap().parent();
                            commands.entity(parent).despawn();
                        },
                    );
                }
            }
            let before = app.world().entities().count_spawned();
            let prompt = Prompt::new(agent, "removed by observer")
                .max_turns(2)
                .history([MessageParts::User {
                    content: vec![rig_core::message::UserContent::text("earlier")],
                }]);
            if deferred {
                let id = app
                    .world_mut()
                    .run_system_once(move |mut commands: Commands| commands.prompt(prompt.clone()))
                    .unwrap();
                let failures: Vec<_> = app
                    .world_mut()
                    .resource_mut::<CommandFailures>()
                    .drain()
                    .collect();
                assert_eq!(failures.len(), 1, "{phase}");
                assert_eq!(
                    failures[0].error,
                    OperationError::MissingEntity(id),
                    "{phase}"
                );
            } else {
                assert!(
                    matches!(
                        prompt.spawn(app.world_mut()),
                        Err(OperationError::MissingEntity(_))
                    ),
                    "{phase}"
                );
            }
            assert_eq!(app.world().entities().count_spawned(), before, "{phase}");
        }
    }
}

#[test]
fn grant_observers_can_remove_the_link_or_agent_without_reporting_success() {
    for remove_parent in [false, true] {
        let mut app = app();
        let (model, _) = Capturing::new("model", "answer");
        let model = Handlers::register_in(app.world_mut(), "model", model).unwrap();
        let tool = Handlers::register_in(
            app.world_mut(),
            "tool",
            NeverCalled {
                name: "tool".into(),
            },
        )
        .unwrap();
        let agent = Agent::new(model).spawn(app.world_mut()).unwrap();
        app.add_observer(
            move |added: On<Add, agent::Grant>, mut commands: Commands| {
                commands
                    .entity(if remove_parent {
                        agent
                    } else {
                        added.event().entity
                    })
                    .despawn();
            },
        );
        assert!(matches!(
            rig_ecs::lifecycle::grant_tool(app.world_mut(), agent, tool),
            Err(OperationError::MissingEntity(_))
        ));
        assert_eq!(
            app.world_mut()
                .query::<&agent::Grant>()
                .iter(app.world())
                .count(),
            0
        );
    }
}

#[test]
fn rejected_child_construction_removes_the_destination_and_never_dispatches() {
    for deferred in [false, true] {
        let mut app = app();
        let (model, requests) = Capturing::new("model", "must not run");
        let model = Handlers::register_in(app.world_mut(), "model", model).unwrap();
        let tool = Handlers::register_in(
            app.world_mut(),
            "tool",
            NeverCalled {
                name: "tool".into(),
            },
        )
        .unwrap();
        let agent = Agent::new(model).spawn(app.world_mut()).unwrap();
        remove_added::<agent::Grant>(&mut app);
        remove_added::<agent::Utterance>(&mut app);
        let before = app.world().entities().count_spawned();
        if deferred {
            let (partial_agent, partial_run) = app
                .world_mut()
                .run_system_once(move |mut commands: Commands| {
                    (
                        commands.spawn_agent(Agent::new(model).tools([tool])),
                        commands.prompt(Prompt::new(agent, "rejected")),
                    )
                })
                .unwrap();
            assert!(app.world().get_entity(partial_agent).is_err());
            assert!(app.world().get_entity(partial_run).is_err());
            assert_eq!(app.world().resource::<CommandFailures>().iter().count(), 2);
        } else {
            assert!(matches!(
                Agent::new(model).tools([tool]).spawn(app.world_mut()),
                Err(OperationError::MissingEntity(_))
            ));
            assert!(matches!(
                Prompt::new(agent, "rejected").spawn(app.world_mut()),
                Err(OperationError::MissingEntity(_))
            ));
        }
        assert_eq!(app.world().entities().count_spawned(), before);
        for _ in 0..4 {
            app.update();
        }
        assert!(requests.lock().unwrap().is_empty());
        assert_eq!(
            app.world_mut()
                .query::<&rig_ecs::bus::PendingEffect>()
                .iter(app.world())
                .count(),
            0
        );
        assert!(app.world().get::<agent::UsesModel>(agent).is_some());
    }
}

#[test]
fn construction_keeps_the_validated_handler_descriptor_snapshot() {
    use rig_core::{
        effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, family},
        serve::{Dispatch, Reply, Serve},
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    struct Changing {
        model: Capturing,
        reads: Arc<AtomicUsize>,
    }
    impl Serve for Changing {
        type Family = family::Completion;
        fn descriptor(&self) -> HandlerDescriptor {
            let mut descriptor = self.model.descriptor();
            if self.reads.fetch_add(1, Ordering::SeqCst) != 0 {
                descriptor.family = FamilyDescriptor::Custom {
                    kind: "changed".into(),
                };
            }
            descriptor
        }
        async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
            self.model.serve(kind, dispatch).await
        }
    }
    let mut app = app();
    let reads = Arc::new(AtomicUsize::new(0));
    let (model, captured) = Capturing::new("model", "answer");
    let model = Handlers::register_in(
        app.world_mut(),
        "model",
        Changing {
            model,
            reads: reads.clone(),
        },
    )
    .expect("register");
    let agent = Agent::new(model)
        .spawn(app.world_mut())
        .expect("validated model family");
    let run = Prompt::new(agent, "hello")
        .spawn(app.world_mut())
        .expect("submit");
    run_support::tick_until(&mut app, "answer", |world| {
        world.get::<RunResult>(run).is_some()
    });
    assert_eq!(captured.lock().expect("requests").len(), 1);
    assert_eq!(reads.load(Ordering::SeqCst), 1);
}

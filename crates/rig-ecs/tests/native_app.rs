//! App installation and the host-owned scheduling boundary.

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use bevy_app::App;
use bevy_ecs::{
    prelude::*,
    schedule::{LogLevel, Schedules},
};
use rig_core::{message::AssistantContent, serve::ServingPolicy};
use rig_ecs::{
    RigPlugin,
    agent::{DefaultMaxTurns, Owner, RunResult, Settled, UsesModel},
    bus::{Bus, BusSet, Handlers, IdCounter, Policy, Progress, RigSchedule},
    systems::{RunCommands, install_agent},
    testing::{Scripted, tick_until},
};

#[derive(Resource, Default)]
struct Passes(usize);

fn count_pass(mut passes: ResMut<Passes>) {
    passes.0 += 1;
}

#[test]
fn plugin_preserves_installed_bus_agent_and_host_schedule() {
    let mut app = App::new();
    let policy = ServingPolicy {
        command_capacity: 7,
        ..Default::default()
    };
    Bus::with_policy(policy)
        .ambiguity_detection(LogLevel::Error)
        .install(app.world_mut());
    install_agent(app.world_mut());
    app.world_mut().resource_mut::<IdCounter>().0 = 41;
    app.insert_resource(Passes::default());
    app.add_systems(RigSchedule, count_pass.in_set(BusSet::Gate));
    app.add_plugins(RigPlugin::default());
    install_agent(app.world_mut());

    assert_eq!(app.world().resource::<Policy>().0, policy);
    assert_eq!(app.world().resource::<IdCounter>().0, 41);
    assert_eq!(
        app.world()
            .resource::<Schedules>()
            .get(RigSchedule)
            .unwrap()
            .get_build_settings()
            .ambiguity_detection,
        LogLevel::Error
    );
    app.update();
    assert_eq!(app.world().resource::<Passes>().0, 1);
    app.update();
    assert_eq!(app.world().resource::<Passes>().0, 2);
}

#[test]
fn update_drives_to_quiescence_without_an_extra_main_schedule_pass() {
    let mut app = App::new();
    app.add_plugins(RigPlugin::default());
    install_agent(app.world_mut());
    app.init_resource::<Passes>();
    app.add_systems(
        RigSchedule,
        (|mut passes: ResMut<Passes>, mut progress: ResMut<Progress>| {
            passes.0 += 1;
            if passes.0 == 1 {
                progress.0 = true;
            }
        })
        .in_set(BusSet::Gate),
    );
    app.update();
    assert_eq!(app.world().resource::<Passes>().0, 2);
    app.update();
    assert_eq!(app.world().resource::<Passes>().0, 3);
}

#[test]
fn plugin_runs_normal_dependency_handlers_with_and_without_preinstallation() {
    for preinstall in [false, true] {
        for streamed in [false, true] {
            let mut app = App::new();
            let bus = Bus::default().ambiguity_detection(LogLevel::Error);
            if preinstall {
                bus.install(app.world_mut());
                install_agent(app.world_mut());
            }
            app.add_plugins(RigPlugin { bus });
            let (handler, requests) = Scripted::new(
                "test/model",
                vec![vec![AssistantContent::text("hello world")]],
            );
            let model = Handlers::with(app.world_mut(), |handlers| {
                handlers.register("test/model", handler)
            })
            .unwrap()
            .unwrap();
            let owner = app
                .world_mut()
                .spawn((
                    Owner("test".to_owned()),
                    UsesModel(model),
                    DefaultMaxTurns(Some(1)),
                ))
                .id();
            let run = app.world_mut().spawn_run(
                owner,
                "hello",
                rig_ecs::systems::RunConfig {
                    history: &[],
                    streamed,
                    max_turns: None,
                },
            );
            tick_until(&mut app, "scripted run settles", |world| {
                world.get::<Settled>(run).is_some()
            });
            assert_eq!(app.world().get::<RunResult>(run).unwrap().0, "hello world");
            assert_eq!(requests.lock().unwrap().len(), 1);
        }
    }
}

#[derive(Resource, Default)]
struct DeliveredImages(Vec<serde_json::Value>);

#[test]
fn scripted_content_and_call_identity_survive_unary_and_streamed_dispatch() {
    use rig_core::{
        completion::CompletionRequestBuilder,
        effect::{EffectKind, Outcome},
        message::{DocumentSourceKind, Image, Reasoning},
        streaming::StreamEvent,
    };
    use rig_ecs::bus::{EffectOutcome, PendingEffect, StreamItemsDelivered};

    let image = Image {
        data: DocumentSourceKind::base64("aW1hZ2U="),
        ..Image::default()
    };
    let mut call = AssistantContent::tool_call_with_call_id(
        "provider-item",
        "custom-call-id".to_owned(),
        "calculate",
        serde_json::json!({"x": 4}),
    );
    if let AssistantContent::ToolCall(call) = &mut call {
        call.signature = Some("call-signature".to_owned());
        call.additional_params = Some(serde_json::json!({"opaque": "tool-metadata"}));
    }
    let parts = vec![
        AssistantContent::text("before"),
        AssistantContent::Reasoning(
            Reasoning::new_with_signature("reasoning", Some("reasoning-signature".to_owned()))
                .with_id("reasoning-id".to_owned()),
        ),
        call,
        AssistantContent::Image(image.clone()),
        AssistantContent::text("after"),
    ];
    for streamed in [false, true] {
        let mut app = App::new();
        app.add_plugins(RigPlugin::default());
        app.init_resource::<DeliveredImages>();
        app.add_observer(
            |delivery: On<StreamItemsDelivered>, mut images: ResMut<DeliveredImages>| {
                for item in &delivery.items {
                    if let Ok(StreamEvent::Unknown(payload)) = item {
                        images.0.push(payload.value().clone());
                    }
                }
            },
        );
        let (handler, _) = Scripted::new("content/model", vec![parts.clone()]);
        Handlers::with(app.world_mut(), |handlers| {
            handlers.register("content/model", handler)
        })
        .unwrap()
        .unwrap();
        let effect = app
            .world_mut()
            .spawn(PendingEffect::new(
                "content/model",
                EffectKind::Completion {
                    request: CompletionRequestBuilder::unbound("content").build(),
                    stream: streamed,
                },
            ))
            .id();
        tick_until(&mut app, "scripted content arrives", |world| {
            world.get::<EffectOutcome>(effect).is_some()
        });
        let Ok(Outcome::Completion(response)) =
            &app.world().get::<EffectOutcome>(effect).unwrap().0
        else {
            panic!("expected a completion outcome");
        };
        // Core's stream accumulator has no image block. Images are delivered
        // verbatim as Unknown events; all modeled parts retain their metadata.
        let expected: Vec<_> = parts
            .iter()
            .filter(|part| !streamed || !matches!(part, AssistantContent::Image(_)))
            .cloned()
            .collect();
        assert_eq!(response.choice, expected);
        let expected_images = if streamed {
            vec![serde_json::to_value(&image).unwrap()]
        } else {
            Vec::new()
        };
        assert_eq!(app.world().resource::<DeliveredImages>().0, expected_images);
    }
}

#[test]
fn concurrent_scripts_keep_responses_paired_with_captured_requests() {
    use rig_core::{
        completion::CompletionRequestBuilder,
        effect::{EffectId, EffectKind, Outcome},
        message::{Message, UserContent},
        serve::{Dispatch, Reply, Serve},
    };
    const WORKERS: usize = 8;
    const EACH: usize = 32;
    for _ in 0..4 {
        let (scripted, captured) = Scripted::new(
            "concurrent",
            (0..WORKERS * EACH)
                .map(|turn| vec![AssistantContent::text(turn.to_string())])
                .collect(),
        );
        let barrier = std::sync::Barrier::new(WORKERS);
        let responses = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..WORKERS)
                .map(|worker| {
                    let scripted = &scripted;
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        (0..EACH)
                            .map(|offset| {
                                let id = worker * EACH + offset;
                                let streamed = id.is_multiple_of(2);
                                let request =
                                    CompletionRequestBuilder::unbound(id.to_string()).build();
                                let reply = futures::executor::block_on(scripted.serve(
                                    EffectKind::Completion {
                                        request,
                                        stream: streamed,
                                    },
                                    Dispatch::new(EffectId::from_raw(id as u64), streamed),
                                ));
                                let Reply::Outcome(Ok(Outcome::Completion(response))) = reply
                                else {
                                    panic!("expected a completion reply");
                                };
                                let AssistantContent::Text(text) = response.choice.first().unwrap()
                                else {
                                    panic!("expected the selected scripted turn");
                                };
                                (id, text.text.parse::<usize>().unwrap())
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|worker| worker.join().unwrap())
                .collect::<std::collections::HashMap<_, _>>()
        });
        let requests = captured.lock().unwrap();
        assert_eq!(requests.len(), WORKERS * EACH);
        // Arrival order is intentionally unconstrained; response pairing is not.
        for (turn, request) in requests.iter().enumerate() {
            let Message::User { content } = request.chat_history.first().unwrap() else {
                panic!("expected the captured request");
            };
            let UserContent::Text(text) = content.first().unwrap() else {
                panic!("expected request identity");
            };
            let id = text.text.parse::<usize>().unwrap();
            assert_eq!(responses.get(&id), Some(&turn));
        }
    }
}

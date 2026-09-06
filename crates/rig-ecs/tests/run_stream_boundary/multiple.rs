//! Controlled multi-decision prefixes; provider transport cannot enforce these gates.

use super::*;

type Stage = (Vec<StreamEvent>, oneshot::Receiver<()>);
struct Stages(Mutex<Vec<Stage>>);

impl Serve for Stages {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> HandlerDescriptor {
        NameThenGate(Arc::default()).descriptor()
    }
    async fn serve(&self, _: EffectKind, mut sink: OutcomeSink) {
        let stages = std::mem::take(&mut *self.0.lock().expect("stages"));
        for (events, gate) in stages {
            for event in events {
                if sink.send(Ok(event)).await.is_err() {
                    return;
                }
            }
            if gate.await.is_err() {
                return;
            }
        }
    }
}

fn open() -> StreamEvent {
    StreamEvent::BlockStart {
        id: BlockId::Wire("reused".into()),
        kind: BlockKind::ToolCall,
    }
}
fn name(value: &str) -> StreamEvent {
    StreamEvent::BlockDelta {
        id: BlockId::Wire("reused".into()),
        delta: Delta::ToolName { name: value.into() },
    }
}
fn close(value: &str) -> StreamEvent {
    StreamEvent::BlockEnd {
        id: BlockId::Wire("reused".into()),
        end: BlockClose::ToolCall(
            ToolCallEnd::whole(value, serde_json::json!({})).with_call_id("first-final"),
        ),
        block: None,
    }
}

#[derive(Resource)]
struct Decisions {
    first: Resolution,
    seen: Vec<InvalidCall>,
}
fn decide(
    mut commands: Commands,
    calls: Query<(Entity, &InvalidCall), Added<InvalidCall>>,
    mut state: ResMut<Decisions>,
) {
    for (entity, call) in &calls {
        let resolution = if state.seen.is_empty() {
            state.first.clone()
        } else {
            Resolution::Fail
        };
        state.seen.push(call.clone());
        commands.entity(entity).insert(resolution);
    }
}

fn setup(first: Resolution, stages: Vec<Stage>) -> (bevy_app::App, Entity) {
    let mut app = app();
    app.insert_resource(Decisions {
        first,
        seen: vec![],
    });
    app.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, decide.in_set(RigSet::Judge));
    let model = register(&mut app, "boundary/model", Stages(Mutex::new(stages)));
    let agent = spawn_agent(app.world_mut(), "boundary", model);
    let run = spawn_run(app.world_mut(), agent, &[], "calls", true, Some(2));
    (app, run)
}

#[test]
fn buffered_names_and_eof_keep_ordered_policy_prefixes() {
    let (release, gate) = oneshot::channel();
    release.send(()).expect("receiver exists");
    let mut second_close = close("later");
    if let StreamEvent::BlockEnd {
        end: BlockClose::ToolCall(end),
        ..
    } = &mut second_close
    {
        end.call_id = Some("second-final".into());
    }
    let (mut app, run) = setup(
        Resolution::Repair { to: "fixed".into() },
        vec![(
            vec![
                open(),
                name("wrong"),
                close("wrong"),
                open(),
                name("later"),
                second_close,
                StreamEvent::Final(StreamFinal::new("boundary", ProviderUsage::new())),
            ],
            gate,
        )],
    );
    // Deliberately batch the producer's entire delivery before agent folding.
    // The real bus still owns collection and builds the completion outcome.
    app.world_mut()
        .resource_mut::<Schedules>()
        .get_mut(RigSchedule)
        .expect("schedule")
        .configure_sets(
            RigSet::Fold.run_if(|outcomes: Query<&EffectOutcome>| !outcomes.is_empty()),
        );
    tick_until(&mut app, "buffered failure", |world| {
        world.get::<Failed>(run).is_some()
    });
    let seen = &app.world().resource::<Decisions>().seen;
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].name, "wrong");
    assert_eq!(seen[1].name, "later");
    assert_eq!(seen[0].stream_offset, Some(1));
    assert_eq!(seen[1].stream_offset, Some(4));
    let names: Vec<_> = seen[1]
        .prefix
        .iter()
        .filter_map(|part| match part {
            AssistantContent::ToolCall(call) => Some(call.function.name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, ["fixed", "later"]);
}

#[test]
fn completed_block_without_name_delta_is_actionable_before_eof() {
    let (_hold, gate) = oneshot::channel();
    let (mut app, run) = setup(Resolution::Fail, vec![(vec![open(), close("wrong")], gate)]);
    tick_until(&mut app, "full call before EOF", |world| {
        world.get::<Failed>(run).is_some()
    });
    let seen = &app.world().resource::<Decisions>().seen;
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].stream_offset, Some(1));
    assert_eq!(
        seen[0].id,
        ToolCallId::new("first-final").expect("provider id")
    );
    assert_eq!(seen[0].arguments, serde_json::json!({}));
    assert_eq!(seen[0].prefix.len(), 1);
    let AssistantContent::ToolCall(call) = &seen[0].prefix[0] else {
        panic!("retained completed call")
    };
    assert!(call.provider.is_some());
    assert_eq!(call.id, seen[0].id);
    assert_eq!(
        app.world_mut()
            .query::<&EffectOutcome>()
            .iter(app.world())
            .count(),
        0
    );
}

#[test]
fn later_failure_keeps_its_prefix_with_earlier_repair_and_reused_block() {
    let (release, first) = oneshot::channel();
    let (_hold, second) = oneshot::channel();
    let (mut app, run) = setup(
        Resolution::Repair { to: "fixed".into() },
        vec![
            (vec![open(), name("wrong")], first),
            (vec![close("wrong"), open(), name("later")], second),
        ],
    );
    tick_until(&mut app, "first repair", |world| {
        world.resource::<Decisions>().seen.len() == 1
    });
    release.send(()).expect("producer live");
    tick_until(&mut app, "later failure", |world| {
        world.get::<Failed>(run).is_some()
    });
    assert!(
        matches!(app.world().get::<Failed>(run), Some(Failed(Failure::UnknownToolCall { name })) if name == "later")
    );
    assert_eq!(app.world().resource::<Decisions>().seen.len(), 2);
    assert_eq!(
        app.world_mut()
            .query::<&EffectOutcome>()
            .iter(app.world())
            .count(),
        0
    );
    let content: Vec<_> = app
        .world_mut()
        .query::<(&ChildOf, &Parts)>()
        .iter(app.world())
        .filter(|(parent, _)| parent.parent() == run)
        .flat_map(|(_, parts)| match &parts.0 {
            MessageParts::Assistant { content, .. } => content.clone(),
            _ => vec![],
        })
        .collect();
    let names: Vec<_> = content
        .iter()
        .filter_map(|part| match part {
            AssistantContent::ToolCall(call) => Some(call.function.name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, ["fixed", "later"]);
}

#[test]
fn ignore_suppresses_later_name_delta_but_not_reused_block() {
    let (release_first, first) = oneshot::channel();
    let (release_second, second) = oneshot::channel();
    let (_hold, third) = oneshot::channel();
    let (mut app, run) = setup(
        Resolution::Ignore,
        vec![
            (vec![open(), name("wrong")], first),
            (vec![name("renamed")], second),
            (vec![close("renamed"), open(), name("later")], third),
        ],
    );
    tick_until(&mut app, "first ignore", |world| {
        world.resource::<Decisions>().seen.len() == 1
    });
    release_first.send(()).expect("producer live");
    tick_until(&mut app, "second name delivered", |world| {
        world.query::<&Streamed>().iter(world).any(|stream| stream.events.iter().any(|event| matches!(event, StreamEvent::BlockDelta { delta: Delta::ToolName { name }, .. } if name == "renamed")))
    });
    app.update();
    assert_eq!(app.world().resource::<Decisions>().seen.len(), 1);
    assert!(app.world().get::<Failed>(run).is_none());
    release_second.send(()).expect("producer live");
    tick_until(&mut app, "reused block rejected", |world| {
        world.get::<Failed>(run).is_some()
    });
    assert_eq!(app.world().resource::<Decisions>().seen.len(), 2);
    assert_eq!(app.world().resource::<Decisions>().seen[1].name, "later");
}

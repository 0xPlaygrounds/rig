//! Ordered stream errors bound invalid-name discovery, including buffered EOF.
use super::*;
use rig_core::error::{ErrorKind, ErrorReport};

struct ErrorAndName {
    error_first: bool,
}

impl Serve for ErrorAndName {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> HandlerDescriptor {
        NameThenGate(Arc::default()).descriptor()
    }
    async fn serve(&self, _: EffectKind, _dispatch: Dispatch) -> Reply {
        let error_first = self.error_first;

        Reply::written(move |mut writer| async move {
            let id = BlockId::Wire("failed-stream".into());
            let error = Err(ErrorReport::new(
                ErrorKind::Provider,
                "first provider failure",
            ));
            let name = Ok(StreamEvent::BlockDelta {
                id: id.clone(),
                delta: Delta::ToolName {
                    name: "wrong".into(),
                },
            });
            writer
                .event(StreamEvent::BlockStart {
                    id: id.clone(),
                    kind: BlockKind::ToolCall,
                })
                .await
                .expect("open stream");
            let items = if error_first {
                [error, name]
            } else {
                [name, error]
            };
            for item in items {
                (match item {
                    Ok(event) => writer.event(event).await,
                    Err(error) => writer.error(error).await,
                })
                .expect("open stream");
            }
            // A later event must neither create a second decision nor keep an
            // earlier repair waiting forever for events that cannot be validated.
            writer
                .event(StreamEvent::BlockEnd {
                    id,
                    end: BlockClose::ToolCall(ToolCallEnd::whole("wrong", serde_json::json!({}))),
                    block: None,
                })
                .await
                .expect("open stream");
        })
    }
}

fn outcome(error_first: bool, resolution: Resolution) -> (Failure, usize) {
    let mut app = app();
    app.init_resource::<RepairCount>()
        .insert_resource(Decision(resolution));
    app.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, decide_name.in_set(RigSet::Judge));
    app.world_mut()
        .resource_mut::<Schedules>()
        .get_mut(RigSchedule)
        .expect("schedule")
        .configure_sets(
            RigSet::Fold.run_if(|outcomes: Query<&EffectOutcome>| !outcomes.is_empty()),
        );
    let model = register(&mut app, "boundary/model", ErrorAndName { error_first });
    let agent = spawn_agent(app.world_mut(), "boundary", model);
    let run = spawn_run(app.world_mut(), agent, &[], "calls", true, Some(1));
    tick_until(&mut app, "ordered stream failure", |world| {
        world.get::<Failed>(run).is_some()
    });
    (
        app.world().get::<Failed>(run).expect("failed").0.clone(),
        app.world().resource::<RepairCount>().0,
    )
}

#[test]
fn provider_error_before_invalid_name_prevents_policy_intervention() {
    let (failure, decisions) = outcome(true, Resolution::Fail);
    assert!(
        matches!(failure, Failure::Provider(ref error) if error.message == "first provider failure")
    );
    assert_eq!(decisions, 0);
}

#[test]
fn invalid_name_before_provider_error_keeps_its_failure_precedence() {
    let (failure, decisions) = outcome(false, Resolution::Fail);
    assert!(matches!(failure, Failure::UnknownToolCall { ref name } if name == "wrong"));
    assert_eq!(decisions, 1);
}

#[test]
fn repaired_name_does_not_wait_on_events_after_provider_error() {
    let (failure, decisions) = outcome(false, Resolution::Repair { to: "fixed".into() });
    assert!(
        matches!(failure, Failure::Provider(ref error) if error.message == "first provider failure")
    );
    assert_eq!(decisions, 1);
}

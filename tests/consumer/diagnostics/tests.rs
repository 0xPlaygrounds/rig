use super::super::{Case, Error, Provider};
use super::*;
use rig_core::effect::{EffectKind, HandlerDescriptor};
use rig_core::serve::{OutcomeSink, Serve};

struct PendingModel(tokio::sync::mpsc::UnboundedSender<()>);
impl Serve for PendingModel {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> HandlerDescriptor {
        super::super::Scripted.descriptor()
    }
    async fn serve(&self, _: EffectKind, _sink: OutcomeSink) {
        self.0.send(()).expect("receiver waits for serving");
        std::future::pending::<()>().await;
    }
}

fn repair_case() -> Case {
    let mut case = super::super::cases()
        .into_iter()
        .find(|case| case.provider == Provider::Synthetic)
        .expect("synthetic case");
    case.repair = true;
    case
}

/// Dropping the actual running future must retain its world before teardown.
#[tokio::test]
async fn dropped_execution_preserves_pending_effects_and_repair_state() -> Result<(), Error> {
    let case = repair_case();
    let (started, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let (_, diagnostics) = capture(async {
        let mut execution = Box::pin(super::super::execute(&case, PendingModel(started)));
        tokio::select! {
            result = &mut execution => panic!("execution unexpectedly ended: {result:?}"),
            started = receiver.recv() => assert_eq!(started, Some(())),
        }
        drop(execution);
    })
    .await;
    let runtime = diagnostics
        .iter()
        .find(|item| item["kind"] == "runtime")
        .expect("retained runtime");
    let path =
        super::super::artifacts::root().join(runtime["path"].as_str().expect("artifact path"));
    let evidence: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    assert!(
        !evidence["pending"]
            .as_array()
            .expect("pending effects")
            .is_empty()
    );
    assert!(evidence["repair"].is_object());
    assert!(evidence["effects"].is_object());
    Ok(())
}

/// A schedule panic must retain the live world rather than only a generic error.
#[tokio::test]
async fn unwound_execution_preserves_runtime_evidence() -> Result<(), Error> {
    use bevy_ecs::schedule::IntoScheduleConfigs;
    use futures::FutureExt;
    let case = repair_case();
    let mut app = super::super::build(&case, None)?;
    rig_ecs::bus::Handlers::with(app.world_mut(), |handlers| {
        handlers.register(super::super::MODEL, super::super::Scripted)
    })??;
    app.add_systems(
        rig_ecs::bus::RigSchedule,
        (|| panic!("controlled schedule failure")).ambiguous_with_all(),
    );
    let run = super::super::program(&mut app, &case)?;
    let (result, diagnostics) =
        capture(std::panic::AssertUnwindSafe(super::super::drive(app, run)).catch_unwind()).await;
    let panic = result.expect_err("schedule must panic");
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"controlled schedule failure")
    );
    let runtime = diagnostics
        .iter()
        .find(|item| item["kind"] == "runtime")
        .expect("retained runtime");
    let path =
        super::super::artifacts::root().join(runtime["path"].as_str().expect("artifact path"));
    let evidence: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    assert!(evidence["repair"].is_object());
    assert!(evidence["observations"].is_array());
    Ok(())
}

use std::sync::atomic::{AtomicUsize, Ordering};

use bevy_tasks::AsyncComputeTaskPool;
use futures_timer::Delay;
use rig_core::effect::{EffectKind, HandlerDescriptor, family};
use rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness::rig_ecs::effects::Handler;
use rig_harness_test_support::{
    Collected, Ended, ToolOutputs, app_in, calls, collect, connect, first_agent, log_ended,
    log_tool_outputs, reply, run_until, scratch_home,
};
use rig_tui::StatusItems;

use super::{FILE, StallWatchdogPlugin};

/// A model whose first stream never sends anything; later calls go to
/// `inner`.
struct StallsOnce {
    inner: ErasedHandler,
    served: AtomicUsize,
}

impl Serve for StallsOnce {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        if self.served.fetch_add(1, Ordering::SeqCst) == 0 {
            return Reply::Stream(Box::pin(futures::stream::pending()));
        }
        self.inner.serve(kind, dispatch).await
    }
}

/// A tool that takes a second.
fn sleepy(called: On<ToolCalled<serde_json::Value>>, wake: Res<Wake>, mut commands: Commands) {
    let work = async {
        Delay::new(Duration::from_secs(1)).await;
        ToolResult::success("slept".into())
    };
    let running = Running::answering(AsyncComputeTaskPool::get(), &wake, work);
    commands.entity(called.call).insert(running);
}

/// Whether the status line ever said the tool ran long.
#[derive(Resource, Default)]
struct ShownLong(bool);

fn watch_status(items: Query<&StatusItems>, mut shown: ResMut<ShownLong>) {
    shown.0 |= items
        .iter()
        .flat_map(StatusItems::iter)
        .any(|item| item.text.starts_with("sleepy running"));
}

#[test]
fn a_stalled_stream_is_sent_again_and_a_long_tool_is_reported() {
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "sleepy", serde_json::json!({})),
        reply("ok"),
    ]);
    let home = scratch_home("stall-watchdog");
    let session = SessionId::generate();
    let (mut app, wakes) = app_in(&home, &session);
    let settings = "[stall_watchdog]\nmodel_timeout = 0.2\n\
                    [stall_watchdog.tools.sleepy]\nsoft_limit = 0.2\n";
    app.insert_resource(Settings::from_toml(settings))
        .add_plugins((AgentPlugin, SessionLogPlugin, StallWatchdogPlugin))
        .add_open_tool("sleepy", "Sleep a second.", ToolOptions::default(), sleepy)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .register_required_components::<TurnOf, StatusItems>()
        .init_resource::<ShownLong>()
        .add_systems(Last, watch_status);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();

    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let connection = connect(&model);
    assert!(connection.is_some(), "the scripted model connects");
    let Some(mut connection) = connection else {
        return;
    };
    let inner = connection.handler.0.clone();
    let served = AtomicUsize::new(0);
    connection.handler = Handler(ErasedHandler::new(StallsOnce { inner, served }));
    app.world_mut().entity_mut(agent).insert(connection);
    app.world_mut().trigger(Deliver::new(
        agent,
        "Sleep, then say ok.",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let world = app.world();
    let ended = &world.resource::<Ended>().0;
    let answer = match ended.first() {
        Some(TurnOutcome::Answered(message)) => message.final_answer(),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("ok"), "{ended:?}");
    let notices: Vec<&str> = world
        .resource::<Collected<Notice>>()
        .0
        .iter()
        .map(|notice| notice.text.as_str())
        .collect();
    let stalls = notices
        .iter()
        .filter(|text| text.starts_with("Model stalled"))
        .count();
    assert_eq!(stalls, 1, "{notices:?}");
    assert!(
        notices
            .iter()
            .any(|text| text.starts_with("sleepy running")),
        "{notices:?}"
    );
    assert!(world.resource::<ShownLong>().0);
    let output = world.resource::<ToolOutputs>().0.get("c1").cloned();
    assert_eq!(output.as_deref(), Some("slept"));

    let paths = world.resource::<SessionPaths>();
    let log = std::fs::read_to_string(paths.path().join(FILE)).unwrap_or_default();
    let entries: Vec<serde_json::Value> = log
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let ended: Vec<(&str, &str)> = entries
        .iter()
        .map(|entry| {
            let what = entry["what"].as_str().unwrap_or_default();
            (what, entry["ended"].as_str().unwrap_or_default())
        })
        .collect();
    assert_eq!(ended, [("model", "retried"), ("tool", "finished")], "{log}");

    // The session's next app restores it without a word about its logs.
    drop(app);
    let (mut app, _) = app_in(&home, &session);
    app.add_plugins((AgentPlugin, SessionLogPlugin, StallWatchdogPlugin));
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let failed = notices.iter().any(|notice| notice.text.contains("restore"));
    assert!(!failed, "{notices:?}");
}

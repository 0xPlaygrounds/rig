use bevy_ecs::prelude::*;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ToolAdapter;
use rig_core::test_utils::MockCompletionModel;
use rig_core::tool::{Tool, ToolContext, ToolExecutionError};
use rig_ecs::effects::Handler;
use rig_ecs::prelude::*;
use rig_harness_test_support::{
    ToolOutputs, app_on, calls, connect, log_tool_outputs, reply, run_until,
};
use rig_memory::MemoryStore;
use rig_tools::Live;

use super::{CallOutput, ProgressPlugin, ToolProgress};

/// A build that reports two lines, then never ends.
struct Build(Live);

impl Tool for Build {
    const NAME: &'static str = "build";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Builds the project.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        _: serde_json::Value,
    ) -> Result<String, ToolExecutionError> {
        self.0.line("compiling a\n");
        self.0.line("compiling b\n");
        std::future::pending::<()>().await;
        Ok(String::new())
    }
}

/// Gives each `build` call a [`CallOutput`] and a handler that reports
/// into it.
fn follow_builds(
    mut calls: Query<(Entity, &mut CallHandler), (ToolCalls, Added<Admitted>)>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    for (call, mut handler) in &mut calls {
        let (output, live) = CallOutput::new(&wake);
        handler.0 = Handler(ErasedHandler::new(ToolAdapter::new(Build(live))));
        commands.entity(call).insert(output);
    }
}

#[test]
fn a_calls_reported_output_is_its_progress_and_what_it_answers_when_stopped() {
    let model = MockCompletionModel::from_stream_turns([
        calls("b1", "build", serde_json::json!({})),
        reply("unused"),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((AgentPlugin, ProgressPlugin))
        .add_tool(Build(Live::new(|_| {})))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .add_systems(AgentPass, follow_builds.in_set(PrepareSystems::Compose));
    app.finish();
    app.update();
    let Some(connection) = connect(&model) else {
        return;
    };
    let world = app.world_mut();
    let agent = world.spawn((Agent, connection)).id();
    world.trigger(Deliver::new(agent, "build it", DeliveryMode::Steer));
    let mut progress = world.query::<&ToolProgress>();
    run_until(&mut app, &wakes, |world| {
        let calls = world.try_query::<&ToolProgress>();
        calls.is_some_and(|mut calls| calls.iter(world).any(|line| line.0 == "compiling b"))
    });
    let lines: Vec<String> = progress
        .iter(app.world())
        .map(|line| line.0.clone())
        .collect();
    assert_eq!(lines, ["compiling b"]);
    app.world_mut().trigger(StopTurn::new(agent));
    app.update();
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let answer = outputs.get("b1").cloned().unwrap_or_default();
    assert_eq!(
        answer,
        "compiling a\ncompiling b\n[stopped before it finished]"
    );
}

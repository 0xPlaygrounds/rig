use std::path::PathBuf;
use std::time::{Duration, Instant};

use bevy_app::TaskPoolPlugin;
use rig_core::effect::{Answered, EffectId};
use rig_core::message::{ToolFunction, ToolName};
use rig_core::tool::{ToolContext, ToolExecutionError};

use bevy_ecs::system::ScheduleSystem;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ToolAdapter;
use rig_ecs::effects::Handler;

use super::AppPerCallExt;
use crate::prelude::*;

/// A tool `say` answering its word.
#[derive(Clone)]
struct Say(&'static str);

impl Tool for Say {
    const NAME: &'static str = "say";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Says a word.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        _: serde_json::Value,
    ) -> Result<String, ToolExecutionError> {
        Ok(self.0.to_owned())
    }
}

impl PerCallTool for Say {
    fn at(&self, _: CallPlace) -> Self {
        Self(self.0)
    }
}

/// A plugin replaces a per-call tool the documented way, with `Disabled`
/// on its entity and a tool of the same name: the new tool answers.
#[test]
fn a_disabled_per_call_tool_gives_way_to_its_replacement() {
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin))
        .add_tool_per_call(Say("old"), ToolOptions::default());
    let mut tools = app.world_mut().query::<(Entity, &ToolDef)>();
    let old: Vec<Entity> = tools
        .iter(app.world())
        .filter(|(_, tool)| tool.0.name == Say::NAME)
        .map(|(tool, _)| tool)
        .collect();
    for tool in old {
        app.world_mut().entity_mut(tool).insert(Disabled);
    }
    app.add_tool(Say("new"));
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    let Ok(name) = ToolName::new(Say::NAME) else {
        return;
    };
    let call = ToolCall::from_wire("s1", ToolFunction::new(name, serde_json::json!({})));
    let parent = Effect::with_parent_id(Some(EffectId::from_raw(1)));
    let call = world
        .spawn((ToolRequest { call }, parent, ForAgent(agent)))
        .id();
    let started = Instant::now();
    while app.world().get::<Outcome>(call).is_none() && started.elapsed() < Duration::from_secs(5) {
        app.update();
        std::thread::sleep(Duration::from_millis(5));
    }
    let said = app
        .world()
        .get::<Outcome>(call)
        .and_then(|outcome| match &outcome.0 {
            Ok(Answered::ToolResult { result }) => Some(result.output().render()),
            _ => None,
        });
    assert_eq!(said.as_deref(), Some("new"));
}

/// A tool `where` answering the base its call works from.
#[derive(Clone, Default)]
struct Where(PathBuf);

impl Tool for Where {
    const NAME: &'static str = "where";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Says where it works.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        _: serde_json::Value,
    ) -> Result<String, ToolExecutionError> {
        Ok(self.0.display().to_string())
    }
}

impl PerCallTool for Where {
    fn at(&self, place: CallPlace) -> Self {
        Self(place.base)
    }
}

/// Moves every per-call tool call to `/elsewhere`.
fn move_calls(mut calls: Query<&mut CallPlace, Added<Admitted>>) {
    for mut place in &mut calls {
        place.base = PathBuf::from("/elsewhere");
    }
}

/// Answers `name`'s call in a fresh app with `systems` in its pass.
fn answer_of<M>(
    name: &str,
    systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
) -> Option<String> {
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin))
        .add_tool_per_call(Where::default(), ToolOptions::default())
        .add_systems(AgentPass, systems);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    let call = ToolCall::from_wire(
        "w1",
        ToolFunction::new(ToolName::new(name).ok()?, serde_json::json!({})),
    );
    let parent = Effect::with_parent_id(Some(EffectId::from_raw(1)));
    let call = world
        .spawn((ToolRequest { call }, parent, ForAgent(agent)))
        .id();
    let started = Instant::now();
    while app.world().get::<Outcome>(call).is_none() && started.elapsed() < Duration::from_secs(5) {
        app.update();
        std::thread::sleep(Duration::from_millis(5));
    }
    app.world()
        .get::<Outcome>(call)
        .and_then(|outcome| match &outcome.0 {
            Ok(Answered::ToolResult { result }) => Some(result.output().render()),
            _ => None,
        })
}

#[test]
fn a_system_moves_a_call_before_its_handler_is_built() {
    let moved = move_calls
        .after(PerCallSystems::Place)
        .before(PerCallSystems::Build);
    assert_eq!(answer_of(Where::NAME, moved).as_deref(), Some("/elsewhere"));
}

#[test]
fn a_system_replaces_the_built_handler() {
    let replace = |mut calls: Query<&mut CallHandler, Added<Admitted>>| {
        for mut handler in &mut calls {
            let other = Where(PathBuf::from("replaced"));
            *handler = CallHandler(Handler(ErasedHandler::new(ToolAdapter::new(other))));
        }
    };
    let replace = replace
        .after(PerCallSystems::Build)
        .in_set(PrepareSystems::Compose);
    assert_eq!(answer_of(Where::NAME, replace).as_deref(), Some("replaced"));
}

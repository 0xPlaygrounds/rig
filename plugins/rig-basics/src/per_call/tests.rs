use std::time::{Duration, Instant};

use bevy_app::TaskPoolPlugin;
use rig_core::effect::{Answered, EffectId};
use rig_core::message::{ToolFunction, ToolName};
use rig_core::tool::{ToolContext, ToolExecutionError};

use super::AppPerCallExt;
use rig_harness::prelude::*;

/// A tool `say` answering its word.
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

/// A plugin replaces a per-call tool the documented way, with `Disabled`
/// on its entity and a tool of the same name: the new tool answers.
#[test]
fn a_disabled_per_call_tool_gives_way_to_its_replacement() {
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin))
        .add_tool_per_call(Say("old"), ToolOptions::default(), |_| Say("old"));
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

use bevy_app::App;
use bevy_ecs::prelude::*;
use rig_core::completion::Message;
use rig_core::effect::EffectId;
use rig_core::message::{ToolCall, ToolFunction, ToolName};

use super::{MESSAGE, SubagentsPlugin, TASK};
use crate::AgentPlugin;
use crate::agent::{
    ActiveTurn, Agent, AgentId, Ending, ModelChoice, Spawned, ToolCallRun, TurnOutcome,
};
use crate::inbox::{Deliver, DeliveryMode, Origin, RequestId};
use crate::tools::{Footprint, ToolCalled, ToolDef, ToolOutput};

/// The request of each message delivered, marked when it is a note, with
/// the agent it went to.
#[derive(Resource, Default)]
struct Delivered(Vec<(Entity, String)>);

fn record(sent: On<Deliver>, mut delivered: ResMut<Delivered>) {
    let request = sent.origin.request.clone().map(|request| request.0);
    let note = (sent.mode == DeliveryMode::Note).then_some(" (note)");
    let request = request.unwrap_or_default() + note.unwrap_or_default();
    delivered.0.push((sent.entity, request));
}

/// An app with the subagent tools and a parent agent, driven by hand: no
/// frame runs and no model is called, so a turn lasts until `end`.
struct Session(App, Entity);

impl Session {
    fn new() -> Self {
        let mut app = App::new();
        app.add_plugins((AgentPlugin, SubagentsPlugin))
            .init_resource::<Delivered>()
            .add_observer(record);
        let parent = (Agent, ModelChoice("a/b".to_owned()));
        let parent = app.world_mut().spawn(parent).id();
        Self(app, parent)
    }

    /// `agent` calls `tool` with `args` in call `id`, and gets its output.
    fn call(&mut self, agent: Entity, tool: &str, id: &str, args: serde_json::Value) -> String {
        let world = self.0.world_mut();
        let mut tools = world.query::<(Entity, &ToolDef)>();
        let found = tools
            .iter(world)
            .find(|(_, def)| def.0.name.as_str() == tool);
        let (Some((entity, _)), Ok(name)) = (found, ToolName::new(tool)) else {
            return String::new();
        };
        let call = ToolCall::from_wire(id, ToolFunction::new(name, args));
        let (parent, footprint) = (Some(EffectId::from_raw(1)), Footprint::Independent);
        let call = world
            .spawn(ToolCallRun {
                call,
                parent,
                footprint,
            })
            .id();
        world.trigger(ToolCalled {
            entity,
            call,
            agent,
        });
        world.flush();
        let output = world
            .get::<ToolOutput>(call)
            .map(|output| format!("{:?}", output.0));
        output.unwrap_or_default()
    }

    /// Starts a subagent with `peers` on the task `id`, in the batch of
    /// the parent's reply.
    fn task(&mut self, id: &str) -> Entity {
        let args = serde_json::json!({ "description": id, "prompt": "Work.", "peers": true });
        self.call(self.1, TASK, id, args);
        let spawned = self.0.world().get::<Spawned>(self.1);
        let child = spawned.and_then(|spawned| spawned.iter().last());
        child.unwrap_or(Entity::PLACEHOLDER)
    }

    /// `from` sends `to` a `message` in call `id`, and gets its output.
    fn message(&mut self, from: Entity, to: Entity, id: &str) -> String {
        let to = self.0.world().get::<AgentId>(to).map(AgentId::short);
        let args = serde_json::json!({ "agent": to, "text": "Hi." });
        self.call(from, MESSAGE, id, args)
    }

    /// Ends `agent`'s turn with an answer.
    fn end(&mut self, agent: Entity) {
        let world = self.0.world_mut();
        if let Some(turn) = world.get::<ActiveTurn>(agent).map(ActiveTurn::turn) {
            let mut turn = world.entity_mut(turn);
            turn.insert(Ending(TurnOutcome::Answered(Message::assistant("Done."))));
            turn.despawn();
        }
    }

    /// The requests of the messages delivered to `to`, in order.
    fn to(&self, to: Entity) -> Vec<&str> {
        let delivered = self.0.world().resource::<Delivered>().0.iter();
        let delivered = delivered.filter(|(entity, _)| *entity == to);
        delivered.map(|(_, request)| request.as_str()).collect()
    }
}

#[test]
fn report_header_names_the_subagent_and_task_not_the_request() {
    let origin = Origin::agent(
        AgentId("fa7bc7dc-8bed-4938".to_owned()),
        Some(RequestId("call_00_rIBjJGaY".to_owned())),
    )
    .titled("Fix the parser");
    assert_eq!(
        origin.header().as_deref(),
        Some("[Output of agent fa7bc7dc \"Fix the parser\", not the user's words]")
    );
    assert_eq!(
        origin.request,
        Some(RequestId("call_00_rIBjJGaY".to_owned()))
    );
}

#[test]
fn a_batch_reports_together_and_each_request_once() {
    let mut session = Session::new();
    let (a, b) = (session.task("ta"), session.task("tb"));
    let sent = session.message(a, b, "m1");
    assert!(sent.contains("Sent to peer"), "{sent}");
    session.end(b);
    assert_eq!(session.to(a), ["ta", "m1"]);
    assert!(session.to(session.1).is_empty());
    // A follow-up and the task, answered together by one turn.
    session.message(session.1, a, "f1");
    session.end(a);
    assert_eq!(session.to(session.1), ["ta (note)", "f1", "tb"]);
}

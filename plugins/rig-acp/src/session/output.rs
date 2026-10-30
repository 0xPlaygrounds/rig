//! What the main agent does, as session updates: its streamed text and
//! reasoning, its tool calls, and what rig says to the user meanwhile.

use agent_client_protocol::schema::v1 as acp;
use bevy_ecs::message::{MessageCursor, Messages};
use bevy_ecs::system::SystemParam;
use rig_harness::prelude::*;
use rig_harness::services::WorkDir;

use super::{Session, questions};
use crate::translate;

/// Where reading the streamed deltas got to.
#[derive(Resource, Default)]
pub(super) struct Read(MessageCursor<StreamDelta>);

/// The session's output: each update goes after the text streamed before
/// it, so the client sees them in the order they happened.
#[derive(SystemParam)]
pub(super) struct Out<'w, 's> {
    pub(super) session: ResMut<'w, Session>,
    deltas: Res<'w, Messages<StreamDelta>>,
    read: ResMut<'w, Read>,
    pub(super) primary: MainAgentQuery<'w, 's>,
    calls: Query<'w, 's, (), With<TurnCall>>,
}

impl Out<'_, '_> {
    /// Sends the main agent's text and reasoning streamed so far, each run
    /// of one kind as one chunk.
    pub(super) fn stream(&mut self) {
        let agent = main_agent(&self.primary);
        let mut runs: Vec<(bool, String)> = Vec::new();
        for delta in self.read.0.read(&self.deltas) {
            if delta.agent != agent || !self.calls.contains(delta.effect) {
                continue;
            }
            let (thought, text) = match &delta.event {
                StreamEvent::Text { text, .. } => (false, text),
                StreamEvent::Reasoning { text, .. } => (true, text),
                _ => continue,
            };
            match runs.last_mut() {
                Some((kind, run)) if *kind == thought => run.push_str(text),
                _ => runs.push((thought, text.clone())),
            }
        }
        for (thought, text) in runs {
            let chunk = acp::ContentChunk::new(acp::ContentBlock::from(text));
            self.session.send(match thought {
                true => acp::SessionUpdate::AgentThoughtChunk(chunk),
                false => acp::SessionUpdate::AgentMessageChunk(chunk),
            });
        }
    }

    /// Sends `update` after the text streamed before it.
    pub(super) fn send(&mut self, update: acp::SessionUpdate) {
        self.stream();
        self.session.send(update);
    }

    /// Whether `agent` is the main agent.
    fn main(&self, agent: Entity) -> bool {
        main_agent(&self.primary) == Some(agent)
    }
}

/// Sends what was streamed this frame.
pub(super) fn stream(mut out: Out) {
    out.stream();
}

/// A call of the main agent's model asked for: pending. One the session
/// restored is not announced again: its history has it, under its id.
pub(super) fn call_asked(
    asked: On<Add<ToolRequest>>,
    calls: Query<(&ToolRequest, &ForAgent, Has<Restored>)>,
    agents: Query<(&AgentId, &Conversation)>,
    dirs: Query<&WorkDir>,
    root: Res<ProjectRoot>,
    mut out: Out,
) {
    if let Ok((request, &ForAgent(agent), restored)) = calls.get(asked.entity)
        && let Ok((AgentId(of), conversation)) = agents.get(agent)
        && out.main(agent)
    {
        let id = translate::call_id_in(of, conversation, &request.call.id);
        out.session.calls.insert(asked.entity, id.clone());
        if restored {
            return;
        }
        let cwd = translate::cwd_of(dirs.get(agent).ok(), &root);
        let call = translate::tool_call(&request.call, id, &cwd);
        out.send(acp::SessionUpdate::ToolCall(call));
    }
}

/// A call of the main agent starts: in progress. A question starts: asked.
pub(super) fn started(
    started: On<Add<Started>>,
    calls: Query<(), With<ToolRequest>>,
    asking: questions::Asking,
    mut out: Out,
) {
    if calls.contains(started.entity) {
        if let Some(id) = out.session.calls.get(&started.entity).cloned() {
            let update = translate::tool_status(id, acp::ToolCallStatus::InProgress);
            out.send(acp::SessionUpdate::ToolCallUpdate(update));
        }
        return;
    }
    asking.ask(started.entity, &mut out);
}

/// A call of the main agent settles: completed or failed, with what it
/// answered. A question settles: no longer asked.
pub(super) fn settled(
    settled: On<Add<Settled>>,
    calls: Query<&Outcome, With<ToolRequest>>,
    mut out: Out,
) {
    if let Some(task) = out.session.asking.remove(&settled.entity) {
        task.abort();
    }
    if let Ok(outcome) = calls.get(settled.entity)
        && let Some(id) = out.session.calls.remove(&settled.entity)
    {
        let result = outcome.tool_result();
        let done = translate::tool_done(&id, !result.is_success(), result.output().render());
        out.send(acp::SessionUpdate::ToolCallUpdate(done));
    }
}

/// What rig says about the main agent or the app, such as a command's
/// output or an MCP server that could not connect, and why a command was
/// refused.
pub(super) fn notices(
    mut notices: MessageReader<Notice>,
    mut recalled: MessageReader<Recalled>,
    mut out: Out,
) {
    let agent = main_agent(&out.primary);
    let said: Vec<_> = notices
        .read()
        .filter(|notice| notice.agent.is_none() || notice.agent == agent)
        .map(|notice| (notice.text.clone(), notice.level))
        .collect();
    let cancelled = out
        .session
        .prompt
        .as_ref()
        .is_some_and(|prompt| prompt.cancelled);
    let refused: Vec<_> = recalled
        .read()
        .filter(|recalled| Some(recalled.agent) == agent && !cancelled)
        .filter_map(|recalled| recalled.why.clone())
        .map(|why| (why, NoticeLevel::Error))
        .collect();
    if said.is_empty() && refused.is_empty() {
        return;
    }
    out.stream();
    for (text, level) in said.into_iter().chain(refused) {
        out.session.note(text, level);
    }
}

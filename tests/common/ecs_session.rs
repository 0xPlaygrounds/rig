//! Actual native run output and ordered history for recorded tool sessions.
#[path = "ecs_session/tests.rs"]
mod tests;
use crate::ecs_agent::EcsAgent;
use anyhow::{Result, anyhow};
use bevy_ecs::prelude::*;
use rig::message::Message;
use rig_ecs::{
    agent::{Order, Parts, Utterance},
    systems::spawn_run,
};

pub(crate) struct SessionResult {
    pub output: String,
    pub history: Vec<Message>,
}

/// The typed Native surface accepts a JSON value surrounded by prose or fences.
/// This is deserialization only; it does not drive either agent runtime.
pub(crate) fn parse_native_output<T: serde::de::DeserializeOwned>(
    text: &str,
) -> Result<T, serde_json::Error> {
    let trimmed = text.trim();
    match serde_json::from_str(trimmed) {
        Ok(value) => Ok(value),
        Err(direct_error) => {
            let Some(start) = trimmed.find(['{', '[']) else {
                return Err(direct_error);
            };
            serde_json::Deserializer::from_str(&trimmed[start..])
                .into_iter::<T>()
                .next()
                .unwrap_or(Err(direct_error))
        }
    }
}

pub(crate) async fn run_session(
    ecs: &mut EcsAgent,
    prompt: &str,
    streamed: bool,
    max_turns: Option<usize>,
) -> Result<SessionResult> {
    let run = spawn_run(
        ecs.app.world_mut(),
        ecs.agent,
        &[],
        prompt,
        streamed,
        max_turns,
    );
    let output = ecs
        .wait_for_outcome(run)
        .await
        .map_err(|error| anyhow!("native session failed: {error:?}"))?;
    let mut history: Vec<_> = ecs
        .app
        .world_mut()
        .query_filtered::<(&ChildOf, &Order, &Parts), With<Utterance>>()
        .iter(ecs.app.world())
        .filter(|(parent, _, _)| parent.parent() == run)
        .map(|(_, order, parts)| (order.0, parts.0.to_message()))
        .collect();
    history.sort_by_key(|(order, _)| *order);
    Ok(SessionResult {
        output,
        history: history.into_iter().map(|(_, message)| message).collect(),
    })
}

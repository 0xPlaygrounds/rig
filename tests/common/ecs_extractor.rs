//! Native typed extraction for the recorded, zero-retry extractor scenarios.
//!
//! Configuration becomes native components. Each call spawns an independent
//! run; deserialization and usage read its actual result, never expected data.

use std::marker::PhantomData;

use anyhow::{Result, anyhow, ensure};
use bevy_ecs::prelude::*;
use rig_core::{
    completion::CompletionModel,
    message::{AssistantContent, Message, ToolChoice},
};
use rig_ecs::{
    agent::{
        AdditionalParams, DefaultMaxTurns, InvalidCalls, Output, OutputKind, OutputToolConfig,
        Outputs, Turn, Unhandled, Usage,
    },
    systems::spawn_run,
};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::ecs_agent::EcsAgent;

#[path = "ecs_extractor/preamble.rs"]
mod preamble;
use preamble::PREAMBLE;

pub(crate) struct Extracted<T> {
    pub output: T,
    pub usage: rig_core::completion::Usage,
}

pub(crate) struct EcsExtractor<T> {
    ecs: EcsAgent,
    output: PhantomData<T>,
}

impl<T: JsonSchema + DeserializeOwned> EcsExtractor<T> {
    pub fn new(
        model: impl CompletionModel + 'static,
        extra_preamble: Option<&str>,
        params: Option<serde_json::Value>,
    ) -> Self {
        let preamble = extra_preamble.map_or_else(
            || PREAMBLE.to_owned(),
            |extra| {
                format!(
                    "{PREAMBLE}\n\n=============== ADDITIONAL INSTRUCTIONS ===============\n{extra}"
                )
            },
        );
        let mut ecs = EcsAgent::new(model, &preamble, 1);
        ecs.app.world_mut().entity_mut(ecs.agent).insert((
            DefaultMaxTurns(None),
            AdditionalParams(params),
            rig_ecs::agent::ToolChoiceSpec(Some(ToolChoice::Required)),
            Output {
                mode: OutputKind::Tool,
                schema: Some(schemars::schema_for!(T).into()),
            },
            OutputToolConfig {
                name: Some("submit".into()),
                description: Some(
                    "Submit the structured data you extracted from the provided text.".into(),
                ),
                augment_preamble: false,
            },
            InvalidCalls {
                retries: 0,
                unhandled: Unhandled::Ignore,
            },
        ));
        Self {
            ecs,
            output: PhantomData,
        }
    }

    pub async fn extract(&mut self, prompt: &str, history: &[Message]) -> Result<Extracted<T>> {
        let history = history
            .iter()
            .map(|message| {
                rig_ecs::agent::MessageParts::from_message(message).ok_or_else(|| {
                    anyhow!("system history requires an explicit native preamble mapping")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let run = spawn_run(
            self.ecs.app.world_mut(),
            self.ecs.agent,
            &history,
            prompt,
            false,
            Some(1),
        );
        let text = self
            .ecs
            .wait_for_outcome(run)
            .await
            .map_err(|failure| anyhow!("native extraction failed: {failure:?}"))?;
        // The original TypedRun rejects a text-only response even if it is
        // valid JSON. Require a real submit call belonging to this run.
        let has_submission = self
            .ecs
            .app
            .world_mut()
            .query_filtered::<(&ChildOf, &Outputs), With<Turn>>()
            .iter(self.ecs.app.world())
            .any(|(parent, outputs)| {
                parent.parent() == run && outputs.content.iter().any(|part| {
                matches!(part, AssistantContent::ToolCall(call) if call.function.name == "submit")
            })
            });
        ensure!(has_submission, "the output tool was not called");
        let usage = self
            .ecs
            .app
            .world()
            .get::<Usage>(run)
            .ok_or_else(|| anyhow!("settled extraction has no usage component"))?
            .0;
        Ok(Extracted {
            output: serde_json::from_str(&text)?,
            usage,
        })
    }
}

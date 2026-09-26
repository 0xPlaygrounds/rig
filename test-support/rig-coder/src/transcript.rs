//! A JSONL transcript written as the run happens, so a killed or timed-out
//! run still leaves evidence. One object per line, tagged by `kind`:
//! `task`, `assistant`, `usage`, `tool_call`, `tool_result`, `settled`, `failed`.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rig::agent::{
    AgentHook, DispatchAction, DispatchEvent, HookContext, ModelTurnAction, ModelTurnFinished,
    OutcomeAction, OutcomeEvent,
};
use rig::completion::Usage;
use rig::effect::EffectKind;
use serde_json::{Value, json};

/// Characters of a tool result kept in the transcript; the effect log keeps all of it.
const RESULT_CHARS: usize = 4_000;

pub struct Transcript {
    file: Option<Mutex<File>>,
    started: Instant,
}

impl Transcript {
    /// A transcript at `path`, or one that discards events when `path` is `None`.
    pub fn create(path: Option<&Path>) -> std::io::Result<Arc<Self>> {
        let file = match path {
            Some(path) => {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                Some(Mutex::new(File::create(path)?))
            }
            None => None,
        };
        Ok(Arc::new(Self {
            file,
            started: Instant::now(),
        }))
    }

    pub fn event(&self, kind: &str, mut fields: Value) {
        let Some(file) = &self.file else {
            return;
        };
        if let Value::Object(map) = &mut fields {
            map.insert("kind".into(), kind.into());
            map.insert(
                "t".into(),
                json!((self.started.elapsed().as_millis() as f64) / 1000.0),
            );
        }
        if let (Ok(mut file), Ok(line)) = (file.lock(), serde_json::to_string(&fields)) {
            let _written = writeln!(file, "{line}").and_then(|()| file.flush());
        }
    }
}

pub fn usage_fields(usage: &Usage) -> Value {
    json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
    })
}

/// Appends every model turn, tool call and tool result to the transcript.
pub struct TranscriptHook(pub Arc<Transcript>);

impl AgentHook for TranscriptHook {
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        self.0.event(
            "assistant",
            json!({
                "turn": event.turn,
                "content": event.content,
                "finish_reason": event.finish_reason.map(|reason| format!("{reason:?}")),
            }),
        );
        let mut usage = usage_fields(&event.usage);
        if let Value::Object(map) = &mut usage {
            map.insert("turn".into(), event.turn.into());
        }
        self.0.event("usage", usage);
        ModelTurnAction::continue_run()
    }

    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if let EffectKind::ToolCall { name, args } = event.kind {
            self.0.event(
                "tool_call",
                json!({ "turn": event.turn, "name": name, "args": args }),
            );
        }
        DispatchAction::proceed()
    }

    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if let EffectKind::ToolCall { name, .. } = event.kind {
            let (ok, output) = match event.outcome {
                Ok(outcome) => (
                    true,
                    serde_json::to_string(outcome).unwrap_or_else(|error| error.to_string()),
                ),
                Err(report) => (false, report.to_string()),
            };
            let output: String = output.chars().take(RESULT_CHARS).collect();
            self.0.event(
                "tool_result",
                json!({ "turn": event.turn, "name": name, "ok": ok, "output": output }),
            );
        }
        OutcomeAction::proceed()
    }
}

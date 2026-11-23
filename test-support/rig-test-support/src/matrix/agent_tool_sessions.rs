//! Shared agent tool sessions: the complex-argument tools, their checks, and
//! the bodies each wire runs over its own scenario rows.
//!
//! The invoking module defines `SESSION_MODEL`, `SESSION_MAX_TOKENS` (an
//! `Option<u64>` output cap for wires whose rate limiter rejects the default)
//! and `COMPLEX_SESSION`, its [`ComplexSession`].

use std::sync::{Arc, Mutex};

use rig_agent::{Agent, AgentBuilder};
use rig_core::message::{AssistantContent, Message, UserContent};
use rig_core::tool::{Tool, ToolContext};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal, TWO_TOOL_STREAM_PROMPT,
    assert_contains_all_case_insensitive, collect_stream_observation,
};

/// One wire's production-readiness session: the texts it sends and the
/// arguments its tools must then receive.
pub struct ComplexSession {
    /// The system preamble naming the tool order.
    pub preamble: &'static str,
    /// The user prompt carrying the note and the escaped text.
    pub prompt: &'static str,
    /// The `project` the manifest tool must receive.
    pub project: &'static str,
    /// The `note` the manifest tool must receive.
    pub note: &'static str,
    /// The `text` the echo tool must receive.
    pub escaped_text: &'static str,
    /// The schema of [`OptionalNullableProbe`]'s `note` when the session calls it.
    pub optional_probe: Option<ProbeNote>,
}

/// The JSON type [`OptionalNullableProbe`] declares for its optional `note`.
#[derive(Clone, Copy)]
pub enum ProbeNote {
    /// `["string", "null"]`.
    Nullable,
    /// `"string"`.
    String,
}

impl ComplexSession {
    /// `builder` with the session preamble and its tools, in call order.
    pub fn equip(&self, builder: AgentBuilder, log: &InvocationLog) -> AgentBuilder {
        let builder = builder
            .preamble(self.preamble)
            .tool(PingEmpty(log.clone()))
            .tool(InspectManifest(log.clone()))
            .tool(JoinLabels(log.clone()));
        let builder = match self.optional_probe {
            Some(note) => builder.tool(OptionalNullableProbe {
                log: log.clone(),
                note,
            }),
            None => builder,
        };
        builder.tool(EscapeEcho(log.clone()))
    }

    /// The tool names in call order.
    pub fn tool_names(&self) -> Vec<&'static str> {
        let mut names = vec![PingEmpty::NAME, InspectManifest::NAME, JoinLabels::NAME];
        names.extend(self.optional_probe.map(|_| OptionalNullableProbe::NAME));
        names.push(EscapeEcho::NAME);
        names
    }

    fn answer_tokens(&self) -> Vec<&'static str> {
        let mut tokens = vec!["EMPTY-OK", "MANIFEST-OK", "LABELS-OK"];
        tokens.extend(self.optional_probe.map(|_| "OPTIONAL-OK"));
        tokens.push("ESCAPE-OK");
        tokens
    }

    /// Assert `log` holds one call of each tool, in order, with this session's arguments.
    pub fn assert_invocations(&self, log: &InvocationLog) {
        let invocations = log
            .lock()
            .expect("tool invocation log lock should not be poisoned")
            .clone();
        let names = invocations.iter().map(|call| call.name).collect::<Vec<_>>();
        assert_eq!(
            names,
            self.tool_names(),
            "expected one complex tool call of each shape in order"
        );

        assert_eq!(invocations[0].args, json!({}));
        let manifest = &invocations[1].args;
        assert_eq!(manifest["project"], self.project);
        assert_eq!(manifest["flags"], json!({"critical": true, "retries": 2}));
        assert_eq!(manifest["steps"].as_array().map(Vec::len), Some(2));
        assert_eq!(manifest["note"], self.note);
        assert_eq!(
            invocations[2].args,
            json!({
                "labels": ["north", "beta gamma", "quote:\"delta\"", "slash\\path"],
                "separator": "|"
            })
        );
        if self.optional_probe.is_some() {
            let probe = &invocations[3].args;
            assert_eq!(probe["name"], "sentinel");
            assert!(
                probe.get("note").is_none() || probe["note"].is_null() || probe["note"].is_string(),
                "optional note should be omitted, null, or a string when supplied"
            );
            assert!(probe["nullable_code"].is_null());
        }
        assert_eq!(invocations[names.len() - 1].args["text"], self.escaped_text);
    }
}

/// Run `session` unary on `agent`, built from [`ComplexSession::equip`], and
/// check the answer, the tool arguments and the caller-owned history.
pub async fn chat_complex_session(
    agent: &Agent,
    session: &ComplexSession,
    log: &InvocationLog,
) -> anyhow::Result<()> {
    let mut history = Vec::<Message>::new();
    let response = agent.chat(session.prompt, &mut history).await?;
    assert_contains_all_case_insensitive(&response.output(), &session.answer_tokens());
    session.assert_invocations(log);
    assert_history_records_sequential_tool_roundtrips(&history, &session.tool_names());
    Ok(())
}

/// Stream `session` on `agent`, built from [`ComplexSession::equip`], and
/// check the streamed calls, results, answer and tool arguments.
pub async fn stream_complex_session(
    agent: &Agent,
    session: &ComplexSession,
    log: &InvocationLog,
) -> anyhow::Result<()> {
    let mut stream = agent
        .prompt(session.prompt)
        .history(Vec::<Message>::new())
        .max_turns(10)
        .stream();
    let observation = collect_stream_observation(&mut stream).await;
    anyhow::ensure!(
        observation.errors.is_empty(),
        "stream should not emit errors: {:?}",
        observation.errors
    );
    let names = session.tool_names();
    anyhow::ensure!(
        observation.tool_calls == names,
        "stream should expose ordered tool calls, saw {:?}",
        observation.tool_calls
    );
    anyhow::ensure!(
        observation.tool_results == names.len(),
        "expected {} streamed tool results, saw {}",
        names.len(),
        observation.tool_results
    );
    let response = observation
        .final_response_text
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("stream should produce final response text"))?;
    assert_contains_all_case_insensitive(response, &session.answer_tokens());
    session.assert_invocations(log);
    Ok(())
}

/// Chat the two-signal prompt on `agent`, which offers [`AlphaSignal`] and
/// [`BetaSignal`] in parallel, and check both calls share one assistant message.
pub async fn chat_parallel_signals(agent: &Agent) -> anyhow::Result<()> {
    let mut history = Vec::<Message>::new();
    let response = agent.chat(TWO_TOOL_STREAM_PROMPT, &mut history).await?;
    assert_contains_all_case_insensitive(
        &response.output(),
        &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
    );
    let calls = history_tool_calls(&history);
    let call_names = calls
        .iter()
        .map(|(_, name)| name.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        calls.len() == 2
            && call_names.contains(&AlphaSignal::NAME)
            && call_names.contains(&BetaSignal::NAME),
        "expected both zero-argument tools, saw {call_names:?}"
    );
    anyhow::ensure!(
        calls[0].0 == calls[1].0,
        "parallel tool calls should be recorded on one assistant message"
    );
    anyhow::ensure!(
        history_tool_results(&history).len() == 2,
        "expected two tool results"
    );
    Ok(())
}

/// The message index and tool name of every tool call in `history`.
fn history_tool_calls(history: &[Message]) -> Vec<(usize, String)> {
    let mut calls = Vec::new();
    for (index, message) in history.iter().enumerate() {
        if let Message::Assistant(rig_core::message::AssistantMessage { content, .. }) = message {
            for item in content.iter() {
                if let AssistantContent::ToolCall(tool_call) = item {
                    calls.push((index, tool_call.function.name.clone().into()));
                }
            }
        }
    }
    calls
}

/// The message index of every tool result in `history`.
fn history_tool_results(history: &[Message]) -> Vec<usize> {
    let mut results = Vec::new();
    for (index, message) in history.iter().enumerate() {
        if let Message::User { content } = message {
            for item in content.iter() {
                if let UserContent::ToolResult(_) = item {
                    results.push(index);
                }
            }
        }
    }
    results
}

fn assert_history_records_sequential_tool_roundtrips(history: &[Message], expected_tools: &[&str]) {
    let calls = history_tool_calls(history);
    let results = history_tool_results(history);
    assert_eq!(
        calls
            .iter()
            .map(|(_, name)| name.as_str())
            .collect::<Vec<_>>(),
        expected_tools,
        "caller-owned chat history should preserve tool call order"
    );
    assert_eq!(
        results.len(),
        expected_tools.len(),
        "caller-owned chat history should contain one tool result per call"
    );
    for (index, (call_index, _)) in calls.iter().enumerate() {
        let result_index = results[index];
        assert!(
            *call_index < result_index,
            "tool result should follow its assistant tool call"
        );
        if let Some((next_call_index, _)) = calls.get(index + 1) {
            assert!(
                result_index < *next_call_index,
                "next tool call should occur after the previous tool result"
            );
        }
    }
}

/// One recorded tool call: the tool and the arguments it received.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolInvocation {
    name: &'static str,
    args: serde_json::Value,
}

/// The tool calls a session's tools record, in call order.
pub type InvocationLog = Arc<Mutex<Vec<ToolInvocation>>>;

fn push_invocation<T: Serialize>(log: &InvocationLog, name: &'static str, args: &T) {
    log.lock()
        .expect("tool invocation log lock should not be poisoned")
        .push(ToolInvocation {
            name,
            args: serde_json::to_value(args).expect("tool args should serialize"),
        });
}

/// Zero-argument tool returning `EMPTY-OK`.
#[derive(Clone, Default)]
pub struct PingEmpty(pub InvocationLog);

/// Tool validating a nested deployment manifest.
#[derive(Clone, Default)]
pub struct InspectManifest(pub InvocationLog);

/// Tool joining labels with a separator.
#[derive(Clone, Default)]
pub struct JoinLabels(pub InvocationLog);

/// Tool with an optional and a nullable argument.
#[derive(Clone)]
pub struct OptionalNullableProbe {
    log: InvocationLog,
    note: ProbeNote,
}

/// Tool echoing escaping-sensitive text.
#[derive(Clone, Default)]
pub struct EscapeEcho(pub InvocationLog);

/// Arguments of [`PingEmpty`].
#[derive(Debug, Deserialize, Serialize)]
pub struct EmptyArgs {}

/// Arguments of [`InspectManifest`].
#[derive(Debug, Deserialize, Serialize)]
pub struct ManifestArgs {
    project: String,
    flags: ManifestFlags,
    steps: Vec<ManifestStep>,
    note: String,
}

/// The manifest's flags.
#[derive(Debug, Deserialize, Serialize)]
pub struct ManifestFlags {
    critical: bool,
    retries: u8,
}

/// One manifest step.
#[derive(Debug, Deserialize, Serialize)]
pub struct ManifestStep {
    name: String,
    weight: i32,
}

/// Arguments of [`JoinLabels`].
#[derive(Debug, Deserialize, Serialize)]
pub struct JoinArgs {
    labels: Vec<String>,
    separator: String,
}

/// Arguments of [`OptionalNullableProbe`].
#[derive(Debug, Deserialize, Serialize)]
pub struct OptionalNullableArgs {
    name: String,
    #[serde(default)]
    note: Option<String>,
    #[serde(default)]
    nullable_code: Option<String>,
}

/// Arguments of [`EscapeEcho`].
#[derive(Debug, Deserialize, Serialize)]
pub struct EchoArgs {
    text: String,
}

/// The error type of the session tools, which never fail.
#[derive(Debug, thiserror::Error)]
#[error("session tool error")]
pub struct SessionToolError;

impl Tool for PingEmpty {
    const NAME: &'static str = "ping_empty";
    type Error = SessionToolError;
    type Args = EmptyArgs;
    type Output = String;

    fn description(&self) -> String {
        "Return EMPTY-OK. This tool takes no arguments.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        push_invocation(&self.0, Self::NAME, &args);
        Ok("EMPTY-OK".to_string())
    }
}

impl Tool for InspectManifest {
    const NAME: &'static str = "inspect_manifest";
    type Error = SessionToolError;
    type Args = ManifestArgs;
    type Output = String;

    fn description(&self) -> String {
        "Validate a nested deployment manifest.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "project": { "type": "string" },
                "flags": {
                    "type": "object",
                    "properties": {
                        "critical": { "type": "boolean" },
                        "retries": { "type": "integer" }
                    },
                    "required": ["critical", "retries"]
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "weight": { "type": "integer" }
                        },
                        "required": ["name", "weight"]
                    }
                },
                "note": { "type": "string" }
            },
            "required": ["project", "flags", "steps", "note"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        push_invocation(&self.0, Self::NAME, &args);
        Ok(format!(
            "MANIFEST-OK project={} steps={} retries={}",
            args.project,
            args.steps.len(),
            args.flags.retries
        ))
    }
}

impl Tool for JoinLabels {
    const NAME: &'static str = "join_labels";
    type Error = SessionToolError;
    type Args = JoinArgs;
    type Output = String;

    fn description(&self) -> String {
        "Join label strings with the requested separator.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "labels": { "type": "array", "items": { "type": "string" } },
                "separator": { "type": "string" }
            },
            "required": ["labels", "separator"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        push_invocation(&self.0, Self::NAME, &args);
        Ok(format!("LABELS-OK {}", args.labels.join(&args.separator)))
    }
}

impl Tool for OptionalNullableProbe {
    const NAME: &'static str = "optional_nullable_probe";
    type Error = SessionToolError;
    type Args = OptionalNullableArgs;
    type Output = String;

    fn description(&self) -> String {
        "Validate optional and nullable argument serialization.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        let note = match self.note {
            ProbeNote::Nullable => json!(["string", "null"]),
            ProbeNote::String => json!("string"),
        };
        json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "note": { "type": note },
                "nullable_code": { "type": ["string", "null"] }
            },
            "required": ["name", "nullable_code"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        push_invocation(&self.log, Self::NAME, &args);
        Ok(format!(
            "OPTIONAL-OK name={} note={} nullable={}",
            args.name,
            args.note.unwrap_or_else(|| "missing".to_string()),
            args.nullable_code.unwrap_or_else(|| "null".to_string())
        ))
    }
}

impl Tool for EscapeEcho {
    const NAME: &'static str = "escape_echo";
    type Error = SessionToolError;
    type Args = EchoArgs;
    type Output = String;

    fn description(&self) -> String {
        "Echo a string containing escaping-sensitive characters.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        push_invocation(&self.0, Self::NAME, &args);
        Ok(format!("ESCAPE-OK {}", args.text))
    }
}

/// The session agent for `client`: `SESSION_MODEL`, capped at
/// `SESSION_MAX_TOKENS` when the invoking wire sets one.
#[doc(hidden)]
#[macro_export]
macro_rules! session_agent {
    ($client:expr) => {{
        let agent = rig::AgentBuilder::new($client.completion(SESSION_MODEL));
        match SESSION_MAX_TOKENS {
            Some(max_tokens) => agent.max_tokens(max_tokens),
            None => agent,
        }
    }};
}

/// Emit registered test rows with the shared execution body.
#[macro_export]
macro_rules! agent_tool_sessions_case {
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, sequential_complex_tool_calls_nonstreaming) => {
        $(#[$attribute])*
        async fn $name() -> Result<()> {
            $wrapper($scenario, |client| async move {
                let log = $crate::matrix::agent_tool_sessions::InvocationLog::default();
                let agent = COMPLEX_SESSION
                    .equip($crate::matrix::session_agent!(client), &log)
                    .parallel_tool_calls(false)
                    .default_max_turns(10)
                    .build();
                $crate::matrix::agent_tool_sessions::chat_complex_session(&agent, &COMPLEX_SESSION, &log).await
            })
            .await
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, sequential_complex_tool_calls_streaming) => {
        $(#[$attribute])*
        async fn $name() -> Result<()> {
            $wrapper($scenario, |client| async move {
                let log = $crate::matrix::agent_tool_sessions::InvocationLog::default();
                let agent = COMPLEX_SESSION
                    .equip($crate::matrix::session_agent!(client), &log)
                    .parallel_tool_calls(false)
                    .build();
                $crate::matrix::agent_tool_sessions::stream_complex_session(&agent, &COMPLEX_SESSION, &log).await
            })
            .await
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, parallel_tool_calls_single_turn_nonstreaming) => {
        $(#[$attribute])*
        async fn $name() -> Result<()> {
            $wrapper($scenario, |client| async move {
                let agent = $crate::matrix::session_agent!(client)
                    .preamble(TWO_TOOL_STREAM_PREAMBLE)
                    .tool(AlphaSignal)
                    .tool(BetaSignal)
                    .parallel_tool_calls(true)
                    .default_max_turns(5)
                    .build();
                $crate::matrix::agent_tool_sessions::chat_parallel_signals(&agent).await
            })
            .await
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, parallel_tool_calls_single_turn_streaming) => {
        $(#[$attribute])*
        async fn $name() -> Result<()> {
            $wrapper($scenario, |client| async move {
                let agent = $crate::matrix::session_agent!(client)
                    .preamble(TWO_TOOL_STREAM_PREAMBLE)
                    .tool(AlphaSignal)
                    .tool(BetaSignal)
                    .parallel_tool_calls(true)
                    .build();
                let mut stream = agent.prompt(TWO_TOOL_STREAM_PROMPT).max_turns(5).stream();
                let observation = collect_stream_observation(&mut stream).await;
                assert_two_tool_roundtrip_contract(
                    &observation,
                    &[AlphaSignal::NAME, BetaSignal::NAME],
                    &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
                );
                Ok(())
            })
            .await
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, long_history_replay_with_tool_result_continuation) => {
        $(#[$attribute])*
        async fn $name() -> Result<()> {
            $wrapper ($scenario , | client | async move { let model = client . completion (SESSION_MODEL) ; let tool_call_id = "call_REDACTED_1" ; let request = rig::completion::CompletionRequest::new ("Answer in one short sentence: what is my favorite color, which label came from the tool, and which release lane did I choose? Do not call any tools." ,) . preamble ("You are concise and should rely on the provided chat history." . to_string ()) . message (Message :: user ("My favorite color is teal. Please remember it.")) . message (Message :: assistant ("Noted: your favorite color is teal.")) . message (Message :: user ("For this release, use the canary lane.")) . message (Message :: assistant ("Understood: the release lane is canary.")) . message (Message :: user ("Look up the harbor label with the tool.")) . message (Message :: Assistant (rig_core::message::AssistantMessage::new (vec ! [AssistantContent :: tool_call (tool_call_id , rig_core::message::ToolName::new(AlphaSignal :: NAME).expect("tool name") , json ! ({ }) ,)]))) . message (Message::tool_result(rig_core::message::CallId::from_wire(tool_call_id), rig_core::message::ToolName::new(AlphaSignal::NAME).expect("tool name"), ALPHA_SIGNAL_OUTPUT)) . message (Message :: assistant ("The harbor label is crimson-harbor.")) . tool (rig :: tool :: tool_definition (& AlphaSignal)) . tool_choice (ToolChoice :: None) . max_tokens (SESSION_MAX_TOKENS) ; let (raw , response) = raw_and_normalized_completion (& model , request) . await ? ; let text = assistant_text_response (& response . choice) . ok_or_else (| | anyhow :: anyhow ! ("response should include assistant text")) ? ; assert_contains_all_case_insensitive (& text , & ["teal" , ALPHA_SIGNAL_OUTPUT , "canary"]) ; assert_response_metadata (& response , & raw) ; Ok (()) } ,) . await
        }
    };
}

pub use agent_tool_sessions_case;
pub use session_agent;

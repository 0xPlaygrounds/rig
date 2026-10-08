//! Code mode for the rig-harness agent. [`CodeModePlugin`] adds the
//! `run_code` tool: the model writes one Python script that spawns agents,
//! sends them requests, awaits their replies, calls the model's own tools,
//! and loops and branches over the results; the value of the script's last
//! expression is the tool's output.
//!
//! The script runs in [Pydantic Monty](https://github.com/pydantic/monty),
//! a sandboxed Python interpreter written in Rust, embedded in process. It
//! reaches the host only through four host functions, each one call of the
//! [`Harness`] handle: `spawn_agent`, `send`, `reply` and `call_tool`. It
//! has no file system, network or environment. Its execution time,
//! recursion, single allocations and host calls are limited, and Esc
//! cancels it like any tool call. A crash-level abort of the interpreter
//! (such as a native stack overflow) takes the agent down with it; the
//! launcher resumes the session, as after any crash.
//!
//! The plugin is written only against rig-harness's public primitives: an
//! open tool ([`AppToolsExt::add_open_tool`]) whose call is completed by a
//! [`Running`] task, and the [`Harness`] handle, so the calls a script makes
//! go through the one recorded dispatch and are recorded under its
//! `run_code` call. It is not in the default `plugins.toml`; add it with
//!
//! ```toml
//! [[plugin]]
//! crate = "rig-harness-codemode"
//! path = "/path/to/rig/crates/rig-harness-codemode"
//! plugin = "rig_harness_codemode::CodeModePlugin"
//! ```
//!
//! Monty 1.1 needs a newer toolchain than the rig workspace's 1.96, so this
//! crate is a workspace of its own and the agent must be built with one
//! Monty builds on.

mod interpreter;

use std::future::poll_fn;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::task::Poll;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use rig_core::completion::{AssistantContent, Message};
use rig_core::message::{ToolCall, ToolResult, ToolResultContent};
use rig_harness::core::agent::{AgentId, ToolCallRun, TurnOutcome};
use rig_harness::core::calls::{Running, Wake};
use rig_harness::core::harness::{AgentSpec, Harness};
use rig_harness::core::inbox::{Origin, RequestId};
use rig_harness::core::tools::{
    AppToolsExt, Footprint, OpenCall, ToolCalled, ToolOptions, ToolOutput,
};
use serde::Deserialize;
use serde_json::{Map, Value};

use interpreter::{Answer, Ask, FromScript, Host, Ran};

/// The tool that runs a script.
pub const RUN_CODE: &str = "run_code";

/// The native stack of a script's thread; Monty bounds its own recursion
/// well below it.
const STACK_BYTES: usize = 16 * 1024 * 1024;

const DESCRIPTION: &str = "Run a Python script that orchestrates agents and tools, and return \
    the value of its last expression (a str as it is, anything else as JSON) with what it \
    printed. Use it when the work needs several agents or many tool calls tied together by \
    logic: a relay between agents, a fan-out with all replies gathered, a loop until a check \
    passes.\n\n\
    The script is Monty Python: a sandboxed subset of Python 3 with async/await, \
    asyncio.gather, functions, simple classes, comprehensions and the modules json, re, math, \
    itertools, collections and datetime among others; no third-party packages, no file \
    system, network or environment. Top-level `await` works. These async host functions are \
    defined; await each call:\n\n\
    async def spawn_agent(name: str, model: str | None = None, system_prompt: str | None = \
    None, tools: list[str] | None = None) -> str\n    \
    Starts an idle agent spawned by you and returns its id. Unset settings are yours.\n\
    async def send(agent: str, text: str) -> str\n    \
    Sends text to an agent this script spawned, as a request, and returns the request's id. \
    The agent starts on it, or queues it when busy, and keeps its conversation between \
    requests.\n\
    async def reply(request: str) -> str\n    \
    Waits for the agent's answer to the request and returns its text; raises RuntimeError \
    when its turn failed or was stopped.\n\
    async def call_tool(name: str, args: dict | None = None) -> str\n    \
    Calls one of your own tools with args and returns its text output; raises RuntimeError \
    when the tool fails.\n\n\
    Example, fan out and pick one:\n\
    import asyncio\n\
    ids = await asyncio.gather(*[spawn_agent(f\"poet{i}\") for i in range(3)])\n\
    asks = [await send(a, \"Write a haiku about rust.\") for a in ids]\n\
    poems = await asyncio.gather(*[reply(r) for r in asks])\n\
    judge = await spawn_agent(\"judge\")\n\
    await reply(await send(judge, \"Pick the best haiku and quote it:\\n\\n\" + \"\\n\\n\".join(poems)))";

const RULES: &[&str] = &[
    "Use `run_code` when the work needs several agents or many tool calls tied together by \
     logic; for one subagent or one tool call, call it directly. A `run_code` script waits \
     for its agents' replies, so its result holds their real answers.",
    "Never invent, simulate or paraphrase as fact another agent's reply. If the requested \
     interaction is not supported, say so before offering an alternative.",
];

/// Adds the `run_code` tool.
#[derive(Default)]
pub struct CodeModePlugin;

impl Plugin for CodeModePlugin {
    fn build(&self, app: &mut App) {
        app.add_open_tool(
            RUN_CODE,
            DESCRIPTION,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Monty Python script. Its last expression is the result."
                    }
                },
                "required": ["code"]
            }),
            ToolOptions {
                rules: RULES,
                footprint: Footprint::Exclusive,
                ..ToolOptions::default()
            },
            on_run_code,
        );
    }
}

/// The arguments of a `run_code` call.
#[derive(Deserialize)]
struct RunCodeArgs {
    code: String,
}

/// Starts a `run_code` call's script as the call's [`Running`] task; its
/// result becomes the call's [`ToolOutput`], and Esc, which despawns the
/// call, cancels it.
fn on_run_code(
    called: On<ToolCalled>,
    calls: Query<(&ToolCallRun, Option<&OpenCall>)>,
    agents: Query<&AgentId>,
    harness: Option<Res<Harness>>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok((run, open)) = calls.get(called.call) else {
        return;
    };
    let call = run.call.clone();
    let refuse = |why: String| ToolOutput(call.error_result(vec![ToolResultContent::text(why)]));
    let args =
        serde_json::from_value::<RunCodeArgs>(Value::Object(call.function.arguments.clone()));
    let (me, harness, code) = match (agents.get(called.agent), harness, args) {
        (Err(_), ..) => {
            commands
                .entity(called.call)
                .insert_if_new(refuse("The calling agent is gone.".to_owned()));
            return;
        }
        (_, None, _) => {
            commands
                .entity(called.call)
                .insert_if_new(refuse("The app has no Harness handle.".to_owned()));
            return;
        }
        (_, _, Err(error)) => {
            commands.entity(called.call).insert_if_new(refuse(format!(
                "The arguments do not fit: {error}. Nothing ran."
            )));
            return;
        }
        (Ok(me), Some(harness), Ok(args)) => (me.clone(), harness, args.code),
    };
    let harness = match open {
        Some(open) => harness.within(open.0.id()),
        None => (*harness).clone(),
    };
    let script = Script {
        harness,
        me,
        call,
        sent: AtomicU32::new(0),
        spawned: Mutex::new(Vec::new()),
    };
    commands.entity(called.call).insert(Running::spawn(
        AsyncComputeTaskPool::get_or_init(TaskPool::default),
        &wake,
        script.run(code),
    ));
}

/// One `run_code` call's script, seen from the async side: it answers the
/// script's host function calls through the [`Harness`].
struct Script {
    harness: Harness,
    /// The calling agent.
    me: AgentId,
    /// The `run_code` call.
    call: ToolCall,
    /// How many requests the script sent, which numbers their ids.
    sent: AtomicU32,
    /// The agents the script spawned, which `send` reaches.
    spawned: Mutex<Vec<AgentId>>,
}

/// What happened next while a script runs.
enum Event {
    /// The script asked something, ended, or its thread went away.
    Script(Option<FromScript>),
    /// A host function call was answered.
    Answered(Answer),
}

impl Script {
    /// Runs `code` on a thread of its own and answers its host function
    /// calls, several at once, until it ends. Dropping this future, as
    /// cancelling the call does, closes the script's channels, and its
    /// thread stops at its next pause.
    async fn run(self, code: String) -> ToolResult {
        let (asks, mut asked) = futures::channel::mpsc::unbounded();
        let (answer, answers) = mpsc::channel::<Answer>();
        let started = std::thread::Builder::new()
            .name(RUN_CODE.to_owned())
            .stack_size(STACK_BYTES)
            .spawn(move || interpreter::run(code, asks, answers));
        if let Err(error) = started {
            return self.failed(format!("The interpreter did not start: {error}"));
        }
        let mut serving = FuturesUnordered::new();
        loop {
            let event = poll_fn(|cx| {
                if let Poll::Ready(message) = asked.poll_next_unpin(cx) {
                    return Poll::Ready(Event::Script(message));
                }
                if let Poll::Ready(Some(answered)) = serving.poll_next_unpin(cx) {
                    return Poll::Ready(Event::Answered(answered));
                }
                Poll::Pending
            })
            .await;
            match event {
                Event::Script(Some(FromScript::Ask(ask))) => serving.push(self.serve(ask)),
                Event::Script(Some(FromScript::Done(ended))) => return self.ended(ended),
                Event::Script(None) => {
                    return self.failed("The interpreter stopped without a result.".to_owned());
                }
                Event::Answered(answered) => {
                    answer.send(answered).ok();
                }
            }
        }
    }

    /// Runs one host function call.
    async fn serve(&self, ask: Ask) -> Answer {
        let answer = match ask.host {
            Host::SpawnAgent(spec) => self.spawn(spec).await,
            Host::Send { agent, text } => self.send(agent, text).await,
            Host::Reply { request } => self.reply(request).await,
            Host::CallTool { name, args } => self.call_tool(name, args).await,
        };
        (ask.id, answer)
    }

    async fn spawn(&self, spec: AgentSpec) -> Result<Value, String> {
        let id = self
            .harness
            .spawn_agent(spec, Some(self.me.clone()))
            .await
            .map_err(|error| error.to_string())?;
        if let Ok(mut spawned) = self.spawned.lock() {
            spawned.push(id.clone());
        }
        Ok(Value::String(id.0))
    }

    async fn send(&self, agent: String, text: String) -> Result<Value, String> {
        let to = AgentId(agent);
        let mine = self
            .spawned
            .lock()
            .map(|spawned| spawned.contains(&to))
            .unwrap_or(false);
        if !mine {
            return Err(format!(
                "`{}` is not an agent this script spawned; send reaches only those",
                to.0
            ));
        }
        let number = self.sent.fetch_add(1, Ordering::Relaxed);
        let request = RequestId(format!("{}.{number}", self.call.id));
        let origin = Origin::agent(self.me.clone(), Some(request));
        self.harness
            .send(to, text, origin)
            .await
            .map(|request| Value::String(request.0))
            .map_err(|error| error.to_string())
    }

    async fn reply(&self, request: String) -> Result<Value, String> {
        match self.harness.reply(RequestId(request)).await {
            Ok(TurnOutcome::Answered(message)) => Ok(Value::String(text_of(&message))),
            Ok(TurnOutcome::Failed(why)) => Err(format!("the agent's turn failed: {why}")),
            Ok(TurnOutcome::Stopped) => Err("the agent was stopped before it answered".to_owned()),
            Err(error) => Err(error.to_string()),
        }
    }

    async fn call_tool(&self, name: String, args: Map<String, Value>) -> Result<Value, String> {
        if name == RUN_CODE {
            return Err("a script cannot call run_code".to_owned());
        }
        let result = self
            .harness
            .call_tool(self.me.clone(), &name, Value::Object(args))
            .await
            .map_err(|error| error.to_string())?;
        let text = result
            .content
            .iter()
            .map(|item| match item.as_json() {
                Some(value) => value.to_string(),
                None => item.as_text().unwrap_or("[image]").to_owned(),
            })
            .collect::<Vec<_>>()
            .join("\n");
        if result.is_error {
            Err(text)
        } else {
            Ok(Value::String(text))
        }
    }

    /// The call's result for a script that ended.
    fn ended(&self, ended: Result<Ran, Ran>) -> ToolResult {
        let (ran, failed) = match ended {
            Ok(ran) => (ran, false),
            Err(ran) => (ran, true),
        };
        let mut text = String::new();
        if !ran.printed.is_empty() {
            text.push_str("Printed:\n");
            text.push_str(&ran.printed);
            text.push_str("\n\n");
        }
        if failed {
            text.push_str("The script failed:\n");
            text.push_str(&ran.text);
            return self.call.error_result(vec![ToolResultContent::text(text)]);
        }
        if ran.text.is_empty() {
            text.push_str("The script ended without a result.");
        } else {
            text.push_str(&ran.text);
        }
        self.call.result(vec![ToolResultContent::text(text)])
    }

    fn failed(&self, why: String) -> ToolResult {
        self.call.error_result(vec![ToolResultContent::text(why)])
    }
}

/// The text of an agent's answer.
fn text_of(message: &Message) -> String {
    let Message::Assistant(reply) = message else {
        return String::new();
    };
    reply
        .content
        .iter()
        .filter_map(|item| match item {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim()
        .to_owned()
}

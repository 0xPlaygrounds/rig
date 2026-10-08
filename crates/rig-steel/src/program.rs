//! The thread side of a `run_steel` call: the sandboxed Steel engine, the
//! host functions it calls, and the conversion of Scheme values. Each host
//! function blocks this thread, never the app's loop or a task pool, on one
//! [`Harness`] call.

use std::io::{self, Write};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use futures::channel::oneshot;
use futures::future::Shared;
use rig_core::completion::{AssistantContent, Message};
use rig_harness::core::agent::{AgentId, TurnOutcome};
use rig_harness::core::inbox::{Origin, RequestId};
use serde_json::{Map, Value};
use steel::SteelVal;
use steel::parser::lexer::TokenStream;
use steel::parser::tokens::TokenType;
use steel::rerrs::{ErrorKind, SteelErr};
use steel::steel_vm::builtin::BuiltInModule;
use steel::steel_vm::engine::Engine;

use crate::{AgentSpec, Control, Harness, MAX_OUTPUT_BYTES, RUN_STEEL, capped, wait};

/// How many host function calls a program may make.
const MAX_HOST_CALLS: u32 = 1000;

/// Steel's modules that reach the host. Steel's sandboxed engine still
/// registers some of them for `require-builtin`; each is replaced by an
/// empty module.
const HOST_MODULES: [&str; 7] = [
    "steel/process",
    "steel/git",
    "steel/tcp",
    "steel/http",
    "steel/polling",
    "steel/ffi",
    "steel/threads",
];

/// Names a program may not use: module loading, evaluation of built code
/// (through which a program could reach anything above), procedural macros
/// (which can build such names), the environment, native threads, and
/// sleeps and standard streams that no interrupt reaches.
const DENIED: [&str; 22] = [
    "require",
    "require-builtin",
    "provide",
    "load",
    "load-expanded",
    "eval",
    "eval!",
    "eval-string",
    "expand!",
    "run!",
    "env-var",
    "defmacro",
    "datum->syntax",
    "syntax-case",
    "spawn-native-thread",
    "time/sleep-ms",
    "block-on",
    "local-executor/block-on",
    "async-exec",
    "stdin",
    "stdout",
    "load-from-module!",
];

/// Prefixes of the engine's private names, which reach its modules and
/// default ports directly.
const DENIED_PREFIXES: [&str; 3] = ["#", "%", "Engine::"];

/// The global the program's captured output port is bound to; its `#`
/// prefix keeps it out of the program's reach.
const OUTPUT_PORT: &str = "#%rig-steel-output";

/// Why a host function stops when its call is cancelled.
const CANCELLED: &str = "the run_steel call was cancelled";

/// How a program ended.
pub(crate) struct Ended {
    /// What it printed, capped.
    pub printed: String,
    /// The text of its final value, or why it failed.
    pub outcome: Result<String, String>,
}

/// Runs `code` to its end on the calling thread.
pub(crate) fn run(code: String, host: Host) -> Ended {
    let printed = Printed::default();
    let outcome = run_program(code, host, printed.clone());
    Ended {
        printed: printed.text(),
        outcome,
    }
}

fn run_program(code: String, host: Host, printed: Printed) -> Result<String, String> {
    refuse_denied(&code)?;
    let mut engine = Engine::new_sandboxed();
    for module in HOST_MODULES {
        engine.register_module(BuiltInModule::new(module));
    }
    let host = Arc::new(host);
    if !host.control.attach(engine.get_thread_state_controller()) {
        return Err(CANCELLED.to_owned());
    }
    engine.register_value(OUTPUT_PORT, SteelVal::new_dyn_writer_port(printed));
    engine
        .run(format!(
            "(current-output-port {OUTPUT_PORT}) (current-error-port {OUTPUT_PORT}) \
             (current-input-port (open-input-string \"\"))"
        ))
        .map_err(|error| format!("the sandbox did not start: {error}"))?;
    register_host_functions(&mut engine, &host);
    host.control.resume_clock();
    let ran = engine.run(code);
    host.control.pause_clock();
    match ran {
        Ok(values) => Ok(capped(text_of_value(values.last()))),
        Err(error) => Err(engine
            .raise_error_to_string(error.clone())
            .unwrap_or_else(|| error.to_string())),
    }
}

/// Refuses a program that names something outside the sandbox.
fn refuse_denied(code: &str) -> Result<(), String> {
    for token in TokenStream::new(code, true, None).flatten() {
        let name = match token.ty {
            TokenType::Require => "require",
            TokenType::Identifier(_) => token.source,
            _ => continue,
        };
        let denied = DENIED.contains(&name)
            || DENIED_PREFIXES
                .iter()
                .any(|prefix| name.starts_with(prefix));
        if denied {
            return Err(format!(
                "`{name}` is not available: a {RUN_STEEL} program reaches the host only \
                 through spawn-agent, send, reply and call-tool. Nothing ran."
            ));
        }
    }
    Ok(())
}

/// The text a program's final value stands for: a string as it is,
/// nothing as empty, anything else as JSON, or as Scheme writes it.
fn text_of_value(value: Option<&SteelVal>) -> String {
    match value {
        None | Some(SteelVal::Void) => String::new(),
        Some(SteelVal::StringV(text)) => text.to_string(),
        Some(value) => to_json(value)
            .ok()
            .and_then(|json| serde_json::to_string_pretty(&json).ok())
            .unwrap_or_else(|| value.to_string()),
    }
}

/// A Scheme value as JSON: hash maps with string or symbol keys become
/// objects, lists and vectors arrays, symbols and characters strings.
fn to_json(value: &SteelVal) -> Result<Value, String> {
    Ok(match value {
        SteelVal::Void => Value::Null,
        SteelVal::BoolV(value) => Value::Bool(*value),
        SteelVal::IntV(value) => Value::from(*value),
        SteelVal::NumV(value) => serde_json::Number::from_f64(*value)
            .map(Value::Number)
            .ok_or_else(|| format!("{value} has no JSON form"))?,
        SteelVal::StringV(text) | SteelVal::SymbolV(text) => Value::String(text.to_string()),
        SteelVal::CharV(value) => Value::String(value.to_string()),
        SteelVal::ListV(items) => {
            Value::Array(items.iter().map(to_json).collect::<Result<_, _>>()?)
        }
        SteelVal::VectorV(items) => {
            Value::Array(items.iter().map(to_json).collect::<Result<_, _>>()?)
        }
        SteelVal::HashMapV(map) => {
            let mut object = Map::new();
            for (key, item) in map.iter() {
                let key = match key {
                    SteelVal::StringV(key) | SteelVal::SymbolV(key) => key.to_string(),
                    other => return Err(format!("the key {other} is not a string or symbol")),
                };
                object.insert(key, to_json(item)?);
            }
            Value::Object(object)
        }
        other => return Err(format!("{other} has no JSON form")),
    })
}

/// A string argument.
fn string(value: &SteelVal, what: &str) -> Result<String, String> {
    match value {
        SteelVal::StringV(text) => Ok(text.to_string()),
        other => Err(format!("{what} must be a string, not {other}")),
    }
}

/// An optional hash map argument, as a JSON object.
fn options(value: Option<&SteelVal>, what: &str) -> Result<Map<String, Value>, String> {
    match value.map(to_json).transpose()? {
        None => Ok(Map::new()),
        Some(Value::Object(map)) => Ok(map),
        Some(_) => Err(format!("{what} must be a hash map")),
    }
}

/// Binds the host functions in `engine`, each to one [`Host`] method.
fn register_host_functions(engine: &mut Engine, host: &Arc<Host>) {
    let functions: [(&'static str, HostFunction); 4] = [
        ("spawn-agent", |host, args| match args {
            [name] => host.spawn_agent(name, None),
            [name, options] => host.spawn_agent(name, Some(options)),
            _ => Err("expects a name and optionally an options hash".to_owned()),
        }),
        ("send", |host, args| match args {
            [agent, text] => host.send(agent, text),
            _ => Err("expects an agent id and a text".to_owned()),
        }),
        ("reply", |host, args| match args {
            [request] => host.reply(request),
            _ => Err("expects a request id".to_owned()),
        }),
        ("call-tool", |host, args| match args {
            [name] => host.call_tool(name, None),
            [name, args] => host.call_tool(name, Some(args)),
            _ => Err("expects a tool name and optionally an args hash".to_owned()),
        }),
    ];
    for (name, function) in functions {
        let host = host.clone();
        let bound = move |args: &[SteelVal]| {
            host.count_call()
                .and_then(|()| function(&host, args))
                .map_err(|why| SteelErr::new(ErrorKind::Generic, format!("{name}: {why}")))
        };
        engine.register_value(name, SteelVal::anonymous_boxed_function(Arc::new(bound)));
    }
}

/// A host function: checks its Scheme arguments and makes one [`Harness`]
/// call.
type HostFunction = fn(&Host, &[SteelVal]) -> Result<SteelVal, String>;

/// One `run_steel` call's way to the host: the [`Harness`] calls its
/// program makes, as the calling agent.
pub(crate) struct Host {
    harness: Harness,
    /// The calling agent.
    me: AgentId,
    /// The `run_steel` call's id, which prefixes the ids of its requests.
    call: String,
    /// The execution clock and the interrupt switch.
    control: Arc<Control>,
    /// Resolves when the call is cancelled.
    cancelled: Shared<oneshot::Receiver<()>>,
    /// How many host functions the program called.
    calls: AtomicU32,
    /// How many requests it sent, which numbers their ids.
    sent: AtomicU32,
    /// The agents it spawned, which `send` reaches.
    spawned: Mutex<Vec<AgentId>>,
}

impl Host {
    pub(crate) fn new(
        harness: Harness,
        me: AgentId,
        call: String,
        control: Arc<Control>,
        cancelled: Shared<oneshot::Receiver<()>>,
    ) -> Self {
        Self {
            harness,
            me,
            call,
            control,
            cancelled,
            calls: AtomicU32::new(0),
            sent: AtomicU32::new(0),
            spawned: Mutex::new(Vec::new()),
        }
    }

    fn count_call(&self) -> Result<(), String> {
        if self.calls.fetch_add(1, Ordering::Relaxed) >= MAX_HOST_CALLS {
            return Err(format!(
                "the program made more than {MAX_HOST_CALLS} host function calls"
            ));
        }
        Ok(())
    }

    /// Waits for a [`Harness`] call on this thread.
    fn wait<T>(&self, work: impl Future<Output = T>) -> Result<T, String> {
        wait(&self.control, &self.cancelled, work).ok_or_else(|| CANCELLED.to_owned())
    }

    fn spawn_agent(
        &self,
        name: &SteelVal,
        settings: Option<&SteelVal>,
    ) -> Result<SteelVal, String> {
        let mut spec = AgentSpec::named(string(name, "the name")?);
        for (key, value) in options(settings, "the options")? {
            match (key.as_str(), value) {
                ("model", Value::String(model)) => spec.model = Some(model),
                ("system-prompt", Value::String(prompt)) => spec.system_prompt = Some(prompt),
                ("tools", Value::Array(tools)) => {
                    let tools = tools
                        .into_iter()
                        .map(|tool| match tool {
                            Value::String(tool) => Ok(tool),
                            other => Err(format!("a tool name must be a string, not {other}")),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    spec.tools = Some(tools);
                }
                ("model" | "system-prompt", _) => return Err(format!("'{key} must be a string")),
                ("tools", _) => return Err("'tools must be a list of tool names".to_owned()),
                _ => {
                    return Err(format!(
                        "unknown option '{key}; the options are 'model, 'system-prompt and 'tools"
                    ));
                }
            }
        }
        let id = self
            .wait(self.harness.spawn_agent(spec, Some(self.me.clone())))?
            .map_err(|error| error.to_string())?;
        if let Ok(mut spawned) = self.spawned.lock() {
            spawned.push(id.clone());
        }
        Ok(SteelVal::StringV(id.0.into()))
    }

    fn send(&self, agent: &SteelVal, text: &SteelVal) -> Result<SteelVal, String> {
        let to = AgentId(string(agent, "the agent id")?);
        let text = string(text, "the text")?;
        let mine = self
            .spawned
            .lock()
            .map(|spawned| spawned.contains(&to))
            .unwrap_or(false);
        if !mine {
            return Err(format!(
                "`{}` is not an agent this program spawned; send reaches only those",
                to.0
            ));
        }
        let number = self.sent.fetch_add(1, Ordering::Relaxed);
        let request = RequestId(format!("{}.{number}", self.call));
        let origin = Origin::agent(self.me.clone(), Some(request));
        let request = self
            .wait(self.harness.send(to, text, origin))?
            .map_err(|error| error.to_string())?;
        Ok(SteelVal::StringV(request.0.into()))
    }

    fn reply(&self, request: &SteelVal) -> Result<SteelVal, String> {
        let request = RequestId(string(request, "the request id")?);
        match self.wait(self.harness.reply(request))? {
            Ok(TurnOutcome::Answered(message)) => Ok(SteelVal::StringV(text_of(&message).into())),
            Ok(TurnOutcome::Failed(why)) => Err(format!("the agent's turn failed: {why}")),
            Ok(TurnOutcome::Stopped) => Err("the agent was stopped before it answered".to_owned()),
            Err(error) => Err(error.to_string()),
        }
    }

    fn call_tool(&self, name: &SteelVal, args: Option<&SteelVal>) -> Result<SteelVal, String> {
        let name = string(name, "the tool name")?;
        if name == RUN_STEEL {
            return Err(format!("a program cannot call {RUN_STEEL}"));
        }
        let args = options(args, "the args")?;
        let result = self
            .wait(
                self.harness
                    .call_tool(self.me.clone(), &name, Value::Object(args)),
            )?
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
            Ok(SteelVal::StringV(text.into()))
        }
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

/// The program's output port: keeps the first [`MAX_OUTPUT_BYTES`] bytes
/// written to it and drops the rest.
#[derive(Clone, Default)]
struct Printed(Arc<Mutex<PrintedBytes>>);

#[derive(Default)]
struct PrintedBytes {
    bytes: Vec<u8>,
    cut: bool,
}

impl Printed {
    fn text(&self) -> String {
        let Ok(printed) = self.0.lock() else {
            return String::new();
        };
        let mut text = String::from_utf8_lossy(&printed.bytes).into_owned();
        if printed.cut {
            text.push_str(&format!(
                "\n[printed output cut at {MAX_OUTPUT_BYTES} bytes]"
            ));
        }
        text
    }
}

impl Write for Printed {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Ok(mut printed) = self.0.lock() {
            let room = MAX_OUTPUT_BYTES.saturating_sub(printed.bytes.len());
            printed.bytes.extend(buf.iter().take(room));
            if buf.len() > room {
                printed.cut = true;
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

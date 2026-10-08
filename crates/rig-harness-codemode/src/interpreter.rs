//! The interpreter side of a `run_code` call. A script runs in the
//! in-process Monty interpreter on a thread of its own, so its Python never
//! holds up the app's loop or Bevy's task pools. Each host function call
//! pauses the interpreter: the call goes to the async side as an [`Ask`]
//! and the script gets a pending future, which it awaits; once every task
//! of the script waits, the thread blocks until answers come back and
//! resumes with them. A closed channel means the call was cancelled (Esc
//! dropped its task), and the thread stops at the next pause.

use std::sync::mpsc::Receiver;
use std::time::Duration;

use futures::channel::mpsc::UnboundedSender;
use monty::{MontyRun, RunProgress};
use monty_types::{
    CallArgs, CompileOptions, ExcType, ExtFunctionResult, MontyException, MontyObject,
    NameLookupResult, ObjectRef, OsPolicy, PrintWriter, ResourceLimits, ResourceTracker, SleepMode,
};
use rig_harness::core::harness::AgentSpec;
use serde_json::{Map, Value};

/// The host functions a script can call, one per [`Harness`] call.
///
/// [`Harness`]: rig_harness::core::harness::Harness
pub(crate) const FUNCTIONS: [&str; 4] = ["spawn_agent", "send", "reply", "call_tool"];

/// The script's name in tracebacks.
const SCRIPT_NAME: &str = "run_code.py";

/// Execution time a script may take, not counting its waits on host
/// functions.
const MAX_RUNTIME: Duration = Duration::from_secs(60);

/// The largest single allocation a script may make. In process, Monty
/// cannot count the interpreter's live memory apart from the agent's, so
/// this bounds each allocation it can foresee (`'x' * 10**12`), not the
/// total.
const MAX_ALLOCATION_BYTES: usize = 256 * 1024 * 1024;

/// How deep the script's Python calls may nest.
const MAX_RECURSION: usize = 200;

/// How many times the script may pause for the host.
const MAX_SUSPENSIONS: usize = 1000;

/// The most bytes of `print` output kept.
const MAX_PRINTED_BYTES: usize = 64 * 1024;

/// Why a script stops when its call is cancelled.
const CANCELLED: &str = "the run_code call was cancelled";

/// A host function call of the script, to run on the async side.
pub(crate) struct Ask {
    /// Monty's id of the call, which its answer names.
    pub id: u32,
    /// What to do.
    pub host: Host,
}

/// A host function call, with its arguments checked.
pub(crate) enum Host {
    /// `spawn_agent(name, model=None, system_prompt=None, tools=None)`.
    SpawnAgent(AgentSpec),
    /// `send(agent, text)`.
    Send { agent: String, text: String },
    /// `reply(request)`.
    Reply { request: String },
    /// `call_tool(name, args=None)`.
    CallTool {
        name: String,
        args: Map<String, Value>,
    },
}

/// What the interpreter thread sends to the async side.
pub(crate) enum FromScript {
    /// A host function call to run.
    Ask(Ask),
    /// The script ended: its result, or why it failed.
    Done(Result<Ran, Ran>),
}

/// How a script ended.
pub(crate) struct Ran {
    /// The result, or the error with its traceback.
    pub text: String,
    /// What it printed.
    pub printed: String,
}

/// A host function call's answer: a JSON value, or an error the script
/// gets as a `RuntimeError`.
pub(crate) type Answer = (u32, Result<Value, String>);

/// Runs `code` to the end on this thread, sending its host function calls
/// on `asks` and taking their answers from `answers`, then sends how it
/// ended.
pub(crate) fn run(code: String, asks: UnboundedSender<FromScript>, answers: Receiver<Answer>) {
    let mut printed = String::new();
    let ended = interpret(code, &asks, &answers, &mut printed);
    let ended = match ended {
        Ok(text) => Ok(Ran { text, printed }),
        Err(text) => Err(Ran { text, printed }),
    };
    asks.unbounded_send(FromScript::Done(ended)).ok();
}

fn limits() -> ResourceLimits {
    ResourceLimits {
        max_feed_duration: Some(MAX_RUNTIME),
        max_memory: Some(MAX_ALLOCATION_BYTES),
        max_recursion_depth: MAX_RECURSION,
        max_suspensions: MAX_SUSPENSIONS,
        ..ResourceLimits::default()
    }
}

/// Where `print` output goes.
fn out(printed: &mut String) -> PrintWriter<'_> {
    PrintWriter::CollectString(printed, Some(MAX_PRINTED_BYTES))
}

fn interpret(
    code: String,
    asks: &UnboundedSender<FromScript>,
    answers: &Receiver<Answer>,
    printed: &mut String,
) -> Result<String, String> {
    let names = FUNCTIONS.iter().map(|name| (*name).to_owned()).collect();
    let runner = MontyRun::new(code, SCRIPT_NAME, names, CompileOptions::default())
        .map_err(|error| error.to_string())?
        .with_os_policy(OsPolicy {
            // A script has nothing to wait for but its host calls.
            sleep: SleepMode::Zero,
            ..OsPolicy::default()
        });
    let inputs = FUNCTIONS
        .iter()
        .map(|name| MontyObject::function(*name, None))
        .collect();
    let mut progress = runner.start(inputs, ResourceTracker::new(limits()), out(printed));
    let mut suspensions = 0;
    // Answers that came in before the script awaited them.
    let mut early: Vec<Answer> = Vec::new();
    loop {
        let step = progress.map_err(|error| error.to_string())?;
        if !matches!(step, RunProgress::Complete(_)) {
            suspensions += 1;
            if suspensions > MAX_SUSPENSIONS {
                return Err(format!(
                    "the script paused for the host more than {MAX_SUSPENSIONS} times"
                ));
            }
        }
        progress = match step {
            RunProgress::Complete(value) => return Ok(render(&value)),
            RunProgress::FunctionCall(call) => {
                let asked = host(&call.function_name, &call.args, call.object_id.is_some());
                match asked {
                    Err(error) => call.resume(error, out(printed)),
                    Ok(host) => {
                        let ask = Ask {
                            id: call.call_id,
                            host,
                        };
                        if asks.unbounded_send(FromScript::Ask(ask)).is_err() {
                            return Err(CANCELLED.to_owned());
                        }
                        call.resume_pending(out(printed))
                    }
                }
            }
            RunProgress::ResolveFutures(waiting) => {
                let ready = settle(waiting.pending_call_ids(), &mut early, answers)?;
                waiting.resume(ready, out(printed))
            }
            RunProgress::NameLookup(lookup) => {
                lookup.resume(NameLookupResult::Undefined, out(printed))
            }
            RunProgress::OsCall(os) => {
                let name = os.function_call.name();
                let refused = MontyException::new(
                    ExcType::OSError,
                    Some(format!(
                        "{name} is not available: a run_code script has no file system, \
                         network or environment"
                    )),
                );
                os.resume(refused, out(printed))
            }
        };
    }
}

/// The answers to give a script that waits on the calls `pending`: those
/// already in, else the next ones to come.
fn settle(
    pending: &[u32],
    early: &mut Vec<Answer>,
    answers: &Receiver<Answer>,
) -> Result<Vec<(u32, ExtFunctionResult)>, String> {
    if pending.is_empty() {
        return Err("the script waits on nothing".to_owned());
    }
    loop {
        while let Ok(answer) = answers.try_recv() {
            early.push(answer);
        }
        let (ready, later): (Vec<Answer>, Vec<Answer>) =
            early.drain(..).partition(|(id, _)| pending.contains(id));
        *early = later;
        if !ready.is_empty() {
            return Ok(ready
                .into_iter()
                .map(|(id, answer)| (id, returned(answer)))
                .collect());
        }
        match answers.recv() {
            Ok(answer) => early.push(answer),
            Err(_) => return Err(CANCELLED.to_owned()),
        }
    }
}

/// A host answer as the script gets it.
fn returned(answer: Result<Value, String>) -> ExtFunctionResult {
    match answer {
        Ok(value) => ExtFunctionResult::Return(from_json(&value)),
        Err(why) => ExtFunctionResult::Error(MontyException::new(ExcType::RuntimeError, Some(why))),
    }
}

/// The script's result as the tool's text: a string as it is, `None` as
/// nothing, anything else as JSON, or as Python's `repr` when JSON cannot
/// hold it.
fn render(value: &MontyObject) -> String {
    let value = value.as_ref();
    match value.type_name() {
        "str" => value.as_str().unwrap_or_default().to_owned(),
        "NoneType" => String::new(),
        _ => to_json(value)
            .ok()
            .and_then(|json| serde_json::to_string_pretty(&json).ok())
            .unwrap_or_else(|| value.py_repr()),
    }
}

fn type_error(message: String) -> MontyException {
    MontyException::new(ExcType::TypeError, Some(message))
}

/// The host function call `function(args)`, checked, or the exception the
/// script gets instead.
fn host(function: &str, args: &CallArgs, method: bool) -> Result<Host, MontyException> {
    if method {
        return Err(type_error(format!("`{function}` is not a host function")));
    }
    match function {
        "spawn_agent" => {
            let [name, model, system_prompt, tools] = bind(
                function,
                ["name", "model", "system_prompt", "tools"],
                1,
                args,
            )?;
            Ok(Host::SpawnAgent(AgentSpec {
                name: required(function, "name", name)?,
                model: optional(function, "model", model)?,
                system_prompt: optional(function, "system_prompt", system_prompt)?,
                tools: names(function, "tools", tools)?,
            }))
        }
        "send" => {
            let [agent, text] = bind(function, ["agent", "text"], 2, args)?;
            Ok(Host::Send {
                agent: required(function, "agent", agent)?,
                text: required(function, "text", text)?,
            })
        }
        "reply" => {
            let [request] = bind(function, ["request"], 1, args)?;
            Ok(Host::Reply {
                request: required(function, "request", request)?,
            })
        }
        "call_tool" => {
            let [name, tool_args] = bind(function, ["name", "args"], 1, args)?;
            let tool_args = match tool_args {
                None => Map::new(),
                Some(value) if value.type_name() == "NoneType" => Map::new(),
                Some(value) => match to_json(value) {
                    Ok(Value::Object(map)) => map,
                    Ok(_) => {
                        return Err(type_error(format!(
                            "{function}() argument 'args' must be a dict, not {}",
                            value.type_name()
                        )));
                    }
                    Err(why) => {
                        return Err(type_error(format!(
                            "{function}() argument 'args' holds {why}"
                        )));
                    }
                },
            };
            Ok(Host::CallTool {
                name: required(function, "name", name)?,
                args: tool_args,
            })
        }
        other => Err(MontyException::new(
            ExcType::NameError,
            Some(format!("name '{other}' is not defined")),
        )),
    }
}

/// `args` bound to the parameters `params`, positionally then by keyword;
/// the first `required` must be given.
fn bind<'a, const N: usize>(
    function: &str,
    params: [&str; N],
    required: usize,
    args: &'a CallArgs,
) -> Result<[Option<ObjectRef<'a>>; N], MontyException> {
    let mut slots: [Option<ObjectRef<'a>>; N] = [None; N];
    let given = args.args().len();
    if given > N {
        return Err(type_error(format!(
            "{function}() takes at most {N} positional arguments ({given} given)"
        )));
    }
    for (slot, value) in slots.iter_mut().zip(args.args()) {
        *slot = Some(value);
    }
    for (key, value) in args.kwargs() {
        let key = key.as_str().unwrap_or_default();
        let slot = params
            .iter()
            .position(|param| *param == key)
            .and_then(|index| slots.get_mut(index));
        match slot {
            None => {
                return Err(type_error(format!(
                    "{function}() got an unexpected keyword argument '{key}'"
                )));
            }
            Some(Some(_)) => {
                return Err(type_error(format!(
                    "{function}() got multiple values for argument '{key}'"
                )));
            }
            Some(slot) => *slot = Some(value),
        }
    }
    for (param, slot) in params.iter().zip(&slots).take(required) {
        if slot.is_none() {
            return Err(type_error(format!(
                "{function}() missing required argument: '{param}'"
            )));
        }
    }
    Ok(slots)
}

fn required(
    function: &str,
    param: &str,
    value: Option<ObjectRef<'_>>,
) -> Result<String, MontyException> {
    optional(function, param, value)?.ok_or_else(|| {
        type_error(format!(
            "{function}() argument '{param}' must be a str, not None"
        ))
    })
}

fn optional(
    function: &str,
    param: &str,
    value: Option<ObjectRef<'_>>,
) -> Result<Option<String>, MontyException> {
    match value {
        None => Ok(None),
        Some(value) if value.type_name() == "NoneType" => Ok(None),
        Some(value) => value
            .as_str()
            .map(|text| Some(text.to_owned()))
            .ok_or_else(|| {
                type_error(format!(
                    "{function}() argument '{param}' must be a str, not {}",
                    value.type_name()
                ))
            }),
    }
}

fn names(
    function: &str,
    param: &str,
    value: Option<ObjectRef<'_>>,
) -> Result<Option<Vec<String>>, MontyException> {
    let wrong = |value: ObjectRef<'_>| {
        type_error(format!(
            "{function}() argument '{param}' must be a list of str, not {}",
            value.type_name()
        ))
    };
    match value {
        None => Ok(None),
        Some(value) if value.type_name() == "NoneType" => Ok(None),
        Some(value) => {
            let items = value.items().ok_or_else(|| wrong(value))?;
            items
                .into_iter()
                .map(|item| item.as_str().map(str::to_owned).ok_or_else(|| wrong(item)))
                .collect::<Result<Vec<_>, _>>()
                .map(Some)
        }
    }
}

/// A Python value as JSON, or what it holds that JSON cannot.
fn to_json(value: ObjectRef<'_>) -> Result<Value, String> {
    Ok(match value.type_name() {
        "NoneType" => Value::Null,
        "bool" => Value::Bool(value.as_bool().unwrap_or_default()),
        "int" => value
            .as_int()
            .map(Value::from)
            .ok_or("an int too large for JSON")?,
        "float" => value
            .as_float()
            .and_then(serde_json::Number::from_f64)
            .map(Value::Number)
            .ok_or("a float JSON cannot hold")?,
        "str" => Value::String(value.as_str().unwrap_or_default().to_owned()),
        "list" | "tuple" => Value::Array(
            value
                .items()
                .unwrap_or_default()
                .into_iter()
                .map(to_json)
                .collect::<Result<_, _>>()?,
        ),
        "dict" => {
            let mut map = Map::new();
            for (key, item) in value.pairs().unwrap_or_default() {
                let key = key.as_str().ok_or("a dict key that is not a str")?;
                map.insert(key.to_owned(), to_json(item)?);
            }
            Value::Object(map)
        }
        other => return Err(format!("a {other}, which JSON cannot hold")),
    })
}

/// A JSON value as a Python value.
fn from_json(value: &Value) -> MontyObject {
    match value {
        Value::Null => MontyObject::none(),
        Value::Bool(value) => MontyObject::bool(*value),
        Value::Number(number) => number
            .as_i64()
            .map(MontyObject::int)
            .or_else(|| number.as_f64().map(MontyObject::float))
            .unwrap_or_else(MontyObject::none),
        Value::String(text) => MontyObject::string(text.clone()),
        Value::Array(items) => MontyObject::list(items.iter().map(from_json)),
        Value::Object(map) => MontyObject::dict(
            map.iter()
                .map(|(key, item)| (MontyObject::string(key.clone()), from_json(item))),
        ),
    }
}

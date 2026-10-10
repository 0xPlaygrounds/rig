//! Code mode for rig-ecs agents on [Steel](https://github.com/mattwparas/steel),
//! an embeddable Scheme. [`SteelPlugin`] adds the `run_steel` tool: the
//! model writes one Scheme program that spawns agents, sends them requests,
//! waits for their replies, calls the model's own tools, and loops and
//! branches over the results; the value of the program's last expression is
//! the tool's output.
//!
//! The program reaches the host only through four functions, each one call
//! of the [`Harness`] handle: `spawn-agent`, `send`, `reply` and
//! `call-tool`. Steel cannot await a host future from inside the VM, so the
//! program runs on a thread of its own and each host function blocks that
//! thread on its [`Harness`] future with [`bevy_tasks::block_on`]. The
//! agents it messages run concurrently in the app meanwhile, so sending to
//! several agents before waiting on any of them fans out. Neither the app's
//! loop nor a task pool thread ever waits on a program: a pool thread
//! blocked on a child agent would deadlock with that child's own tool calls,
//! which run on the same pool.
//!
//! The sandbox is Steel's own sandboxed engine with every host module
//! (process, git, network, foreign functions) replaced by an empty one, and
//! programs that name a module loader, `eval`, procedural macros, native
//! threads, blocking sleeps or the engine's private `#%` and `%` functions
//! are refused before they run. Printed output is captured and capped, the
//! program's own execution time (its waits on host functions not counted) is
//! limited, host calls are counted, and Esc cancels it like any tool call.
//! Steel has no memory limit.
//!
//! The plugin is written only against rig-ecs's public primitives: an
//! open tool ([`AppToolsExt::add_open_tool`]) whose call is completed by a
//! [`Running`] task, and its own [`Harness`] handle over the core's
//! [`ToolStarter`](rig_ecs::turn::ToolStarter), so the calls a
//! program makes go through the one recorded dispatch and are recorded
//! under its `run_steel` call. It is not in the default `plugins.toml`; add it with
//!
//! ```toml
//! [[plugin]]
//! crate = "rig-steel"
//! path = "/path/to/rig/crates/rig-steel"
//! plugin = "rig_steel::SteelPlugin"
//! ```

mod harness;
mod program;

use std::pin::pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use futures::channel::oneshot;
use futures::future::{self, Either, FutureExt};
use futures_timer::Delay;
use rig_core::message::ToolCall;
use rig_core::tool::{ToolExecutionError, ToolResult};
use rig_ecs::agent::AgentId;
use rig_ecs::calls::{Running, Wake};
use rig_ecs::tools::{AppToolsExt, Footprint, ToolCalled, ToolOptions};
use schemars::JsonSchema;
use serde::Deserialize;
use steel::steel_vm::ThreadStateController;

pub use harness::{AgentSpec, Harness};
use program::{Ended, Host};

/// The tool that runs a program.
pub const RUN_STEEL: &str = "run_steel";

/// The native stack of a program's thread. Steel's VM keeps its own stack
/// on the heap; the compiler and macro expander recurse natively.
const STACK_BYTES: usize = 16 * 1024 * 1024;

/// Execution time a program may take, not counting its waits on host
/// functions.
const MAX_RUNTIME: Duration = Duration::from_secs(60);

/// How often the watchdog compares a program's execution time with
/// [`MAX_RUNTIME`].
const WATCH_EVERY: Duration = Duration::from_millis(250);

/// The most bytes of the result text, and of printed output, kept.
const MAX_OUTPUT_BYTES: usize = 64 * 1024;

const DESCRIPTION: &str = "Run a Steel program (Scheme) that orchestrates agents and tools, and \
    return the value of its last expression (a string as it is, anything else as JSON) with \
    what it printed. Use it when the work needs several agents or many tool calls tied \
    together by logic: a relay between agents, a fan-out with all replies gathered, a loop \
    until a check passes.\n\n\
    The program is Steel, an R7RS-style Scheme: define, let, lambda, if, cond, when, let loop, \
    named lets, map, for-each, filter, foldl, lists, vectors, hash maps (hash, hash-ref), \
    strings (string-append, string-join, string-length), number->string, display and \
    displayln (captured). There is no file system, network, process, module loading or eval. \
    These host functions are defined; each one waits for its answer:\n\n\
    (spawn-agent name [options]) -> string\n    \
    Starts an idle agent spawned by you and returns its id. options is a hash with any of \
    'model (\"vendor/model\"), 'system-prompt (string) and 'tools (list of tool names); unset \
    settings are yours.\n\
    (send agent text) -> string\n    \
    Sends text to an agent this program spawned, as a request, and returns the request's id \
    at once. The agent starts on it, or queues it when busy, and keeps its conversation \
    between requests. Agents work at the same time: send to several, then reply each.\n\
    (reply request) -> string\n    \
    Waits for the agent's answer to the request and returns its text; raises an error when \
    its turn failed or was stopped. Each request is replied once.\n\
    (call-tool name [args]) -> string\n    \
    Calls one of your own tools with args, a hash such as (hash 'path \"src/lib.rs\"), and \
    returns its text output; raises an error when the tool fails.\n\n\
    Two agents write a poem together, relayed four times, then summarised:\n\
    (define a (spawn-agent \"poet-a\"))\n\
    (define b (spawn-agent \"poet-b\"))\n\
    (define (ask agent text) (reply (send agent text)))\n\
    (define poem\n  \
      (let loop ([turn 0] [stanzas (list (ask a \"Write the first stanza of a poem about the sea.\"))])\n    \
        (if (= turn 4)\n        \
            stanzas\n        \
            (loop (+ turn 1)\n              \
                  (append stanzas\n                          \
                          (list (ask (if (even? turn) b a)\n                                     \
                                     (string-append \"Continue this poem with one stanza:\\n\\n\"\n                                                    \
                                                    (string-join stanzas \"\\n\\n\")))))))))\n\
    (hash 'poem poem\n      \
          'summary (ask a (string-append \"Summarise this poem in one sentence:\\n\\n\" (string-join poem \"\\n\\n\"))))\n\n\
    Fan out over three agents at once and keep the shortest answer:\n\
    (define agents (map (lambda (i) (spawn-agent (string-append \"solver-\" (number->string i)))) (range 0 3)))\n\
    (define requests (map (lambda (agent) (send agent \"How does src/lib.rs load plugins? Three sentences.\")) agents))\n\
    (define answers (map reply requests))\n\
    (foldl (lambda (answer best) (if (< (string-length answer) (string-length best)) answer best))\n       \
           (car answers) (cdr answers))";

const RULES: &[&str] = &[
    "Use `run_steel` when the work needs several agents or many tool calls tied together by \
     logic; for one subagent or one tool call, call it directly. A `run_steel` program waits \
     for its agents' replies, so its result holds their real answers.",
    "Never invent, simulate or paraphrase as fact another agent's reply. If the requested \
     interaction is not supported, say so before offering an alternative.",
];

/// Adds the `run_steel` tool.
#[derive(Default)]
pub struct SteelPlugin;

impl Plugin for SteelPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(harness::HarnessPlugin).add_open_tool(
            RUN_STEEL,
            DESCRIPTION,
            ToolOptions {
                rules: RULES,
                footprint: Footprint::Exclusive,
            },
            on_run_steel,
        );
    }
}

/// The arguments of a `run_steel` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SteelArgs {
    /// The Steel (Scheme) program. Its last expression is the result.
    code: String,
}

/// Starts a `run_steel` call's program as the call's [`Running`] task; its
/// result becomes the call's `ToolOutput`, and Esc, which despawns the
/// call, cancels it. [`SteelPlugin`] inserts the [`Harness`] before any turn
/// runs.
fn on_run_steel(
    called: On<ToolCalled<SteelArgs>>,
    harness: Res<Harness>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let program = run_program(
        called.run.call.clone(),
        harness.within(called.effect),
        called.caller.clone(),
        called.args.code.clone(),
    );
    commands.entity(called.call).insert(Running::spawn(
        AsyncComputeTaskPool::get_or_init(TaskPool::default),
        &wake,
        program,
    ));
}

/// Runs `code` on a thread of its own and waits for its end, stopping it
/// when it runs past [`MAX_RUNTIME`]. Dropping this future, as cancelling
/// the call does, interrupts the VM and ends the host function it waits
/// in, so the thread stops soon after.
async fn run_program(call: ToolCall, harness: Harness, me: AgentId, code: String) -> ToolResult {
    let control = Arc::new(Control::default());
    let (cancel, cancelled) = oneshot::channel::<()>();
    let _stopper = Stopper {
        control: control.clone(),
        _cancel: cancel,
    };
    let host = Host::new(
        harness,
        me,
        call.id.to_string(),
        control.clone(),
        cancelled.shared(),
    );
    let (done, mut finished) = oneshot::channel::<Ended>();
    let started = std::thread::Builder::new()
        .name(RUN_STEEL.to_owned())
        .stack_size(STACK_BYTES)
        .spawn(move || {
            done.send(program::run(code, host)).ok();
        });
    if let Err(error) = started {
        return failure(format!("The interpreter did not start: {error}"));
    }
    loop {
        match future::select(&mut finished, pin!(Delay::new(WATCH_EVERY))).await {
            Either::Left((Ok(ended), _)) => return result(ended, control.stopped()),
            Either::Left((Err(_), _)) => {
                return failure("The interpreter stopped without a result.".to_owned());
            }
            Either::Right(_) => {
                if control.spent() > MAX_RUNTIME {
                    control.stop(Stop::TimeLimit);
                }
            }
        }
    }
}

/// Why a program was stopped from outside.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stop {
    /// Its call was cancelled.
    Cancelled,
    /// It ran past [`MAX_RUNTIME`].
    TimeLimit,
}

/// What the async side and a program's thread share: the VM's interrupt
/// switch and the clock of the program's execution time.
#[derive(Default)]
pub(crate) struct Control {
    state: Mutex<ControlState>,
}

#[derive(Default)]
struct ControlState {
    vm: Option<ThreadStateController>,
    stopped: Option<Stop>,
    spent: Duration,
    since: Option<Instant>,
}

impl Control {
    /// Hands over the VM's interrupt switch; false when the program was
    /// already stopped, and should not start.
    pub(crate) fn attach(&self, vm: ThreadStateController) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        if state.stopped.is_some() {
            return false;
        }
        state.vm = Some(vm);
        true
    }

    /// Stops the program: interrupts its VM, which stops at its next
    /// instruction. The first reason given is kept.
    pub(crate) fn stop(&self, why: Stop) {
        if let Ok(mut state) = self.state.lock() {
            state.stopped.get_or_insert(why);
            if let Some(vm) = &state.vm {
                vm.interrupt();
            }
        }
    }

    /// Why the program was stopped, if it was.
    pub(crate) fn stopped(&self) -> Option<Stop> {
        self.state.lock().ok().and_then(|state| state.stopped)
    }

    /// Starts counting execution time.
    pub(crate) fn resume_clock(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.since.get_or_insert_with(Instant::now);
        }
    }

    /// Stops counting execution time, as while waiting on the host.
    pub(crate) fn pause_clock(&self) {
        if let Ok(mut state) = self.state.lock()
            && let Some(since) = state.since.take()
        {
            state.spent += since.elapsed();
        }
    }

    /// The execution time so far.
    fn spent(&self) -> Duration {
        self.state.lock().map_or(Duration::ZERO, |state| {
            state.spent + state.since.map_or(Duration::ZERO, |since| since.elapsed())
        })
    }
}

/// Stops the program when the call's task is dropped or ends, and closes
/// the channel its host functions watch for cancellation.
struct Stopper {
    control: Arc<Control>,
    _cancel: oneshot::Sender<()>,
}

impl Drop for Stopper {
    fn drop(&mut self) {
        self.control.stop(Stop::Cancelled);
    }
}

/// The call's result for a program that ended.
fn result(ended: Ended, stopped: Option<Stop>) -> ToolResult {
    let mut text = String::new();
    if !ended.printed.is_empty() {
        text.push_str("Printed:\n");
        text.push_str(&ended.printed);
        text.push_str("\n\n");
    }
    match (ended.outcome, stopped) {
        (Ok(value), _) if value.is_empty() => text.push_str("The program ended without a result."),
        (Ok(value), _) => text.push_str(&value),
        (Err(_), Some(Stop::TimeLimit)) => {
            text.push_str(&format!(
                "The program was stopped: it ran for more than {} s, not counting its waits on agents and tools.",
                MAX_RUNTIME.as_secs()
            ));
            return failure(capped(text));
        }
        (Err(why), _) => {
            text.push_str("The program failed:\n");
            text.push_str(&why);
            return failure(capped(text));
        }
    }
    ToolResult::success(capped(text).into())
}

/// The result of a call that failed, saying `why`.
fn failure(why: String) -> ToolResult {
    ToolResult::failed(ToolExecutionError::other(why))
}

/// `text` cut to [`MAX_OUTPUT_BYTES`] on a character boundary, with a note
/// when it was cut.
pub(crate) fn capped(mut text: String) -> String {
    if text.len() <= MAX_OUTPUT_BYTES {
        return text;
    }
    text.truncate(text.floor_char_boundary(MAX_OUTPUT_BYTES));
    text.push_str(&format!("\n[output cut at {MAX_OUTPUT_BYTES} bytes]"));
    text
}

/// Waits for `work` on the program's thread, with its clock paused; `None`
/// when the call is cancelled first.
pub(crate) fn wait<T>(
    control: &Control,
    cancelled: &future::Shared<oneshot::Receiver<()>>,
    work: impl Future<Output = T>,
) -> Option<T> {
    let work = pin!(work);
    control.pause_clock();
    let outcome = bevy_tasks::block_on(future::select(work, cancelled.clone()));
    control.resume_clock();
    match outcome {
        Either::Left((value, _)) => Some(value),
        Either::Right(_) => None,
    }
}

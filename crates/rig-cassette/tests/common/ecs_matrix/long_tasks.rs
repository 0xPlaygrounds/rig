//! Long native tasks with independently observed tool state and validation.

#![allow(dead_code, reason = "shared native tasks are registered per provider")]

use rig_core::tool::{Tool, ToolContext, ToolExecutionError};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
};

#[path = "long_tasks/budget.rs"]
mod budget;
#[path = "long_tasks/cache.rs"]
mod cache;
#[path = "long_tasks/state.rs"]
mod state;
use state::{Operation, State, Task};

use super::{
    Wire,
    cells::{CELL, Cell, ToolKind},
    corpus::Program,
};
use rig_cassette::effect_log::EffectLog;
use rig_core::completion::CompletionModel;

const PREAMBLE: &str = "You are a careful task-solving assistant. Use task_operation to inspect source facts, apply structured updates, and validate actual task state. Never invent tool results. Respect the requested sequence so each new operation uses the previous result. A returned validation error is recoverable: retry check. A failed validation is not success: inspect the referenced source and repair the state. Only answer done after check(all) reports passed for the latest revision.";
const BASE: Program = Program {
    preamble: Some(PREAMBLE),
    temperature: Some(0.0),
    max_tokens: Some(2048),
    max_turns: Some(30),
    tool_concurrency: Some(3),
    ..Program::DEFAULT
};
const TASK_TOOLS: &[ToolKind] = &[ToolKind::LongTask];

pub(crate) const REPAIR: Cell = Cell {
    name: "long_task_repair",
    program: Program {
        prompt: "Repair all six calendar source modules. One tool call per turn: read index; check all (retry once if the runner is unavailable). For EACH key in index order: read that source, apply its corrected integer value, check that individual key. Read the contract, including derived dependencies; do not skip modules that seemed unrelated to the first failure. Finally check all and answer done. Set value to null except when applying an integer.",
        ..BASE
    },
    tools: TASK_TOOLS,
    events: true,
    ..CELL
};
pub(crate) const REPAIR_STREAMED: Cell = Cell {
    program: Program {
        streamed: true,
        ..REPAIR.program
    },
    ..REPAIR
};

pub(crate) const RECONCILE: Cell = Cell {
    name: "long_task_reconcile",
    program: Program {
        prompt: "Reconcile six invoice pages. Read index. Read each page separately (six turns), retaining every invoice and credit reference. Then look up the six credit references in THREE turns, TWO reads in parallel per turn. Use the confirmed credit, not its superseded provisional version. For EACH page separately: apply the page's subtotal minus its confirmed credit; then in a separate turn check that page. Finally check all and answer done. Never batch apply or check calls. Set value to null except when applying an integer.",
        ..BASE
    },
    tools: TASK_TOOLS,
    events: true,
    ..CELL
};
pub(crate) const INVENTORY: Cell = Cell {
    name: "long_task_inventory",
    program: Program {
        prompt: "Build the six-depot inventory. Apply under the exact keys depot/1 through depot/6; never append /available or /reserved. Read index. Read each depot in a separate turn. Apply available = stock + inbound - damaged in THREE turns with TWO parallel applies each. Check all and answer done. Do not request a revision until the user asks. Set value to null except when applying an integer.",
        second_prompt: Some(
            "Revise the inventory: reserve five units at each depot. First call revise(all) to apply the new constraint. Update available quantities under the exact keys depot/1 through depot/6; do not create subkeys. Read each depot again in a separate turn, then apply the updated available amounts in THREE turns with TWO parallel applies each. Check all and answer done. Preserve the original stock and inbound facts from our earlier work.",
        ),
        ..BASE
    },
    tools: TASK_TOOLS,
    events: true,
    ..CELL
};

pub(crate) const INVENTORY_WIDE_BATCH: Cell = Cell {
    program: Program {
        prompt: "Build the six-depot inventory. Read index, then read each of depot/1 through depot/6 in six separate turns. Do not apply anything until all six reads are finished. Then submit ALL SIX apply calls together in ONE parallel batch, using the exact depot keys and value = stock + inbound - damaged. Do not append field names to keys. Check all, then answer done. Set value to null for reads/checks. Do not request a revision until the user asks.",
        second_prompt: Some(
            "Reserve five units at each depot. Call revise(all), then read each of depot/1 through depot/6 in six separate turns before applying anything. Submit ALL SIX corrected available quantities together in ONE parallel batch under the exact depot keys. Each new value = stock + inbound - damaged - reserved. Check all, then answer done.",
        ),
        ..BASE
    },
    ..INVENTORY
};

#[derive(Clone)]
pub(crate) struct TaskTool(Arc<Mutex<State>>);
impl Tool for TaskTool {
    const NAME: &'static str = "task_operation";
    type Error = ToolExecutionError;
    type Args = Operation;
    type Output = Value;
    fn description(&self) -> String {
        "Read task source data, apply a structured integer update, validate one key or all, or request the inventory follow-up. Read index first. Check results describe actual state. Apply never implies validation success.".into()
    }
    fn parameters(&self) -> Value {
        let mut schema = json!({"type":"object","properties":{"action":{"type":"string","enum":["read","apply","check","revise"]},"key":{"type":"string"},"value":{"type":["integer","null"]}},"required":["action","key","value"],"additionalProperties":false});
        if self.0.lock().expect("task state").task == Task::Inventory {
            schema["properties"]["key"]["enum"] = json!([
                "index", "all", "depot/1", "depot/2", "depot/3", "depot/4", "depot/5", "depot/6"
            ]);
        }
        schema
    }
    async fn call(&self, _: &mut ToolContext, operation: Operation) -> Result<Value, Self::Error> {
        self.0.lock().expect("task state").execute(operation)
    }
}

struct Slot {
    gate: Arc<tokio::sync::Mutex<()>>,
    tool: Option<TaskTool>,
}
static SLOTS: LazyLock<Mutex<HashMap<&'static str, Slot>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) fn tool(cell: &Cell) -> TaskTool {
    SLOTS
        .lock()
        .expect("task slots")
        .get(cell.name)
        .and_then(|slot| slot.tool.clone())
        .expect("task lease before binding")
}

pub(crate) fn applicable(cell: &Cell) -> bool {
    cell.name.starts_with("long_task_")
}

pub(crate) async fn run_world<M: CompletionModel + Clone + 'static>(
    wire: &Wire<M>,
    cell: &Cell,
    golden: impl FnOnce(&EffectLog),
) -> EffectLog {
    let gate = SLOTS
        .lock()
        .expect("task slots")
        .entry(cell.name)
        .or_insert_with(|| Slot {
            gate: Arc::new(tokio::sync::Mutex::new(())),
            tool: None,
        })
        .gate
        .clone();
    let _lease = gate.lock_owned().await;
    let task = match cell.name {
        "long_task_repair" => Task::Repair,
        "long_task_reconcile" => Task::Reconcile,
        "long_task_inventory" => Task::Inventory,
        _ => panic!("unknown long task"),
    };
    let handle = TaskTool(Arc::new(Mutex::new(State::new(task))));
    SLOTS
        .lock()
        .expect("task slots")
        .get_mut(cell.name)
        .expect("slot")
        .tool = Some(handle.clone());
    let mut cell = *cell;
    cell.provider_retries = Some(0);
    cell.live_resume = cell.resume_after.is_some();
    let log = super::world::run_world(wire, &cell, |log| {
        if let Some(directory) = std::env::var_os("RIG_LONG_TASK_ATTEMPT_DIR") {
            let directory = std::path::PathBuf::from(directory);
            std::fs::create_dir_all(&directory).expect("attempt directory");
            let value =
                crate::cassettes::scrub_artifact(&serde_json::to_value(log).expect("effect log"));
            std::fs::write(
                directory.join(format!("{}.effects.json", cell.name)),
                serde_json::to_vec_pretty(&value).expect("scrubbed effects"),
            )
            .expect("attempt evidence");
        }
        golden(log);
    })
    .await;
    {
        let state = handle.0.lock().expect("task state");
        state.assert_complete();
        assert_invocations(&state, &log);
    }
    assert_history_growth(&log, task);
    eprintln!(
        "LONG_TASK_USAGE {} {:?} {}",
        cell.name,
        wire.thinking,
        cache::assert_usage(wire.thinking, &log)
    );
    let turns = log
        .records
        .iter()
        .filter(|r| r.kind.family() == rig_core::effect::EffectFamily::Completion)
        .count();
    assert!(
        (15..=30).contains(&turns),
        "long tasks require at least 15 turns and respect the 30-turn hard cap; got {turns}"
    );
    log
}

pub(crate) fn assert_intermediate(cell: &Cell) {
    tool(cell)
        .0
        .lock()
        .expect("task state")
        .assert_initial_inventory();
}

pub(crate) fn install_budget(app: &mut bevy_app::App, cell: &Cell) {
    if applicable(cell) {
        budget::install(app, cell);
    }
}

pub(crate) fn remaining_turns(used: usize) -> usize {
    30_usize
        .checked_sub(used)
        .filter(|remaining| *remaining > 0)
        .expect("whole-task completion budget exhausted")
}

pub(crate) fn assert_world(
    world: &mut bevy_ecs::world::World,
    runs: &[bevy_ecs::entity::Entity],
    log: &EffectLog,
) {
    use rig_core::{effect::Outcome, message::Message};
    budget::assert_dispatched(
        world,
        log.records
            .iter()
            .filter(|record| matches!(record.kind, rig_core::effect::EffectKind::ToolCall { .. }))
            .count(),
    );
    for run in runs {
        let scope = world
            .get::<rig_ecs::bus::Scope>(*run)
            .expect("run scope")
            .0
            .clone();
        let usages: Vec<_> = log
            .records
            .iter()
            .filter(|record| record.scope.as_deref() == Some(scope.as_str()))
            .filter_map(|record| match &record.outcome {
                Ok(Outcome::Completion(response)) => Some(response.usage),
                _ => None,
            })
            .collect();
        cache::assert_totals(
            &usages,
            world
                .get::<rig_ecs::agent::Usage>(*run)
                .expect("native run usage")
                .0,
        );
        let history = super::reasoning::assistant_history(world, *run);
        let last = log
            .records
            .iter()
            .rev()
            .find(|record| {
                record.scope.as_deref() == Some(scope.as_str())
                    && matches!(record.kind, rig_core::effect::EffectKind::Completion { .. })
            })
            .expect("last request in scope");
        let request: Vec<_> = super::long_loop::request_history(last)
            .iter()
            .filter(|message| !matches!(message, Message::System { .. }))
            .collect();
        assert_eq!(
            history.len(),
            request.len() + 1,
            "committed final answer extends the last request once"
        );
        assert_eq!(
            serde_json::to_value(history.get(..request.len()).expect("history prefix"))
                .expect("history"),
            serde_json::to_value(request).expect("request"),
            "actual committed history matches dispatched context"
        );
    }
    let observed: Vec<_> = log
        .records
        .iter()
        .filter_map(|record| match &record.outcome {
            Ok(Outcome::Completion(response)) => Some(response.usage),
            _ => None,
        })
        .collect();
    let run_totals: Vec<_> = runs
        .iter()
        .map(|run| {
            world
                .get::<rig_ecs::agent::Usage>(*run)
                .expect("native run usage")
                .0
        })
        .collect();
    cache::assert_totals(&observed, cache::reported_totals(&run_totals));
}

pub(crate) fn assert_requests(provider: &str, scenario: &str) {
    crate::cache_conformance::assert_prefix_stable(provider, scenario);
    let interactions = crate::cassettes::recorded_interaction_bodies(provider, scenario);
    assert!(interactions.len() >= 15, "long task wire turns");
    if provider == "gemini" {
        for path in crate::cassettes::recorded_request_paths(provider, scenario) {
            assert!(
                path.contains("/models/gemini-3.8-flash:"),
                "task model endpoint"
            );
        }
    }
    let mut cache_key = None;
    for (request, _) in interactions {
        let body: Value = serde_json::from_str(&request).expect("task request JSON");
        if provider == "openai" {
            let key = body
                .get("prompt_cache_key")
                .and_then(Value::as_str)
                .expect("wire cache affinity key");
            assert!(!key.is_empty());
            if let Some(previous) = &cache_key {
                assert_eq!(
                    previous, key,
                    "scrubbed cache-key identity stays stable through turns and restore"
                );
            }
            cache_key = Some(key.to_owned());
        }
        if provider == "gemini" {
            assert_eq!(
                body.pointer("/generationConfig/thinkingConfig/thinkingLevel"),
                Some(&json!("low")),
                "model-supported thinking level survives every request"
            );
            assert!(
                body.pointer("/generationConfig/thinkingConfig/thinkingBudget")
                    .is_none()
            );
        }
        if provider == "anthropic" {
            assert_anthropic_request(&body, scenario);
        }
    }
}

fn assert_anthropic_request(body: &Value, scenario: &str) {
    fn markers(value: &Value) -> usize {
        match value {
            Value::Object(fields) => {
                usize::from(fields.contains_key("cache_control"))
                    + fields.values().map(markers).sum::<usize>()
            }
            Value::Array(values) => values.iter().map(markers).sum(),
            _ => 0,
        }
    }
    assert!(
        (1..=4).contains(&markers(body)),
        "cache markers on every turn"
    );
    if scenario.ends_with("repair_streamed") {
        assert_eq!(
            body.pointer("/cache_control/ttl"),
            Some(&json!("1h")),
            "automatic one-hour cache"
        );
    } else if scenario.ends_with("inventory") {
        assert_eq!(
            body.pointer("/cache_control/type"),
            Some(&json!("ephemeral")),
            "automatic tail cache"
        );
        assert!(
            body.pointer("/cache_control/ttl").is_none(),
            "tail uses default five-minute TTL"
        );
        for field in ["system", "tools"] {
            let last = body
                .get(field)
                .and_then(Value::as_array)
                .and_then(|blocks| blocks.last())
                .expect("static prefix block");
            assert_eq!(
                last.pointer("/cache_control/ttl"),
                Some(&json!("1h")),
                "static prefix retains its own TTL"
            );
        }
    } else {
        assert_eq!(markers(body), 3, "manual system, tool and tail breakpoints");
        for field in ["system", "tools"] {
            let block = body
                .get(field)
                .and_then(Value::as_array)
                .and_then(|blocks| blocks.last())
                .expect("manual static prefix");
            assert_eq!(
                block.pointer("/cache_control/type"),
                Some(&json!("ephemeral"))
            );
        }
        let tail = body
            .get("messages")
            .and_then(Value::as_array)
            .and_then(|messages| messages.last())
            .and_then(|message| message.get("content"))
            .and_then(Value::as_array)
            .and_then(|blocks| blocks.last())
            .expect("manual history tail");
        assert_eq!(
            tail.pointer("/cache_control/type"),
            Some(&json!("ephemeral"))
        );
    }
}

fn assert_invocations(state: &State, log: &EffectLog) {
    use super::long_loop::{dispatched_call, dispatched_result};
    let mut actual: Vec<_> = state
        .invocations
        .iter()
        .map(|invocation| {
            serde_json::to_string(&(&invocation.operation, &invocation.result)).expect("invocation")
        })
        .collect();
    let mut recorded: Vec<_> = log
        .records
        .iter()
        .filter(|record| matches!(record.kind, rig_core::effect::EffectKind::ToolCall { .. }))
        .map(|record| {
            let (name, args) = dispatched_call(record);
            assert_eq!(name, TaskTool::NAME);
            let operation: Operation = serde_json::from_value(args).expect("task arguments");
            let result = dispatched_result(record);
            let output: Result<Value, String> = if result.is_success() {
                Ok(result
                    .output()
                    .as_json()
                    .expect("structured task output")
                    .clone())
            } else {
                Err(result
                    .error()
                    .expect("ordinary execution error")
                    .to_string())
            };
            serde_json::to_string(&(operation, output)).expect("recorded invocation")
        })
        .collect();
    actual.sort();
    recorded.sort();
    assert_eq!(
        actual, recorded,
        "every actual invocation and result is recorded exactly once"
    );
}

fn assert_history_growth(log: &EffectLog, task: Task) {
    use super::long_loop::{
        dispatched_call, dispatched_result, request_history, requested_call_ids, requested_calls,
        turns,
    };
    use rig_core::message::{Message, UserContent};
    let turns = turns(log);
    for turn in &turns {
        let calls = requested_calls(turn.completion);
        let operations: Vec<Operation> = calls
            .iter()
            .map(|(_, arguments)| {
                serde_json::from_value(arguments.clone()).expect("task operation")
            })
            .collect();
        for operation in &operations {
            if operation.action == "check"
                || operation.action == "revise"
                || (task == Task::Repair && operation.action == "apply")
            {
                assert_eq!(
                    operations.len(),
                    1,
                    "state-observing mutations require an isolated turn"
                );
            }
        }
        if task != Task::Inventory {
            assert!(
                operations
                    .iter()
                    .filter(|operation| operation.action == "apply")
                    .count()
                    <= 1,
                "revision-bearing writes must not run concurrently"
            );
        }
        assert_eq!(
            requested_calls(turn.completion),
            turn.tools
                .iter()
                .map(|r| dispatched_call(r))
                .collect::<Vec<_>>(),
            "dispatch preserves call order and arguments"
        );
    }
    for pair in turns.windows(2) {
        let [previous, next] = pair else {
            unreachable!("two turns");
        };
        let before = request_history(previous.completion);
        let after = request_history(next.completion);
        assert_eq!(
            after.len(),
            before.len() + 2,
            "one committed answer/result pair or answer/follow-up pair"
        );
        assert_eq!(
            serde_json::to_value(after.get(..before.len()).expect("prior prefix"))
                .expect("history"),
            serde_json::to_value(before).expect("history"),
            "history never rewrites earlier task results"
        );
        if previous.tools.is_empty() {
            continue;
        }
        let Some(Message::User { content }) = after.last() else {
            panic!("tool result message");
        };
        let ids = requested_call_ids(previous.completion);
        assert_eq!(content.len(), previous.tools.len());
        for ((part, record), id) in content.iter().zip(&previous.tools).zip(ids) {
            let UserContent::ToolResult(result) = part else {
                panic!("only tool results");
            };
            assert_eq!(&result.call, id, "call/result association");
            assert_eq!(
                rig_core::tool::ToolOutput::content(result.content.clone())
                    .expect("content")
                    .render(),
                dispatched_result(record).output().render(),
                "exact delivered tool payload"
            );
        }
    }
}

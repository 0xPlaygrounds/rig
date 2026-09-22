use std::collections::BTreeMap;

use rig_core::tool::ToolExecutionError;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Task {
    Repair,
    Reconcile,
    Inventory,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Operation {
    pub action: String,
    pub key: String,
    pub value: Option<i64>,
}

#[derive(Clone, Debug)]
pub(crate) struct Invocation {
    pub operation: Operation,
    pub result: Result<Value, String>,
}

pub(crate) struct State {
    pub task: Task,
    pub values: BTreeMap<String, i64>,
    pub invocations: Vec<Invocation>,
    pub revision: usize,
    pub verified_revision: Option<usize>,
    runner_available: bool,
    revised: bool,
}

const MODULES: [(&str, i64, &str); 6] = [
    ("days_per_week", 7, "A week contains seven days."),
    ("hours_per_day", 24, "A day contains twenty-four hours."),
    ("minutes_per_hour", 60, "An hour contains sixty minutes."),
    ("seconds_per_minute", 60, "A minute contains sixty seconds."),
    (
        "seconds_per_day",
        86400,
        "Multiply hours_per_day, minutes_per_hour and seconds_per_minute.",
    ),
    (
        "seconds_per_week",
        604800,
        "Multiply days_per_week and seconds_per_day. This derived constant must also be repaired.",
    ),
];

impl State {
    pub(crate) fn new(task: Task) -> Self {
        Self {
            task,
            values: BTreeMap::new(),
            invocations: Vec::new(),
            revision: 0,
            verified_revision: None,
            runner_available: task != Task::Repair,
            revised: false,
        }
    }

    pub(crate) fn execute(&mut self, operation: Operation) -> Result<Value, ToolExecutionError> {
        let result = if self.invocations.len() >= 60 {
            Err(ToolExecutionError::other("task operation budget exhausted"))
        } else {
            self.evaluate(&operation)
        };
        self.invocations.push(Invocation {
            operation,
            result: result.as_ref().cloned().map_err(ToString::to_string),
        });
        result
    }

    fn evaluate(&mut self, op: &Operation) -> Result<Value, ToolExecutionError> {
        match op.action.as_str() {
            "read" => self.read(&op.key),
            "apply" => {
                if !self.expected().contains_key(&op.key) {
                    return Err(ToolExecutionError::not_found("unknown task key"));
                }
                let value = op
                    .value
                    .ok_or_else(|| ToolExecutionError::other("apply requires value"))?;
                self.values.insert(op.key.clone(), value);
                self.revision += 1;
                if self.task == Task::Inventory {
                    // Parallel writes publish key-local results, not scheduling-dependent revisions.
                    Ok(json!({"saved": op.key, "value": value}))
                } else {
                    Ok(json!({"saved": op.key, "value": value, "revision": self.revision}))
                }
            }
            "check" => {
                if !self.runner_available {
                    self.runner_available = true;
                    return Err(ToolExecutionError::other(
                        "validation runner temporarily unavailable; retry check",
                    ));
                }
                let expected = self.expected();
                if op.key != "all" && !expected.contains_key(&op.key) {
                    return Err(ToolExecutionError::not_found("unknown check key"));
                }
                let failures: Vec<_> = expected.iter()
                    .filter(|(key, value)| (op.key == "all" || **key == op.key) && self.values.get(*key) != Some(*value))
                    .map(|(key, _)| json!({"key": key, "actual": self.values.get(key), "instruction": "read the source and its referenced detail, then repair this entry"}))
                    .collect();
                if failures.is_empty() && op.key == "all" {
                    self.verified_revision = Some(self.revision);
                }
                Ok(
                    json!({"passed": failures.is_empty(), "failures": failures, "revision": self.revision}),
                )
            }
            "revise" if self.task == Task::Inventory => {
                if self.verified_revision != Some(self.revision) || self.values != self.expected() {
                    return Err(ToolExecutionError::other(
                        "verify the complete initial inventory before requesting the follow-up",
                    ));
                }
                if self.revised {
                    return Err(ToolExecutionError::other("follow-up already delivered"));
                }
                self.revised = true;
                self.revision += 1;
                Ok(
                    json!({"follow_up": "Reserve five units at each depot. Read each depot again for the revised available quantity, update all six entries, then verify all. Preserve the original stock and inbound facts in your history."}),
                )
            }
            _ => Err(ToolExecutionError::other("unknown task action")),
        }
    }

    fn read(&self, key: &str) -> Result<Value, ToolExecutionError> {
        if key == "index" {
            let keys: Vec<String> = match self.task {
                Task::Repair => MODULES
                    .iter()
                    .map(|(name, _, _)| (*name).to_owned())
                    .collect(),
                _ => self.expected().into_keys().collect(),
            };
            return Ok(json!({"keys": keys, "task": format!("{:?}", self.task)}));
        }
        match self.task {
            Task::Repair => MODULES.iter().find(|(name, _, _)| *name == key)
                .map(|(name, _, contract)| json!({"path": format!("src/{name}.rs"), "source": format!("pub const {}: i64 = {};", name.to_uppercase(), self.values.get(*name).copied().unwrap_or(0)), "contract": contract, "edit": "apply the corrected integer constant under this key"}))
                .ok_or_else(|| ToolExecutionError::not_found("source module not found")),
            Task::Reconcile => {
                if let Some(n) = index(key, "page/") {
                    let records: Vec<_> = (1..=24).map(|row| json!({"invoice": format!("{n}-{row}"), "quantity": row, "unit_price": n + 2})).collect();
                    let subtotal: i64 = (1..=24).map(|quantity| quantity * (n + 2)).sum();
                    Ok(json!({"page": n, "records": records, "subtotal": subtotal, "adjustment_reference": format!("credit/{n}"), "instruction": "subtract the independently looked-up confirmed credit from subtotal; apply under the page key"}))
                } else if let Some(n) = index(key, "credit/") {
                    Ok(json!({"credit": n * 11, "status": "confirmed", "supersedes": {"credit": n * 3, "status": "provisional"}, "instruction": "use confirmed credit, never add both versions"}))
                } else {
                    Err(ToolExecutionError::not_found("page or credit not found"))
                }
            }
            Task::Inventory => index(key, "depot/").map(|n| json!({"stock": n * 10, "inbound": n * 2, "damaged": n, "reserved": if self.revised {5} else {0}, "instruction": "available = stock + inbound - damaged - reserved"}))
                .ok_or_else(|| ToolExecutionError::not_found("depot not found")),
        }
    }

    pub(crate) fn expected(&self) -> BTreeMap<String, i64> {
        match self.task {
            Task::Repair => MODULES
                .iter()
                .map(|(name, value, _)| ((*name).to_owned(), *value))
                .collect(),
            Task::Reconcile => (1..=6)
                .map(|n| (format!("page/{n}"), 300 * (n + 2) - n * 11))
                .collect(),
            Task::Inventory => (1..=6)
                .map(|n| {
                    (
                        format!("depot/{n}"),
                        n * 11 - if self.revised { 5 } else { 0 },
                    )
                })
                .collect(),
        }
    }

    pub(crate) fn assert_initial_inventory(&self) {
        assert_eq!(self.task, Task::Inventory);
        assert!(!self.revised, "first run must precede the follow-up");
        assert_eq!(self.values, self.expected(), "complete initial inventory");
        assert_eq!(
            self.verified_revision,
            Some(self.revision),
            "validate before the first answer"
        );
    }

    pub(crate) fn assert_complete(&self) {
        assert_eq!(self.values, self.expected(), "actual task state");
        assert_eq!(
            self.verified_revision,
            Some(self.revision),
            "verify after the last edit"
        );
        assert!(
            self.invocations.len() >= 20,
            "a long task executes at least twenty tools"
        );
        assert!(self.invocations.len() <= 60, "bounded task");
        if self.task == Task::Inventory {
            assert!(self.revised, "the follow-up must be completed");
        }
        if self.task == Task::Repair {
            assert_eq!(
                self.invocations
                    .iter()
                    .filter(|call| call.result.as_ref().is_err_and(
                        |error| error.contains("validation runner temporarily unavailable")
                    ))
                    .count(),
                1,
                "one recoverable runner failure"
            );
            assert!(
                self.invocations.iter().any(|call| call
                    .result
                    .as_ref()
                    .is_ok_and(|result| result.get("passed") == Some(&Value::Bool(false)))),
                "validation exposes incomplete work before repair"
            );
        }
    }
}

fn index(key: &str, prefix: &str) -> Option<i64> {
    key.strip_prefix(prefix)?
        .parse::<i64>()
        .ok()
        .filter(|n| (1..=6).contains(n))
}

#[cfg(test)]
#[path = "state/tests.rs"]
mod tests;

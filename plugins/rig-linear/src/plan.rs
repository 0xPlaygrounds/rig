//! The session's plan: the steps settings or the plan file list, each moved
//! along as the agent reads its skill or runs its command, and closed as
//! the run ends.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[cfg(test)]
mod tests;

/// One step of the plan, as `[[linear.plan]]` or the plan file lists it.
#[derive(Deserialize, Clone, Debug, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlanStep {
    /// What the step is, as Linear shows it.
    pub content: String,
    /// The skill whose `SKILL.md` the agent reads as it starts the step.
    pub skill: Option<String>,
    /// Text of the shell command the agent runs as it starts the step.
    pub command: Option<String>,
}

/// Where a step stands, as Linear names it.
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) enum Status {
    Pending,
    InProgress,
    Completed,
    Canceled,
}

/// How the run ended, for the plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum End {
    /// Done: every step is completed.
    Done,
    /// Given up or failed: the current step and those after it are
    /// canceled.
    Failed,
    /// Waiting on a person: the plan stays as it is.
    NeedsInput,
}

/// The plan and where each step stands.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Plan {
    steps: Vec<(PlanStep, Status)>,
    /// The step in progress, once one started.
    current: Option<usize>,
}

impl Plan {
    /// `steps`, every one pending.
    pub(crate) fn new(steps: Vec<PlanStep>) -> Self {
        Self {
            steps: steps
                .into_iter()
                .map(|step| (step, Status::Pending))
                .collect(),
            current: None,
        }
    }

    /// Whether it has no step.
    pub(crate) fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The statuses, in order.
    #[cfg(test)]
    pub(crate) fn statuses(&self) -> Vec<Status> {
        self.steps.iter().map(|(_, status)| *status).collect()
    }

    /// The plan as `agentSessionUpdate` takes it, whole.
    pub(crate) fn json(&self) -> Value {
        Value::Array(
            self.steps
                .iter()
                .map(|(step, status)| json!({ "content": step.content, "status": status }))
                .collect(),
        )
    }

    /// The agent read `path`: starts the first later step whose skill's
    /// `SKILL.md` it is. Whether the plan changed.
    pub(crate) fn read(&mut self, path: &str) -> bool {
        self.start_first(|step| {
            step.skill.as_deref().is_some_and(|skill| {
                let file = format!("{skill}/SKILL.md");
                path == file || path.ends_with(&format!("/{file}"))
            })
        })
    }

    /// The agent ran `command`: starts the first later step whose command
    /// it contains. Whether the plan changed.
    pub(crate) fn ran(&mut self, command: &str) -> bool {
        self.start_first(|step| {
            step.command
                .as_deref()
                .is_some_and(|wanted| !wanted.is_empty() && command.contains(wanted))
        })
    }

    /// Starts the first step after the current one that `triggered` says
    /// started, completing those before it.
    fn start_first(&mut self, triggered: impl Fn(&PlanStep) -> bool) -> bool {
        let after = self.current.map_or(0, |current| current + 1);
        let Some(index) = self
            .steps
            .iter()
            .enumerate()
            .skip(after)
            .find(|(_, (step, _))| triggered(step))
            .map(|(index, _)| index)
        else {
            return false;
        };
        for (at, (_, status)) in self.steps.iter_mut().enumerate() {
            if at < index {
                *status = Status::Completed;
            } else if at == index {
                *status = Status::InProgress;
            }
        }
        self.current = Some(index);
        true
    }

    /// Closes the plan as the run ended. Whether it changed.
    pub(crate) fn end(&mut self, end: End) -> bool {
        let before = self.steps.clone();
        match end {
            End::Done => {
                for (_, status) in &mut self.steps {
                    *status = Status::Completed;
                }
            }
            End::Failed => {
                let from = self.current.unwrap_or(0);
                for (_, status) in self.steps.iter_mut().skip(from) {
                    if *status != Status::Completed {
                        *status = Status::Canceled;
                    }
                }
            }
            End::NeedsInput => {}
        }
        self.steps != before
    }
}

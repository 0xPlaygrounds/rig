//! A hook that watches tool-call argument fragments as they stream. It
//! counts them per call and, when given a limit, stops the run once that
//! many fragments arrived: the run is cancelled before the call ends, so
//! the tool never executes.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError};

use rig::agent::{AgentHook, HookContext, ObservationAction, ToolCallDelta};
use rig::streaming::parse_partial_arguments;

#[derive(Clone, Default)]
pub struct FragmentHook {
    stop_after: Option<usize>,
    /// Fragments seen per call: its part and its tool.
    counts: Arc<Mutex<BTreeMap<(usize, String), usize>>>,
}

impl FragmentHook {
    pub fn new(stop_after: Option<usize>) -> Self {
        Self {
            stop_after,
            counts: Arc::default(),
        }
    }

    /// Fragments seen per call, by part and tool.
    pub fn counts(&self) -> BTreeMap<(usize, String), usize> {
        self.counts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl AgentHook for FragmentHook {
    async fn on_tool_call_delta(
        &self,
        _ctx: &HookContext,
        event: ToolCallDelta<'_>,
    ) -> ObservationAction {
        let total = {
            let mut counts = self.counts.lock().unwrap_or_else(PoisonError::into_inner);
            let count = counts
                .entry((event.part.index(), event.tool_name.to_owned()))
                .or_default();
            *count += 1;
            counts.values().sum::<usize>()
        };
        match self.stop_after {
            Some(limit) if total >= limit => {
                let partial = parse_partial_arguments(event.aggregated);
                let keys: Vec<&str> = partial.keys().map(String::as_str).collect();
                ObservationAction::stop(format!(
                    "stopped after {total} argument fragments; `{}` had stated {keys:?}",
                    event.tool_name
                ))
            }
            _ => ObservationAction::continue_run(),
        }
    }
}

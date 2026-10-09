//! What each agent's model calls used and cost, and how full its context is.
//! Every finished model call adds the usage its provider reported to its
//! agent's [`Spending`], which its log keeps by model, and to its turn's
//! [`TurnSpending`]. rig-core prices a reply its provider did not price at
//! the model's catalog [`Pricing`](rig_core::catalog::Pricing), so the cost
//! here is the provider's figure or the catalog's list price.
//!
//! The context in use is the last call's input and output: the next request
//! sends all of it again. [`Spending::context_use`] measures it against the
//! connected model's window.

use std::ops::{Deref, DerefMut};

use bevy_ecs::prelude::*;
use bevy_log::info;
use bevy_reflect::prelude::*;
use rig_core::catalog::ModelSpec;
use rig_core::completion::UsageTotals;
use serde::{Deserialize, Serialize};

use super::agent::{AgentId, TurnOf};

/// An agent's model calls' usage summed, as rig-core's [`UsageTotals`]
/// sums it; the session logs it by model. A turn's is a [`TurnSpending`].
#[derive(Component, Reflect, Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Debug, Serialize, Deserialize)]
pub struct Spending(pub UsageTotals);

impl Deref for Spending {
    type Target = UsageTotals;

    fn deref(&self) -> &UsageTotals {
        &self.0
    }
}

impl DerefMut for Spending {
    fn deref_mut(&mut self) -> &mut UsageTotals {
        &mut self.0
    }
}

impl Spending {
    /// The cost as `$0.123`, ending in `+` when some calls were not priced,
    /// or `None` when no call was.
    pub fn cost_label(&self) -> Option<String> {
        if self.unpriced >= self.calls {
            return None;
        }
        let more = if self.unpriced > 0 { "+" } else { "" };
        Some(format!("{}{more}", dollars(self.cost)))
    }

    /// The context in use measured against `spec`'s window, when the last
    /// call reported its tokens.
    pub fn context_use(&self, spec: Option<&ModelSpec>) -> Option<ContextUse> {
        self.context.map(|tokens| ContextUse {
            tokens,
            window: spec.and_then(|spec| spec.context_window),
        })
    }

    /// One line for the user or the log: the calls, tokens by kind and the
    /// cost.
    pub fn summary(&self) -> String {
        let calls = match self.calls {
            1 => "1 model call".to_owned(),
            calls => format!("{calls} model calls"),
        };
        let mut parts = vec![format!("{} in", tokens(self.uncached_input()))];
        if let Some(read) = self.tokens.cached_input_tokens.filter(|read| *read > 0) {
            parts.push(format!("{} cache read", tokens(read)));
        }
        if let Some(written) = self
            .tokens
            .cache_creation_input_tokens
            .filter(|written| *written > 0)
        {
            parts.push(format!("{} cache written", tokens(written)));
        }
        let output = self.tokens.output_tokens.unwrap_or(0);
        match self.tokens.reasoning_tokens.filter(|thought| *thought > 0) {
            Some(thought) => parts.push(format!(
                "{} out ({} reasoning)",
                tokens(output),
                tokens(thought)
            )),
            None => parts.push(format!("{} out", tokens(output))),
        }
        let cost = self
            .cost_label()
            .unwrap_or_else(|| "cost unknown".to_owned());
        format!("{calls}: {}; {cost}", parts.join(", "))
    }
}

/// What the running turn's model calls used so far, on the turn entity.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default, Clone, Debug)]
pub struct TurnSpending(pub Spending);

/// The context in use against the model's window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextUse {
    /// Tokens in use.
    pub tokens: u64,
    /// The model's window, when the catalog lists it.
    pub window: Option<u32>,
}

impl ContextUse {
    /// The share of the window in use, in percent, when the window is known.
    pub fn percent(&self) -> Option<u64> {
        self.window
            .filter(|window| *window > 0)
            .map(|window| self.tokens.saturating_mul(100) / u64::from(window))
    }

    /// `45k/200k (22%)`, or `45k` when the window is not known.
    pub fn label(&self) -> String {
        match (self.window, self.percent()) {
            (Some(window), Some(percent)) => format!(
                "{}/{} ({percent}%)",
                tokens(self.tokens),
                tokens(u64::from(window))
            ),
            _ => tokens(self.tokens),
        }
    }
}

/// A token count in at most four characters plus a unit: `999`, `1.2k`,
/// `45k`, `1.2M`.
pub fn tokens(count: u64) -> String {
    // Shown rounded, so the float conversion's precision does not matter.
    let scaled = |unit: f64| count as f64 / unit;
    match count {
        0..1_000 => count.to_string(),
        1_000..10_000 => format!("{:.1}k", scaled(1e3)),
        10_000..1_000_000 => format!("{:.0}k", scaled(1e3)),
        1_000_000..10_000_000 => format!("{:.1}M", scaled(1e6)),
        _ => format!("{:.0}M", scaled(1e6)),
    }
}

/// A cost in USD: tenths of a cent below a dollar, cents above.
pub fn dollars(cost: f64) -> String {
    if cost < 1.0 {
        format!("${cost:.3}")
    } else {
        format!("${cost:.2}")
    }
}

/// Logs what a turn used when its entity goes away, however it ended.
pub(crate) fn log_turn_spending(
    end: On<Remove<TurnSpending>>,
    turns: Query<(&TurnOf, &TurnSpending)>,
    agents: Query<&AgentId>,
) {
    let Ok((&TurnOf(agent), TurnSpending(spent))) = turns.get(end.entity) else {
        return;
    };
    if spent.calls == 0 {
        return;
    }
    let agent = agents.get(agent).map_or("-", |id| id.0.as_str());
    info!(agent, "turn used {}", spent.summary());
}

//! Narrowing an effect log's records ([`EffectFilter`]).

use rig_core::effect::EffectRecord;

/// Which records of an effect log to read: each part set narrows it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EffectFilter {
    /// The agent whose work it was, by its id (the record's scope).
    pub agent: Option<String>,
    /// The effect's kind by name: `completion`, `tool_call`, or a plugin's
    /// own, such as `subagent`.
    pub kind: Option<String>,
    /// The turn of its agent it was made in (the record's turn; the first
    /// is 0). Turns are numbered per agent, so pair it with
    /// [`agent`](Self::agent).
    pub turn: Option<u64>,
}

impl EffectFilter {
    /// The work of the agent with the id `agent`.
    #[must_use]
    pub fn agent(self, agent: impl Into<String>) -> Self {
        Self {
            agent: Some(agent.into()),
            ..self
        }
    }

    /// Effects of the kind named `kind`.
    #[must_use]
    pub fn kind(self, kind: impl Into<String>) -> Self {
        Self {
            kind: Some(kind.into()),
            ..self
        }
    }

    /// Effects made in turn `turn` of their agent.
    #[must_use]
    pub fn turn(self, turn: u64) -> Self {
        Self {
            turn: Some(turn),
            ..self
        }
    }

    /// Whether it keeps `record`.
    pub fn matches(&self, record: &EffectRecord) -> bool {
        let agent = record.scope.as_deref();
        self.agent
            .as_deref()
            .is_none_or(|wanted| agent == Some(wanted))
            && (self.kind.as_deref()).is_none_or(|kind| record.kind.name() == kind)
            && self.turn.is_none_or(|turn| record.turn == Some(turn))
    }
}

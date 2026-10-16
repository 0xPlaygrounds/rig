//! Classic-agent recording identity and effect replay.
//!
//! Install a caller-owned [`EffectLogRecorder`](crate::effect_log::EffectLogRecorder)
//! with [`AgentBuilder::record_to`](rig_agent::AgentBuilder::record_to), then use
//! [`AgentReplayExt::stamp`] on its snapshot or drained log. The agent runtime
//! never owns a concrete log or depends on this crate.
//!
//! ```
//! use rig_cassette::{agent::AgentReplayExt, effect_log::EffectLog};
//!
//! fn stamp(agent: &rig_agent::Agent, log: EffectLog) -> EffectLog {
//!     agent.stamp(log)
//! }
//! ```

use crate::effect_log::EffectLog;
use rig_agent::Agent;

pub mod replay;

/// Recording and replay identity derived from a classic agent's live configuration.
pub trait AgentReplayExt {
    /// The stable hash of the agent's protocol-facing run specification.
    fn run_spec_hash(&self) -> u64;

    /// Stamp this agent's program identity onto a log.
    fn stamp(&self, log: EffectLog) -> EffectLog;

    /// Refuse logs whose identity or required handlers differ from this agent.
    fn check_replayable(&self, log: &EffectLog) -> Result<(), rig_core::error::ErrorReport>;
}

impl AgentReplayExt for Agent {
    /// Return the stable run-spec hash, or zero if serialization fails.
    fn run_spec_hash(&self) -> u64 {
        crate::effect_log::stable_hash(&self.run_spec()).unwrap_or_default()
    }

    /// `log` with this agent's program identity in its header: the run-spec
    /// hash, the hook stack and the required row. A host can stamp a log
    /// from any recorder attached to the agent's handlers.
    fn stamp(&self, mut log: EffectLog) -> EffectLog {
        log.header.run_spec = Some(self.run_spec_hash());
        log.header.hooks = self.program_names();
        log.header.required = self.required_row();
        log
    }
    /// Validate the log header, run-spec hash, hook stack and required
    /// handlers against this agent. Return an error for any mismatch.
    fn check_replayable(&self, log: &EffectLog) -> Result<(), rig_core::error::ErrorReport> {
        crate::effect_log::EffectLogReplayer::check_header(log)?;
        if let Some(recorded) = log.header.run_spec {
            let mine = self.run_spec_hash();
            if recorded != mine {
                return Err(rig_core::error::ErrorReport::new(
                    rig_core::error::ErrorKind::Internal,
                    format!(
                        "replay refused: the log was recorded under run spec {recorded:#018x}, this agent runs under {mine:#018x}"
                    ),
                ));
            }
        }
        // Hooks execute again during replay, so their identities must match.
        let mine = self.program_names();
        if log.header.hooks != mine {
            return Err(rig_core::error::ErrorReport::new(
                rig_core::error::ErrorKind::Internal,
                format!(
                    "replay refused: the log was recorded under the hook stack {:?}, this agent runs under {mine:?}",
                    log.header.hooks
                ),
            ));
        }
        for (key, family) in &log.header.signature {
            match self.handler_descriptor(key) {
                Some(descriptor) if descriptor.family.family() == *family => {}
                Some(descriptor) => {
                    return Err(rig_core::error::ErrorReport::new(
                        rig_core::error::ErrorKind::HandlerUnavailable,
                        format!(
                            "replay refused: `{key}` serves {} on this bus, the log needs {family}",
                            descriptor.family.family()
                        ),
                    ));
                }
                None => {
                    return Err(rig_core::error::ErrorReport::new(
                        rig_core::error::ErrorKind::HandlerUnavailable,
                        format!("replay refused: nothing serves `{key}`, which the log needs"),
                    ));
                }
            }
        }
        if let Err(gap) = self.required_row().is_subset_of(&log.header.handlers) {
            return Err(rig_core::error::ErrorReport::new(
                rig_core::error::ErrorKind::HandlerUnavailable,
                format!(
                    "replay refused: this agent needs `{}` ({}), which the log never served: {gap}",
                    gap.key, gap.needed
                ),
            ));
        }
        let mine = self.required_row();
        let diffs = log.header.required.diff(&mine);
        if !diffs.is_empty() {
            let diffs: Vec<String> = diffs.iter().map(ToString::to_string).collect();
            return Err(rig_core::error::ErrorReport::new(
                rig_core::error::ErrorKind::HandlerUnavailable,
                format!(
                    "replay refused: the log was recorded by a program requiring {:?}, this agent requires {mine:?}: {}",
                    log.header.required,
                    diffs.join("; ")
                ),
            ));
        }
        Ok(())
    }
}

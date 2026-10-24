//! A recorded effect log as answers a host looks up before an effect
//! starts, by who made it, what kind it is, and its place in that scope's
//! sequence.
//!
//! ```
//! use rig_cassette::effect_log::{EffectLog, RecordedAnswers};
//! use rig_core::effect::{EffectId, EffectKind, EffectRecord, HandlerKey, Answered};
//!
//! let call = |args: &str| EffectKind::ToolCall { name: "read".into(), args: args.into() };
//! let record = |id, args: &str, text: &str| {
//!     let mut record = EffectRecord::new(EffectId::from_raw(id), HandlerKey::from("tool:read"),
//!         call(args), Ok(Answered::Custom { payload: text.into() }));
//!     record.scope = Some("agent-1".into());
//!     record
//! };
//! let log = EffectLog::from_records(vec![record(0, "{\"path\":\"a\"}", "A"), record(1, "{\"path\":\"b\"}", "B")]);
//! let mut answers = RecordedAnswers::new(&log);
//! // Out of order: each call takes the first remaining record it matches.
//! let b = answers.answer(Some("agent-1"), &call("{\"path\":\"b\"}"))?;
//! assert!(matches!(b.outcome, Ok(Answered::Custom { payload }) if payload == "B"));
//! assert!(answers.answer(Some("agent-1"), &call("{\"path\":\"c\"}")).is_err());
//! assert_eq!(answers.remaining(), 1);
//! # Ok::<(), rig_core::error::ErrorReport>(())
//! ```

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use rig_core::effect::{EffectKind, EffectRecord};
use rig_core::error::{ErrorKind, ErrorReport};

use super::replay::Incoming;
use super::{EffectLog, RequestCheck};

/// The records of an effect log, as answers to the same effects made again,
/// such as by a host replaying a recorded session: the host answers each
/// effect with its recorded outcome before it starts, and reruns its own
/// review of the answer, as the log keeps the raw outcome
/// ([`EffectRecord::outcome`]).
///
/// Records are kept per recorded scope and effect kind
/// ([`EffectKind::name`]), in recorded order. An effect is answered by the
/// first remaining record of its scope and kind whose request matches it
/// under the [`RequestCheck`], so calls a reply made side by side match
/// whatever order they settled in. The host maps its own scopes (agents
/// whose ids are new in a replay) to the recorded ones.
///
/// Unlike [`EffectLogReplayer`](super::EffectLogReplayer), a handler that
/// serves recorded answers beneath the host's layers, this is data the host
/// asks, so effects of every kind, its own included, are answered the same
/// way, and the host decides what a divergence means (fail, or go live).
#[derive(Clone, Debug, Default)]
pub struct RecordedAnswers {
    queues: BTreeMap<(Option<Arc<str>>, String), VecDeque<EffectRecord>>,
    check: RequestCheck,
}

impl RecordedAnswers {
    /// Every record of `log`, comparing requests by payload.
    pub fn new(log: &EffectLog) -> Self {
        let mut answers = Self::default();
        for record in log {
            answers
                .queues
                .entry((record.scope.clone(), record.kind.name().to_owned()))
                .or_default()
                .push_back(record.clone());
        }
        answers
    }

    /// The same answers, comparing requests as `check` says.
    #[must_use = "the setting applies to the returned value"]
    pub fn checking(mut self, check: RequestCheck) -> Self {
        self.check = check;
        self
    }

    /// The record that answers `kind`, made in the recorded `scope`, taken
    /// out so it answers once.
    ///
    /// # Errors
    ///
    /// A [`Divergence`] when the log has no remaining record of that scope
    /// and kind, or none whose request matches `kind`: the next record in
    /// order, if any, and an [`ErrorKind::Divergence`] report naming the
    /// first difference from it.
    pub fn answer(
        &mut self,
        scope: Option<&str>,
        kind: &EffectKind,
    ) -> Result<EffectRecord, Divergence> {
        let name = kind.name();
        let scope: Option<Arc<str>> = scope.map(Arc::from);
        let shown = scope.as_deref().unwrap_or("no scope");
        let queue = self
            .queues
            .get_mut(&(scope.clone(), name.to_owned()))
            .filter(|queue| !queue.is_empty())
            .ok_or_else(|| Divergence {
                expected: None,
                report: ErrorReport::new(
                    ErrorKind::Divergence,
                    format!(
                        "replay divergence: `{shown}` made a `{name}` effect the log has no more of"
                    ),
                ),
            })?;
        let incoming = Incoming::new(self.check, kind);
        let matched = queue
            .iter()
            .position(|record| incoming.divergence(&record.kind).is_none());
        match matched.and_then(|at| queue.remove(at)) {
            Some(record) => Ok(record),
            None => {
                let expected = queue.front().cloned();
                let what = expected
                    .as_ref()
                    .and_then(|next| incoming.divergence(&next.kind))
                    .unwrap_or_default();
                Err(Divergence {
                    expected,
                    report: ErrorReport::new(
                        ErrorKind::Divergence,
                        format!(
                            "replay divergence: no recorded `{name}` of `{shown}` matches: {what}"
                        ),
                    ),
                })
            }
        }
    }

    /// Records not yet answered.
    pub fn remaining(&self) -> usize {
        self.queues.values().map(VecDeque::len).sum()
    }
}

#[cfg(test)]
mod tests;

/// Why [`RecordedAnswers::answer`] has no record for an effect: the record
/// it was compared with, for a host's own diff, and the report.
#[derive(Clone, Debug)]
pub struct Divergence {
    /// The next record of the effect's scope and kind, in recorded order;
    /// `None` when the log has no more of them.
    pub expected: Option<EffectRecord>,
    /// An [`ErrorKind::Divergence`] report naming the first difference.
    pub report: ErrorReport,
}

impl From<Divergence> for ErrorReport {
    fn from(divergence: Divergence) -> Self {
        divergence.report
    }
}

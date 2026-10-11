//! A recorded effect log as answers a host looks up before an effect
//! starts, by who made it, what kind it is, and its place in that scope's
//! sequence.
//!
//! ```
//! use rig_cassette::effect_log::{EffectLog, RecordedAnswers};
//! use rig_core::effect::{EffectId, EffectKind, EffectRecord, HandlerKey, Outcome};
//!
//! let call = |args: &str| EffectKind::ToolCall { name: "read".into(), args: args.into() };
//! let record = |id, args: &str, text: &str| {
//!     let mut record = EffectRecord::new(EffectId::from_raw(id), HandlerKey::from("tool:read"),
//!         call(args), Ok(Outcome::Custom { payload: text.into() }));
//!     record.scope = Some("agent-1".into());
//!     record
//! };
//! let log = EffectLog::from_records(vec![record(0, "{\"path\":\"a\"}", "A"), record(1, "{\"path\":\"b\"}", "B")]);
//! let mut answers = RecordedAnswers::new(&log);
//! // Out of order: each call takes the first remaining record it matches.
//! let b = answers.answer(Some("agent-1"), &call("{\"path\":\"b\"}"))?;
//! assert!(matches!(b.outcome, Ok(Outcome::Custom { payload }) if payload == "B"));
//! assert!(answers.answer(Some("agent-1"), &call("{\"path\":\"c\"}")).is_err());
//! assert_eq!(answers.remaining(), 1);
//! # Ok::<(), rig_core::error::ErrorReport>(())
//! ```

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use rig_core::effect::{EffectKind, EffectRecord};
use rig_core::error::{ErrorKind, ErrorReport};

use super::replay::divergence_under;
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
/// whose ids are new in a replay) to the recorded ones, which
/// [`Self::scopes`] lists in order of appearance.
///
/// Unlike [`EffectLogReplayer`](super::EffectLogReplayer), a handler that
/// serves recorded answers beneath the host's layers, this is data the host
/// asks, so effects of every kind, its own included, are answered the same
/// way, and the host decides what a divergence means (fail, or go live).
#[derive(Clone, Debug, Default)]
pub struct RecordedAnswers {
    queues: BTreeMap<(Option<Arc<str>>, String), VecDeque<EffectRecord>>,
    scopes: Vec<Option<Arc<str>>>,
    check: RequestCheck,
}

impl RecordedAnswers {
    /// Every record of `log`, comparing requests by payload.
    pub fn new(log: &EffectLog) -> Self {
        let mut answers = Self::default();
        for record in log {
            if !answers.scopes.contains(&record.scope) {
                answers.scopes.push(record.scope.clone());
            }
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

    /// The recorded scopes, in the order of their first record.
    pub fn scopes(&self) -> &[Option<Arc<str>>] {
        &self.scopes
    }

    /// The record that answers `kind`, made in the recorded `scope`, taken
    /// out so it answers once.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Divergence`] when the log has no remaining record of
    /// that scope and kind, or none whose request matches `kind`; the
    /// report names the first difference from the next one in order.
    pub fn answer(
        &mut self,
        scope: Option<&str>,
        kind: &EffectKind,
    ) -> Result<EffectRecord, ErrorReport> {
        let name = kind.name();
        let scope: Option<Arc<str>> = scope.map(Arc::from);
        let shown = scope.as_deref().unwrap_or("no scope");
        let queue = self
            .queues
            .get_mut(&(scope.clone(), name.to_owned()))
            .filter(|queue| !queue.is_empty())
            .ok_or_else(|| {
                ErrorReport::new(
                    ErrorKind::Divergence,
                    format!(
                        "replay divergence: `{shown}` made a `{name}` effect the log has no more of"
                    ),
                )
            })?;
        let check = self.check;
        let matched = queue
            .iter()
            .position(|record| divergence_under(check, &record.kind, kind).is_none());
        match matched.and_then(|at| queue.remove(at)) {
            Some(record) => Ok(record),
            None => {
                let what = queue
                    .front()
                    .and_then(|next| divergence_under(check, &next.kind, kind))
                    .unwrap_or_default();
                Err(ErrorReport::new(
                    ErrorKind::Divergence,
                    format!("replay divergence: no recorded `{name}` of `{shown}` matches: {what}"),
                ))
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

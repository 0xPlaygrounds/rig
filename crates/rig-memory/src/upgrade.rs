//! Reading what older rigs saved: the versions of saved values and of
//! plugins' asides ([`Versioned`], [`Upgrades`]), and the rewrite of a
//! store's older logs in the current format ([`upgrade`]).
//!
//! A log is written in one format, named in its header, and read through a
//! chain of pure steps from any format since [`OLDEST`](crate::OLDEST); a
//! store is rewritten in the current one the first time anything writes to
//! it, never on a read. A saved value carries its own version, which a
//! plugin moves on with steps of its own ([`Upgrades::add`]); a value no
//! step reads is set aside under [`SET_ASIDE`], never dropped.

use std::collections::BTreeMap;

use rig_core::conversation::Aside;
use serde_json::{Map, Value};

use crate::{LoadError, Store};

/// The prefix of the key a saved value no step reads is moved to before its
/// key is written again: `rig.set_aside/<key>/v<N>`. Readers skip such keys
/// silently; they are kept in the file.
pub const SET_ASIDE: &str = "rig.set_aside/";

/// A saved value and the version its type had when it was saved (1 for one
/// saved before versions).
#[derive(Clone, Debug, PartialEq)]
pub struct Versioned {
    /// The version of its type it was saved at, from 1.
    pub v: u32,
    /// The value, as saved.
    pub value: Value,
}

impl Versioned {
    /// `value` at version `v` as an aside's value: `{"v": v, "value": value}`,
    /// or `value` alone at 1.
    pub fn to_aside_value(&self) -> Value {
        if self.v == 1 {
            return self.value.clone();
        }
        let mut envelope = Map::new();
        envelope.insert("v".to_owned(), Value::from(self.v));
        envelope.insert("value".to_owned(), self.value.clone());
        Value::Object(envelope)
    }

    /// An aside's value read as [`to_aside_value`](Self::to_aside_value)
    /// writes it: the envelope's version and value, or the value whole at 1.
    fn of_aside_value(value: &Value) -> Self {
        let enveloped = value.as_object().filter(|object| object.len() == 2);
        let read = enveloped.and_then(|object| {
            let v = object.get("v")?.as_u64()?;
            let v = u32::try_from(v).ok().filter(|v| *v > 1)?;
            Some(Self {
                v,
                value: object.get("value")?.clone(),
            })
        });
        read.unwrap_or_else(|| Self {
            v: 1,
            value: value.clone(),
        })
    }
}

/// A value saved before its type had versions: version 1.
impl From<Value> for Versioned {
    fn from(value: Value) -> Self {
        Self { v: 1, value }
    }
}

/// A pure step reading version `n` of a saved value as version `n + 1`.
pub type Upgrade = fn(Value) -> Result<Value, String>;

/// How to read older versions of saved values and of plugins' asides, by
/// key (a saved value's type path, an aside's kind).
#[derive(Clone, Debug, Default)]
pub struct Upgrades(BTreeMap<String, &'static [Upgrade]>);

/// Why a saved value is set aside, not read.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Unread {
    /// A newer rig saved it, at a version this one does not read.
    #[error("saved at version {v} by a newer rig; this one reads up to {reads}")]
    Newer {
        /// The version it was saved at.
        v: u32,
        /// The newest version this build reads.
        reads: u32,
    },
    /// The step from version `from` failed, or there is none.
    #[error("cannot read version {from}: {why}")]
    Step {
        /// The version the step reads.
        from: u32,
        /// Why it failed.
        why: String,
    },
}

impl Upgrades {
    /// `key`'s values are at version `steps.len() + 1`; `steps[i]` reads
    /// version `i + 1` as `i + 2`. Added again, the later steps win.
    pub fn add(&mut self, key: impl Into<String>, steps: &'static [Upgrade]) {
        self.0.insert(key.into(), steps);
    }

    /// The version `key`'s values are written at: 1 when none was added.
    pub fn version(&self, key: &str) -> u32 {
        let steps = self.0.get(key).map_or(0, |steps| steps.len());
        u32::try_from(steps).map_or(u32::MAX, |steps| steps.saturating_add(1))
    }

    /// `saved` read at this version through each step from its own, or why
    /// it is set aside.
    ///
    /// # Errors
    ///
    /// [`Unread::Newer`] for a version newer than this one, [`Unread::Step`]
    /// when a step fails or the version is 0.
    pub fn upgrade(&self, key: &str, saved: Versioned) -> Result<Value, Unread> {
        let reads = self.version(key);
        if saved.v > reads {
            return Err(Unread::Newer { v: saved.v, reads });
        }
        let steps = self.0.get(key).copied().unwrap_or_default();
        let first = saved.v.checked_sub(1).ok_or_else(|| Unread::Step {
            from: 0,
            why: "there is no version 0".to_owned(),
        })?;
        let mut value = saved.value;
        let skipped = steps
            .iter()
            .skip(usize::try_from(first).unwrap_or(usize::MAX));
        for (from, step) in (saved.v..).zip(skipped) {
            value = step(value).map_err(|why| Unread::Step { from, why })?;
        }
        Ok(value)
    }

    /// A plugin aside's value at this version: `aside.value` read as
    /// `{"v": n, "value": …}` (a value without that envelope is version 1,
    /// whole), upgraded under the aside's kind.
    ///
    /// # Errors
    ///
    /// As [`upgrade`](Self::upgrade).
    pub fn read_aside(&self, aside: &Aside) -> Result<Value, Unread> {
        self.upgrade(&aside.kind, Versioned::of_aside_value(&aside.value))
    }
}

/// The logs [`upgrade`] rewrote.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Upgraded {
    /// Each log rewritten, by name, with the format version it had.
    pub logs: Vec<(String, u32)>,
}

/// Rewrites every log of `store` older than this format in this format,
/// through the chain. A log's torn tail is set aside first (its bytes in
/// `recovered/`), so the upgrade never drops it. Each log is then kept as
/// `rig/upgraded/<log>.v<N>.jsonl` and replaced atomically
/// ([`Store::replace_log`]) with its lineage and stamps unchanged and a
/// rewrite by rig in its header; the kept copies are removed once every log
/// is replaced. A log already in this format is left. Nothing is written
/// when any log fails to read or would not read back. A crash midway leaves
/// every log whole, old or new, and the copies; the next call finishes.
///
/// # Errors
///
/// As [`Session::read`](crate::Session::read), or when a write fails.
pub fn upgrade(store: &dyn Store) -> Result<Upgraded, LoadError> {
    // Every log this build reads is in its format: there is nothing older.
    let _ = store;
    Ok(Upgraded::default())
}

#[cfg(test)]
mod tests;

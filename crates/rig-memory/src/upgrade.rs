//! Reading what older rigs saved: the versions of saved values and of
//! plugins' asides ([`Versioned`], [`Upgrades`]). A saved value carries its
//! own version, which a plugin moves on with steps of its own
//! ([`Upgrades::add`]); a value no step reads is shelved under [`SHELVED`],
//! never dropped.

use std::collections::BTreeMap;

use rig_core::conversation::Aside;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The prefix of the key a saved value no step reads is moved to before its
/// key is written again: `rig.set_aside/<key>/v<N>`. Readers skip such keys
/// silently; they are kept in the file.
pub const SHELVED: &str = "rig.set_aside/";

/// A saved value and the version its type had when it was saved (1 for one
/// saved before versions). Its serde form is `{"v": n, "value": …}`, without
/// `v` at 1.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Versioned {
    /// The version of its type it was saved at, from 1.
    #[serde(default = "one", skip_serializing_if = "is_one")]
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

fn one() -> u32 {
    1
}

fn is_one(v: &u32) -> bool {
    *v == 1
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

/// Why a saved value could not be read as a type.
#[derive(Debug, thiserror::Error)]
pub enum ValueError {
    /// It was set aside: no step reads its version.
    #[error(transparent)]
    Unread(#[from] Unread),
    /// At this version it does not parse as the type.
    #[error("the saved value does not parse: {0}")]
    Parse(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests;

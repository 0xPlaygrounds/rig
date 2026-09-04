//! The string identity a runtime names a model by.
//!
//! Live model behaviour is reached through the effect bus: a
//! `ModelHandle` (`rig_agent::bus`) is a typed view bound to the key
//! a [`CompletionAdapter`](crate::serve::adapters::CompletionAdapter) was
//! registered under. `ModelRef` is the serializable half — the label under
//! which a runtime resolves that key.

use std::{fmt, sync::Arc};

use serde::{Deserialize, Serialize};

/// The string identity a specification, asset, or registry names a model by.
///
/// A handle is live process state and is never serialized; a `ModelRef` is
/// the serializable half — the label under which a runtime resolves a
/// handle (`ModelRef → HandlerKey → ModelHandle`). It carries no provider
/// semantics: two refs are equal when their strings are equal.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelRef(Arc<str>);

// Transparent string (de)serialization without serde's `rc` feature.
impl Serialize for ModelRef {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ModelRef {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let label = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        Ok(Self(Arc::from(&*label)))
    }
}

impl ModelRef {
    /// Build a reference from any string-like value.
    pub fn new(label: impl Into<Arc<str>>) -> Self {
        Self(label.into())
    }

    /// The label as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::ops::Deref for ModelRef {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for ModelRef {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for ModelRef {
    fn from(label: &str) -> Self {
        Self(Arc::from(label))
    }
}

impl From<String> for ModelRef {
    fn from(label: String) -> Self {
        Self(Arc::from(label))
    }
}

impl From<Arc<str>> for ModelRef {
    fn from(label: Arc<str>) -> Self {
        Self(label)
    }
}

impl From<ModelRef> for String {
    fn from(label: ModelRef) -> Self {
        label.0.to_string()
    }
}

impl PartialEq<str> for ModelRef {
    fn eq(&self, other: &str) -> bool {
        &*self.0 == other
    }
}

impl PartialEq<&str> for ModelRef {
    fn eq(&self, other: &&str) -> bool {
        &*self.0 == *other
    }
}

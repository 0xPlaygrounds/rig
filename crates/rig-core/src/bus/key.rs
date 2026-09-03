//! A family-typed handler key.

use std::{fmt, hash::Hash, marker::PhantomData};

use serde::{Deserialize, Serialize};

use crate::effect::{Family, HandlerKey};

/// A [`HandlerKey`] that carries the family it serves in its type: what rig
/// mints for the registrations it makes (an agent's model, memory and
/// retrieval keys, a registry's tool generations) and what
/// [`Registrar::register_typed`](super::Registrar::register_typed) returns.
/// [`Dispatcher::bind`](super::Dispatcher::bind) binds one with an
/// existence check only — the family was proven when the key was minted.
///
/// On the wire a `Key<F>` is the bare string (`serde(transparent)`), so a
/// log, a scene and a cassette hold exactly what they held before; an
/// explicit or replayed key stays a [`HandlerKey`] and binds through
/// [`Dispatcher::handle`](super::Dispatcher::handle), which checks the
/// family. `Send + Sync` for every `F`.
#[derive(Serialize, Deserialize)]
#[serde(transparent, bound = "")]
pub struct Key<F: Family> {
    raw: HandlerKey,
    #[serde(skip)]
    _family: PhantomData<fn() -> F>,
}

impl<F: Family> Key<F> {
    /// Assert that `raw` serves `F`. The one place a family is asserted
    /// rather than proven: for a host that registered the handler itself
    /// and knows what it serves. A wrong assertion fails at bind time
    /// (`HandlerUnavailable`) or at the first dispatch, not silently.
    pub const fn new_unchecked(raw: HandlerKey) -> Self {
        Self {
            raw,
            _family: PhantomData,
        }
    }

    /// The wire key.
    pub const fn raw(&self) -> &HandlerKey {
        &self.raw
    }

    /// The wire key, by value.
    pub fn into_raw(self) -> HandlerKey {
        self.raw
    }

    /// The key as a string.
    pub fn as_str(&self) -> &str {
        self.raw.as_str()
    }
}

impl<F: Family> From<Key<F>> for HandlerKey {
    fn from(key: Key<F>) -> Self {
        key.raw
    }
}

impl<F: Family> AsRef<HandlerKey> for Key<F> {
    fn as_ref(&self) -> &HandlerKey {
        &self.raw
    }
}

// Written by hand: a derive would demand `F: Clone` and friends.
impl<F: Family> Clone for Key<F> {
    fn clone(&self) -> Self {
        Self::new_unchecked(self.raw.clone())
    }
}
impl<F: Family> PartialEq for Key<F> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}
impl<F: Family> Eq for Key<F> {}
impl<F: Family> PartialOrd for Key<F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<F: Family> Ord for Key<F> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.raw.cmp(&other.raw)
    }
}
impl<F: Family> Hash for Key<F> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}
impl<F: Family> fmt::Debug for Key<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Key<{}>({:?})", F::FAMILY, self.raw.as_str())
    }
}
impl<F: Family> fmt::Display for Key<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.raw, f)
    }
}

// The phantom is `fn() -> F`, so the key crosses threads for every `F`.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<Key<crate::effect::family::Completion>>();
};

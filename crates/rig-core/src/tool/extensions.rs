//! Per-call runtime extensions for tool execution.
//!
//! This module provides [`ToolCallExtensions`], a type-map that allows callers
//! to attach arbitrary typed values and tools to extract them at call time.
//! Tools that don't need any extensions simply ignore them.
//!
//! The implementation follows the `AnyClone` pattern from the [`http::Extensions`]
//! type in the `http` crate, including the no-op [`IdHasher`] over `TypeId` keys.
//!
//! [`http::Extensions`]: https://docs.rs/http/latest/http/struct.Extensions.html

use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

type AnyMap = HashMap<TypeId, Box<dyn AnyClone>, BuildHasherDefault<IdHasher>>;

// --- IdHasher (modeled after `http::Extensions`) ---

/// Hasher for the `TypeId` keys of the type-map.
///
/// A `TypeId` is already a high-quality hash, so the keys do not need to be
/// re-hashed with the default `SipHash`. The fast path stores the `u64` that
/// `TypeId`'s `Hash` impl writes directly. The byte-oriented [`write`](Hasher::write)
/// fallback keeps this correct (never panicking) if a future `TypeId`
/// representation hashes via a different `Hasher` method — a poor hash only
/// costs extra probe work, never correctness, because the map still compares
/// full `TypeId` keys.
#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    #[inline]
    fn write_u64(&mut self, id: u64) {
        self.0 = id;
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Fallback path: fold the bytes so we never panic regardless of how
        // `TypeId` chooses to hash itself.
        for &byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(byte);
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

// --- AnyClone helper trait (modeled after `http::Extensions`) ---
//
// `WasmCompatSend`/`WasmCompatSync` are `Send`/`Sync` on every non-wasm target
// and unconstrained on `wasm`, so a single trait definition covers both targets
// (no `cfg`-duplicated copies). On non-wasm this transitively makes
// `dyn AnyClone` — and therefore `ToolCallExtensions` — `Send + Sync`, which is
// required because the context is borrowed across `.await` points in async tool
// execution. The `assert_send_sync` check below pins that property.

trait AnyClone: Any + WasmCompatSend + WasmCompatSync {
    fn clone_box(&self) -> Box<dyn AnyClone>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
    fn type_name(&self) -> &'static str;
}

impl<T: Clone + WasmCompatSend + WasmCompatSync + 'static> AnyClone for T {
    fn clone_box(&self) -> Box<dyn AnyClone> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
    fn type_name(&self) -> &'static str {
        type_name::<T>()
    }
}

impl Clone for Box<dyn AnyClone> {
    fn clone(&self) -> Self {
        // Explicit deref to dispatch via the trait object's vtable, not the
        // blanket `AnyClone` impl on `Box` itself (which would recurse).
        (**self).clone_box()
    }
}

// --- ToolCallExtensions ---

/// Per-call runtime context for tools.
///
/// A type-map that allows callers to attach arbitrary typed values and tools to
/// extract them. Tools that don't need context ignore it.
///
/// Inspired by [`http::Extensions`](https://docs.rs/http/latest/http/struct.Extensions.html).
/// Uses `Option<Box<HashMap>>` internally so that empty contexts (the common
/// case when no caller-provided values are needed) require zero allocation.
///
/// Values are keyed by [`TypeId`], so [`get`](Self::get) returns `None` both
/// when nothing was inserted under that type *and* when a different type was
/// inserted. For tools that genuinely require a value, prefer
/// [`require`](Self::require), which returns a descriptive error instead of a
/// silent `None`.
///
/// # Example
/// ```
/// use rig_core::tool::ToolCallExtensions;
///
/// let mut extensions = ToolCallExtensions::new();
/// assert_eq!(extensions.insert(42u32), None); // no prior value
/// assert_eq!(extensions.get::<u32>(), Some(&42));
/// assert_eq!(extensions.insert(7u32), Some(42)); // returns the displaced value
/// ```
#[derive(Default, Clone)]
pub struct ToolCallExtensions {
    map: Option<Box<AnyMap>>,
}

impl ToolCallExtensions {
    /// Shared empty instance. Lets dispatch layers that need a default context
    /// hand out a `'static` reference instead of constructing (and having to
    /// own) a fresh value just to borrow it.
    pub(crate) const EMPTY: ToolCallExtensions = ToolCallExtensions { map: None };

    /// Create an empty context.
    pub const fn new() -> Self {
        Self::EMPTY
    }

    /// Insert a typed value, returning the previous value of the same type if
    /// one was present (mirroring [`http::Extensions::insert`] and
    /// [`HashMap::insert`]).
    ///
    /// [`http::Extensions::insert`]: https://docs.rs/http/latest/http/struct.Extensions.html#method.insert
    pub fn insert<T: Clone + WasmCompatSend + WasmCompatSync + 'static>(
        &mut self,
        val: T,
    ) -> Option<T> {
        self.map
            .get_or_insert_with(Default::default)
            .insert(TypeId::of::<T>(), Box::new(val))
            .and_then(|prev| prev.into_any().downcast::<T>().ok())
            .map(|boxed| *boxed)
    }

    /// Get a reference to a value by type. Returns `None` if not present.
    pub fn get<T: WasmCompatSend + WasmCompatSync + 'static>(&self) -> Option<&T> {
        self.map
            .as_ref()
            .and_then(|map| map.get(&TypeId::of::<T>()))
            // Explicit deref to dispatch via the trait object's vtable, not the
            // blanket `AnyClone` impl that `Box` itself satisfies.
            .and_then(|boxed| (**boxed).as_any().downcast_ref::<T>())
    }

    /// Get a reference to a value by type, returning a descriptive error instead
    /// of `None` when it is absent.
    ///
    /// Prefer this over [`get`](Self::get) for tools that *require* a context
    /// value (auth tokens, session IDs, …): the error names the missing type,
    /// turning a silent `None` into an actionable failure.
    pub fn require<T: WasmCompatSend + WasmCompatSync + 'static>(
        &self,
    ) -> Result<&T, MissingExtension> {
        self.get::<T>().ok_or(MissingExtension(type_name::<T>()))
    }

    /// Get a mutable reference to a value by type. Returns `None` if not present.
    pub fn get_mut<T: WasmCompatSend + WasmCompatSync + 'static>(&mut self) -> Option<&mut T> {
        self.map
            .as_mut()
            .and_then(|map| map.get_mut(&TypeId::of::<T>()))
            .and_then(|boxed| (**boxed).as_any_mut().downcast_mut::<T>())
    }

    /// Remove a value by type, returning it if present.
    pub fn remove<T: WasmCompatSend + WasmCompatSync + 'static>(&mut self) -> Option<T> {
        self.map
            .as_mut()
            .and_then(|map| map.remove(&TypeId::of::<T>()))
            .and_then(|boxed| boxed.into_any().downcast::<T>().ok())
            .map(|boxed| *boxed)
    }

    /// Check if a value of the given type is present.
    pub fn contains<T: WasmCompatSend + WasmCompatSync + 'static>(&self) -> bool {
        self.map
            .as_ref()
            .is_some_and(|map| map.contains_key(&TypeId::of::<T>()))
    }

    /// Number of values currently stored.
    pub fn len(&self) -> usize {
        self.map.as_ref().map_or(0, |map| map.len())
    }

    /// Whether no values are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for ToolCallExtensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("ToolCallExtensions");
        if let Some(map) = &self.map {
            dbg.field("entries", &map.len());
            let type_names: Vec<&'static str> = map.values().map(|v| (**v).type_name()).collect();
            dbg.field("types", &type_names);
        } else {
            dbg.field("entries", &0);
        }
        dbg.finish()
    }
}

/// Error returned by [`ToolCallExtensions::require`] when the requested value is
/// not present in the context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("required tool-call context value of type `{0}` was not found")]
pub struct MissingExtension(pub &'static str);

// `ToolCallExtensions` must stay `Send + Sync` on native targets: the agent loop
// borrows it across `.await` while executing tools. This fails to compile if a
// future change (e.g. relaxing the `AnyClone` bounds) drops the property.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ToolCallExtensions>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get_returns_value() {
        let mut extensions = ToolCallExtensions::new();
        assert_eq!(extensions.insert(42u32), None);
        assert_eq!(extensions.get::<u32>(), Some(&42));
    }

    #[test]
    fn get_missing_type_returns_none() {
        let extensions = ToolCallExtensions::new();
        assert_eq!(extensions.get::<u32>(), None);
    }

    #[test]
    fn insert_overwrites_and_returns_previous() {
        let mut extensions = ToolCallExtensions::new();
        assert_eq!(extensions.insert(1u32), None);
        assert_eq!(extensions.insert(2u32), Some(1));
        assert_eq!(extensions.get::<u32>(), Some(&2));
    }

    #[test]
    fn different_types_are_independent() {
        let mut extensions = ToolCallExtensions::new();
        extensions.insert(42u32);
        extensions.insert("hello".to_string());
        assert_eq!(extensions.get::<u32>(), Some(&42));
        assert_eq!(extensions.get::<String>(), Some(&"hello".to_string()));
    }

    #[test]
    fn contains_returns_true_when_present() {
        let mut extensions = ToolCallExtensions::new();
        extensions.insert(42u32);
        assert!(extensions.contains::<u32>());
        assert!(!extensions.contains::<String>());
    }

    #[test]
    fn clone_produces_independent_copy() {
        let mut extensions = ToolCallExtensions::new();
        extensions.insert(42u32);
        let mut cloned = extensions.clone();
        cloned.insert(99u32);
        assert_eq!(extensions.get::<u32>(), Some(&42));
        assert_eq!(cloned.get::<u32>(), Some(&99));
    }

    #[test]
    fn clone_deep_copies_inner_value() {
        // Insert a heap-allocated value, clone the context, then mutate the
        // clone's inner value in place. A shallow clone (sharing the boxed
        // value) would let this mutation leak back into the original; a correct
        // `clone_box` deep-copies, so the original stays unchanged.
        let mut extensions = ToolCallExtensions::new();
        extensions.insert(vec![1u8, 2, 3]);
        let mut cloned = extensions.clone();
        cloned.get_mut::<Vec<u8>>().unwrap().push(4);
        assert_eq!(extensions.get::<Vec<u8>>(), Some(&vec![1, 2, 3]));
        assert_eq!(cloned.get::<Vec<u8>>(), Some(&vec![1, 2, 3, 4]));
    }

    #[test]
    fn debug_shows_entry_count_and_types() {
        let mut extensions = ToolCallExtensions::new();
        extensions.insert(42u32);
        extensions.insert("hi".to_string());
        let debug = format!("{:?}", extensions);
        assert!(debug.contains("entries: 2"));
        assert!(debug.contains("u32"));
        assert!(debug.contains("String"));
    }

    #[test]
    fn empty_context_is_default() {
        let extensions = ToolCallExtensions::default();
        assert!(!extensions.contains::<u32>());
    }

    #[test]
    fn empty_context_has_no_allocation() {
        let extensions = ToolCallExtensions::new();
        assert!(extensions.map.is_none());
    }

    #[test]
    fn get_mut_modifies_in_place() {
        let mut extensions = ToolCallExtensions::new();
        extensions.insert(42u32);
        if let Some(val) = extensions.get_mut::<u32>() {
            *val = 99;
        }
        assert_eq!(extensions.get::<u32>(), Some(&99));
    }

    #[test]
    fn remove_returns_value_and_clears_entry() {
        let mut extensions = ToolCallExtensions::new();
        extensions.insert(42u32);
        assert_eq!(extensions.remove::<u32>(), Some(42));
        assert!(!extensions.contains::<u32>());
    }

    #[test]
    fn remove_missing_type_returns_none() {
        let mut extensions = ToolCallExtensions::new();
        assert_eq!(extensions.remove::<u32>(), None);
    }

    #[test]
    fn require_present_returns_value() {
        let mut extensions = ToolCallExtensions::new();
        extensions.insert(42u32);
        assert_eq!(extensions.require::<u32>().copied(), Ok(42));
    }

    #[test]
    fn require_missing_names_the_type() {
        let extensions = ToolCallExtensions::new();
        let err = extensions.require::<u32>().unwrap_err();
        assert!(err.to_string().contains("u32"));
    }

    #[test]
    fn len_and_is_empty_track_entries() {
        let mut extensions = ToolCallExtensions::new();
        assert!(extensions.is_empty());
        assert_eq!(extensions.len(), 0);
        extensions.insert(1u32);
        extensions.insert("two".to_string());
        assert!(!extensions.is_empty());
        assert_eq!(extensions.len(), 2);
        extensions.remove::<u32>();
        assert_eq!(extensions.len(), 1);
    }

    #[test]
    fn many_distinct_types_round_trip_through_id_hasher() {
        // Guards the custom IdHasher: distinct TypeId keys must not collide in a
        // way that corrupts lookups. Insert several heterogeneous types and
        // confirm each is retrievable with its own value.
        #[derive(Clone, PartialEq, Debug)]
        struct A(u8);
        #[derive(Clone, PartialEq, Debug)]
        struct B(u16);
        #[derive(Clone, PartialEq, Debug)]
        struct C(u32);
        #[derive(Clone, PartialEq, Debug)]
        struct D(u64);

        let mut extensions = ToolCallExtensions::new();
        extensions.insert(A(1));
        extensions.insert(B(2));
        extensions.insert(C(3));
        extensions.insert(D(4));
        extensions.insert(5u8);
        extensions.insert(6i64);
        extensions.insert("seven".to_string());
        extensions.insert(8.0f64);

        assert_eq!(extensions.len(), 8);
        assert_eq!(extensions.get::<A>(), Some(&A(1)));
        assert_eq!(extensions.get::<B>(), Some(&B(2)));
        assert_eq!(extensions.get::<C>(), Some(&C(3)));
        assert_eq!(extensions.get::<D>(), Some(&D(4)));
        assert_eq!(extensions.get::<u8>(), Some(&5));
        assert_eq!(extensions.get::<i64>(), Some(&6));
        assert_eq!(extensions.get::<String>(), Some(&"seven".to_string()));
        assert_eq!(extensions.get::<f64>(), Some(&8.0));
    }
}

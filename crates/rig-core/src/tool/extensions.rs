//! Per-call runtime context for tool execution.
//!
//! This module provides [`ToolCallExtensions`], a type-map that lets callers
//! attach arbitrary typed values to a tool call and lets tools extract them at
//! call time. Tools that don't need any extensions simply ignore them.
//!
//! The implementation follows the `AnyClone` pattern from the
//! [`http::Extensions`](https://docs.rs/http/latest/http/struct.Extensions.html)
//! type in the `http` crate, including the no-op [`IdHasher`] over `TypeId`
//! keys.

use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[cfg(not(target_family = "wasm"))]
type AnyMap = HashMap<TypeId, Box<dyn AnyClone + Send + Sync>, BuildHasherDefault<IdHasher>>;

#[cfg(target_family = "wasm")]
type AnyMap = HashMap<TypeId, Box<dyn AnyClone>, BuildHasherDefault<IdHasher>>;

// --- IdHasher (modeled after http::Extensions) ---

/// Hasher for the `TypeId` keys of the type-map.
///
/// A `TypeId` is already a high-quality hash, so the keys do not need to be
/// re-hashed with the default `SipHash`. The fast path stores the `u64` that
/// `TypeId`'s `Hash` impl writes directly. The byte-oriented [`write`] fallback
/// keeps this correct (never panicking) if a future `TypeId` representation
/// hashes via a different `Hasher` method — a poor hash only costs extra probe
/// work, never correctness, because the map still compares full `TypeId` keys.
///
/// [`write`]: Hasher::write
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

// --- AnyClone helper trait (modeled after http::Extensions) ---

#[cfg(not(target_family = "wasm"))]
trait AnyClone: Any + Send + Sync {
    fn clone_box(&self) -> Box<dyn AnyClone + Send + Sync>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
    fn type_name(&self) -> &'static str;
}

#[cfg(target_family = "wasm")]
trait AnyClone: Any {
    fn clone_box(&self) -> Box<dyn AnyClone>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
    fn type_name(&self) -> &'static str;
}

#[cfg(not(target_family = "wasm"))]
impl<T: Clone + Send + Sync + 'static> AnyClone for T {
    fn clone_box(&self) -> Box<dyn AnyClone + Send + Sync> {
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

#[cfg(target_family = "wasm")]
impl<T: Clone + 'static> AnyClone for T {
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

#[cfg(not(target_family = "wasm"))]
impl Clone for Box<dyn AnyClone + Send + Sync> {
    fn clone(&self) -> Self {
        // Explicit deref to dispatch via the trait object's vtable, not the
        // blanket AnyClone impl on Box itself (which would recurse).
        (**self).clone_box()
    }
}

#[cfg(target_family = "wasm")]
impl Clone for Box<dyn AnyClone> {
    fn clone(&self) -> Self {
        (**self).clone_box()
    }
}

// --- ToolCallExtensions ---

/// Per-call runtime extensions for tools.
///
/// A type-map that lets callers attach arbitrary typed values to a tool call
/// (auth tokens, session IDs, conversation state, A2A `context_id`/`task_id`,
/// …) and lets tools extract them by type. Tools that don't need any extension
/// ignore it.
///
/// Inject extensions into an agent run with
/// [`PromptRequest::with_tool_extensions`](crate::agent::prompt_request::PromptRequest::with_tool_extensions)
/// (or the streaming equivalent); read them inside a tool by overriding
/// [`Tool::call_with_extensions`](crate::tool::Tool::call_with_extensions).
///
/// Inspired by [`http::Extensions`](https://docs.rs/http/latest/http/struct.Extensions.html).
/// Uses `Option<Box<HashMap>>` internally so that empty extensions (the common
/// case when no caller-provided values are needed) require zero allocation, and
/// a no-op [`IdHasher`] over the `TypeId` keys to avoid re-hashing them.
///
/// # Example
/// ```
/// use rig_core::tool::ToolCallExtensions;
///
/// let mut extensions = ToolCallExtensions::new();
/// assert_eq!(extensions.insert(42u32), None);
/// assert_eq!(extensions.get::<u32>(), Some(&42));
/// ```
#[derive(Default, Clone)]
pub struct ToolCallExtensions {
    map: Option<Box<AnyMap>>,
}

impl ToolCallExtensions {
    /// Shared empty instance. Lets dispatch layers that need a default
    /// value hand out a `'static` reference instead of constructing (and
    /// having to own) a fresh value just to borrow it.
    pub(crate) const EMPTY: ToolCallExtensions = ToolCallExtensions { map: None };

    /// Create an empty set of extensions.
    pub const fn new() -> Self {
        Self::EMPTY
    }

    /// Insert a typed value, returning the previous value of the same type if
    /// one was present (matching [`http::Extensions::insert`]).
    #[cfg(not(target_family = "wasm"))]
    pub fn insert<T: Clone + Send + Sync + 'static>(&mut self, val: T) -> Option<T> {
        self.map
            .get_or_insert_with(Default::default)
            .insert(TypeId::of::<T>(), Box::new(val))
            // The displaced value was stored under the same `TypeId`, so the
            // downcast is guaranteed to succeed.
            .and_then(|boxed| boxed.into_any().downcast::<T>().ok())
            .map(|boxed| *boxed)
    }

    /// Insert a typed value, returning the previous value of the same type if
    /// one was present (matching [`http::Extensions::insert`]).
    #[cfg(target_family = "wasm")]
    pub fn insert<T: Clone + 'static>(&mut self, val: T) -> Option<T> {
        self.map
            .get_or_insert_with(Default::default)
            .insert(TypeId::of::<T>(), Box::new(val))
            .and_then(|boxed| boxed.into_any().downcast::<T>().ok())
            .map(|boxed| *boxed)
    }

    /// Get a reference to a value by type. Returns `None` if not present.
    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.map
            .as_ref()
            .and_then(|map| map.get(&TypeId::of::<T>()))
            // Explicit deref to dispatch via the trait object's vtable, not
            // the blanket AnyClone impl that Box itself satisfies.
            .and_then(|boxed| (**boxed).as_any().downcast_ref::<T>())
    }

    /// Get a mutable reference to a value by type. Returns `None` if not present.
    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.map
            .as_mut()
            .and_then(|map| map.get_mut(&TypeId::of::<T>()))
            .and_then(|boxed| (**boxed).as_any_mut().downcast_mut::<T>())
    }

    /// Remove a value by type, returning it if present.
    pub fn remove<T: 'static>(&mut self) -> Option<T> {
        self.map
            .as_mut()
            .and_then(|map| map.remove(&TypeId::of::<T>()))
            .and_then(|boxed| boxed.into_any().downcast::<T>().ok())
            .map(|boxed| *boxed)
    }

    /// Check if a value of the given type is present.
    pub fn contains<T: 'static>(&self) -> bool {
        self.map
            .as_ref()
            .is_some_and(|map| map.contains_key(&TypeId::of::<T>()))
    }

    /// The number of values currently stored.
    pub fn len(&self) -> usize {
        self.map.as_ref().map_or(0, |map| map.len())
    }

    /// Whether no values are stored.
    pub fn is_empty(&self) -> bool {
        self.map.as_ref().is_none_or(|map| map.is_empty())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get_returns_value() {
        let mut extensions = ToolCallExtensions::new();
        extensions.insert(42u32);
        assert_eq!(extensions.get::<u32>(), Some(&42));
    }

    #[test]
    fn get_missing_type_returns_none() {
        let extensions = ToolCallExtensions::new();
        assert_eq!(extensions.get::<u32>(), None);
    }

    #[test]
    fn insert_overwrites_same_type_and_returns_previous() {
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
    fn len_and_is_empty_track_entries() {
        let mut extensions = ToolCallExtensions::new();
        assert!(extensions.is_empty());
        assert_eq!(extensions.len(), 0);
        extensions.insert(42u32);
        extensions.insert("hello".to_string());
        assert!(!extensions.is_empty());
        assert_eq!(extensions.len(), 2);
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
    fn empty_extensions_is_default() {
        let extensions = ToolCallExtensions::default();
        assert!(!extensions.contains::<u32>());
    }

    #[test]
    fn empty_extensions_has_no_allocation() {
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
    fn many_distinct_types_round_trip_through_id_hasher() {
        // Exercises the IdHasher across many TypeId keys to guard against
        // collisions corrupting lookups.
        let mut extensions = ToolCallExtensions::new();
        extensions.insert(1u8);
        extensions.insert(2u16);
        extensions.insert(3u32);
        extensions.insert(4u64);
        extensions.insert(5i8);
        extensions.insert(6i16);
        extensions.insert(7i32);
        extensions.insert("eight".to_string());
        assert_eq!(extensions.get::<u8>(), Some(&1));
        assert_eq!(extensions.get::<u16>(), Some(&2));
        assert_eq!(extensions.get::<u32>(), Some(&3));
        assert_eq!(extensions.get::<u64>(), Some(&4));
        assert_eq!(extensions.get::<i8>(), Some(&5));
        assert_eq!(extensions.get::<i16>(), Some(&6));
        assert_eq!(extensions.get::<i32>(), Some(&7));
        assert_eq!(extensions.get::<String>(), Some(&"eight".to_string()));
        assert_eq!(extensions.len(), 8);
    }
}

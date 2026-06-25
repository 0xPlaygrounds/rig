//! Per-call runtime context for tool execution.
//!
//! This module provides [`ToolCallContext`], a type-map that allows callers
//! to attach arbitrary typed values and tools to extract them at call time.
//! Tools that don't need context simply ignore it.
//!
//! The implementation follows the `AnyClone` pattern from the [`http::Extensions`]
//! type in the `http` crate.
//!
//! [`http::Extensions`]: https://docs.rs/http/latest/http/struct.Extensions.html

use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;

use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

type AnyMap = HashMap<TypeId, Box<dyn AnyClone>>;

// --- AnyClone helper trait (modeled after `http::Extensions`) ---
//
// `WasmCompatSend`/`WasmCompatSync` are `Send`/`Sync` on every non-wasm target
// and unconstrained on `wasm`, so a single trait definition covers both targets
// (no `cfg`-duplicated copies). On non-wasm this transitively makes
// `dyn AnyClone` — and therefore `ToolCallContext` — `Send + Sync`, which is
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

// --- ToolCallContext ---

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
/// use rig_core::tool::ToolCallContext;
///
/// let mut ctx = ToolCallContext::new();
/// assert_eq!(ctx.insert(42u32), None); // no prior value
/// assert_eq!(ctx.get::<u32>(), Some(&42));
/// assert_eq!(ctx.insert(7u32), Some(42)); // returns the displaced value
/// ```
#[derive(Default, Clone)]
pub struct ToolCallContext {
    map: Option<Box<AnyMap>>,
}

impl ToolCallContext {
    /// Shared empty instance. Lets dispatch layers that need a default context
    /// hand out a `'static` reference instead of constructing (and having to
    /// own) a fresh value just to borrow it.
    pub(crate) const EMPTY: ToolCallContext = ToolCallContext { map: None };

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
    ) -> Result<&T, MissingContextValue> {
        self.get::<T>().ok_or(MissingContextValue(type_name::<T>()))
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
}

impl std::fmt::Debug for ToolCallContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("ToolCallContext");
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

/// Error returned by [`ToolCallContext::require`] when the requested value is
/// not present in the context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("required tool-call context value of type `{0}` was not found")]
pub struct MissingContextValue(pub &'static str);

// `ToolCallContext` must stay `Send + Sync` on native targets: the agent loop
// borrows it across `.await` while executing tools. This fails to compile if a
// future change (e.g. relaxing the `AnyClone` bounds) drops the property.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ToolCallContext>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get_returns_value() {
        let mut ctx = ToolCallContext::new();
        assert_eq!(ctx.insert(42u32), None);
        assert_eq!(ctx.get::<u32>(), Some(&42));
    }

    #[test]
    fn get_missing_type_returns_none() {
        let ctx = ToolCallContext::new();
        assert_eq!(ctx.get::<u32>(), None);
    }

    #[test]
    fn insert_overwrites_and_returns_previous() {
        let mut ctx = ToolCallContext::new();
        assert_eq!(ctx.insert(1u32), None);
        assert_eq!(ctx.insert(2u32), Some(1));
        assert_eq!(ctx.get::<u32>(), Some(&2));
    }

    #[test]
    fn different_types_are_independent() {
        let mut ctx = ToolCallContext::new();
        ctx.insert(42u32);
        ctx.insert("hello".to_string());
        assert_eq!(ctx.get::<u32>(), Some(&42));
        assert_eq!(ctx.get::<String>(), Some(&"hello".to_string()));
    }

    #[test]
    fn contains_returns_true_when_present() {
        let mut ctx = ToolCallContext::new();
        ctx.insert(42u32);
        assert!(ctx.contains::<u32>());
        assert!(!ctx.contains::<String>());
    }

    #[test]
    fn clone_produces_independent_copy() {
        let mut ctx = ToolCallContext::new();
        ctx.insert(42u32);
        let mut cloned = ctx.clone();
        cloned.insert(99u32);
        assert_eq!(ctx.get::<u32>(), Some(&42));
        assert_eq!(cloned.get::<u32>(), Some(&99));
    }

    #[test]
    fn clone_deep_copies_inner_value() {
        // Insert a heap-allocated value, clone the context, then mutate the
        // clone's inner value in place. A shallow clone (sharing the boxed
        // value) would let this mutation leak back into the original; a correct
        // `clone_box` deep-copies, so the original stays unchanged.
        let mut ctx = ToolCallContext::new();
        ctx.insert(vec![1u8, 2, 3]);
        let mut cloned = ctx.clone();
        cloned.get_mut::<Vec<u8>>().unwrap().push(4);
        assert_eq!(ctx.get::<Vec<u8>>(), Some(&vec![1, 2, 3]));
        assert_eq!(cloned.get::<Vec<u8>>(), Some(&vec![1, 2, 3, 4]));
    }

    #[test]
    fn debug_shows_entry_count_and_types() {
        let mut ctx = ToolCallContext::new();
        ctx.insert(42u32);
        ctx.insert("hi".to_string());
        let debug = format!("{:?}", ctx);
        assert!(debug.contains("entries: 2"));
        assert!(debug.contains("u32"));
        assert!(debug.contains("String"));
    }

    #[test]
    fn empty_context_is_default() {
        let ctx = ToolCallContext::default();
        assert!(!ctx.contains::<u32>());
    }

    #[test]
    fn empty_context_has_no_allocation() {
        let ctx = ToolCallContext::new();
        assert!(ctx.map.is_none());
    }

    #[test]
    fn get_mut_modifies_in_place() {
        let mut ctx = ToolCallContext::new();
        ctx.insert(42u32);
        if let Some(val) = ctx.get_mut::<u32>() {
            *val = 99;
        }
        assert_eq!(ctx.get::<u32>(), Some(&99));
    }

    #[test]
    fn remove_returns_value_and_clears_entry() {
        let mut ctx = ToolCallContext::new();
        ctx.insert(42u32);
        assert_eq!(ctx.remove::<u32>(), Some(42));
        assert!(!ctx.contains::<u32>());
    }

    #[test]
    fn remove_missing_type_returns_none() {
        let mut ctx = ToolCallContext::new();
        assert_eq!(ctx.remove::<u32>(), None);
    }

    #[test]
    fn require_present_returns_value() {
        let mut ctx = ToolCallContext::new();
        ctx.insert(42u32);
        assert_eq!(ctx.require::<u32>().copied(), Ok(42));
    }

    #[test]
    fn require_missing_names_the_type() {
        let ctx = ToolCallContext::new();
        let err = ctx.require::<u32>().unwrap_err();
        assert!(err.to_string().contains("u32"));
    }
}

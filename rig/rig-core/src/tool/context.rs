//! Per-call runtime context for tool execution.
//!
//! This module provides [`ToolCallContext`], a type-map that allows callers
//! to attach arbitrary typed values and tools to extract them at call time.
//! Tools that don't need context simply ignore it.
//!
//! The implementation follows the `AnyClone` pattern from the [`http::Extensions`]
//! type in the `http` crate.

use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;

#[cfg(not(target_family = "wasm"))]
type AnyMap = HashMap<TypeId, Box<dyn AnyClone + Send + Sync>>;

#[cfg(target_family = "wasm")]
type AnyMap = HashMap<TypeId, Box<dyn AnyClone>>;

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

// --- ToolCallContext ---

/// Per-call runtime context for tools.
///
/// A type-map that allows callers to attach arbitrary typed values
/// and tools to extract them. Tools that don't need context ignore it.
///
/// Inspired by [`http::Extensions`](https://docs.rs/http/latest/http/struct.Extensions.html).
/// Uses `Option<Box<HashMap>>` internally so that empty contexts (the common
/// case when no caller-provided values are needed) require zero allocation.
///
/// # Example
/// ```
/// use rig::tool::ToolCallContext;
///
/// let mut ctx = ToolCallContext::new();
/// ctx.insert(42u32);
/// assert_eq!(ctx.get::<u32>(), Some(&42));
/// ```
#[derive(Default, Clone)]
pub struct ToolCallContext {
    map: Option<Box<AnyMap>>,
}

impl ToolCallContext {
    /// Create an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a typed value. Overwrites any previous value of the same type.
    #[cfg(not(target_family = "wasm"))]
    pub fn insert<T: Clone + Send + Sync + 'static>(&mut self, val: T) {
        self.map
            .get_or_insert_with(Default::default)
            .insert(TypeId::of::<T>(), Box::new(val));
    }

    /// Insert a typed value. Overwrites any previous value of the same type.
    #[cfg(target_family = "wasm")]
    pub fn insert<T: Clone + 'static>(&mut self, val: T) {
        self.map
            .get_or_insert_with(Default::default)
            .insert(TypeId::of::<T>(), Box::new(val));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get_returns_value() {
        let mut ctx = ToolCallContext::new();
        ctx.insert(42u32);
        assert_eq!(ctx.get::<u32>(), Some(&42));
    }

    #[test]
    fn get_missing_type_returns_none() {
        let ctx = ToolCallContext::new();
        assert_eq!(ctx.get::<u32>(), None);
    }

    #[test]
    fn insert_overwrites_same_type() {
        let mut ctx = ToolCallContext::new();
        ctx.insert(1u32);
        ctx.insert(2u32);
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
}

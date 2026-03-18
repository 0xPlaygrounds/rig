//! Per-call runtime context for tool execution.
//!
//! This module provides [`ToolCallContext`], a type-map that allows callers
//! to attach arbitrary typed values and tools to extract them at call time.
//! Tools that don't need context simply ignore it.

use std::any::{Any, TypeId};
use std::collections::HashMap;

// --- CloneableAny helper trait (WASM-gated) ---

#[cfg(not(target_family = "wasm"))]
trait CloneableAny: Any + Send + Sync {
    fn clone_box(&self) -> Box<dyn CloneableAny + Send + Sync>;
    fn as_any(&self) -> &dyn Any;
}

#[cfg(target_family = "wasm")]
trait CloneableAny: Any {
    fn clone_box(&self) -> Box<dyn CloneableAny>;
    fn as_any(&self) -> &dyn Any;
}

#[cfg(not(target_family = "wasm"))]
impl<T: Clone + Send + Sync + 'static> CloneableAny for T {
    fn clone_box(&self) -> Box<dyn CloneableAny + Send + Sync> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(target_family = "wasm")]
impl<T: Clone + 'static> CloneableAny for T {
    fn clone_box(&self) -> Box<dyn CloneableAny> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(not(target_family = "wasm"))]
impl Clone for Box<dyn CloneableAny + Send + Sync> {
    fn clone(&self) -> Self {
        // Explicit deref to dispatch via the trait object's vtable, not the
        // blanket CloneableAny impl on Box itself (which would recurse).
        (**self).clone_box()
    }
}

#[cfg(target_family = "wasm")]
impl Clone for Box<dyn CloneableAny> {
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
    #[cfg(not(target_family = "wasm"))]
    map: HashMap<TypeId, Box<dyn CloneableAny + Send + Sync>>,
    #[cfg(target_family = "wasm")]
    map: HashMap<TypeId, Box<dyn CloneableAny>>,
}

impl ToolCallContext {
    /// Create an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a typed value. Overwrites any previous value of the same type.
    #[cfg(not(target_family = "wasm"))]
    pub fn insert<T: Clone + Send + Sync + 'static>(&mut self, val: T) {
        self.map.insert(TypeId::of::<T>(), Box::new(val));
    }

    /// Insert a typed value. Overwrites any previous value of the same type.
    #[cfg(target_family = "wasm")]
    pub fn insert<T: Clone + 'static>(&mut self, val: T) {
        self.map.insert(TypeId::of::<T>(), Box::new(val));
    }

    /// Get a reference to a value by type. Returns `None` if not present.
    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.map
            .get(&TypeId::of::<T>())
            // Explicit deref to dispatch via the trait object's vtable, not
            // the blanket CloneableAny impl that Box itself satisfies.
            .and_then(|boxed| (**boxed).as_any().downcast_ref::<T>())
    }

    /// Check if a value of the given type is present.
    pub fn contains<T: 'static>(&self) -> bool {
        self.map.contains_key(&TypeId::of::<T>())
    }
}

impl std::fmt::Debug for ToolCallContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolCallContext")
            .field("entries", &self.map.len())
            .finish()
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
    fn debug_shows_entry_count() {
        let mut ctx = ToolCallContext::new();
        ctx.insert(42u32);
        ctx.insert("hi".to_string());
        let debug = format!("{:?}", ctx);
        assert!(debug.contains("entries: 2"));
    }

    #[test]
    fn empty_context_is_default() {
        let ctx = ToolCallContext::default();
        assert!(!ctx.contains::<u32>());
    }
}

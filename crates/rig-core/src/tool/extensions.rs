//! Typed context passed into tool execution.
//!
//! [`ToolContext`] is the single metadata path for tools. Callers populate its
//! inbound type map before dispatch; tools read those values and may attach
//! typed result metadata through the explicit `result_*` methods. Result
//! metadata is surfaced to hooks and telemetry, but is never sent to the model.

use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

type AnyMap = HashMap<TypeId, Box<dyn AnyClone>, BuildHasherDefault<IdHasher>>;

#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    #[inline]
    fn write_u64(&mut self, id: u64) {
        self.0 = id;
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(byte);
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

trait AnyClone: Any + WasmCompatSend + WasmCompatSync {
    fn clone_box(&self) -> Box<dyn AnyClone>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
    fn type_name(&self) -> &'static str;
}

impl<T> AnyClone for T
where
    T: Clone + WasmCompatSend + WasmCompatSync + 'static,
{
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
        (**self).clone_box()
    }
}

#[derive(Default, Clone)]
pub(crate) struct TypeMap {
    map: Option<Box<AnyMap>>,
}

impl TypeMap {
    const EMPTY: Self = Self { map: None };

    fn insert<T>(&mut self, value: T) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .get_or_insert_with(Default::default)
            .insert(TypeId::of::<T>(), Box::new(value))
            .and_then(|previous| previous.into_any().downcast::<T>().ok())
            .map(|value| *value)
    }

    fn get<T>(&self) -> Option<&T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .as_ref()
            .and_then(|map| map.get(&TypeId::of::<T>()))
            .and_then(|value| (**value).as_any().downcast_ref::<T>())
    }

    fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .as_mut()
            .and_then(|map| map.get_mut(&TypeId::of::<T>()))
            .and_then(|value| (**value).as_any_mut().downcast_mut::<T>())
    }

    fn remove<T>(&mut self) -> Option<T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .as_mut()
            .and_then(|map| map.remove(&TypeId::of::<T>()))
            .and_then(|value| value.into_any().downcast::<T>().ok())
            .map(|value| *value)
    }

    fn contains<T>(&self) -> bool
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .as_ref()
            .is_some_and(|map| map.contains_key(&TypeId::of::<T>()))
    }

    fn len(&self) -> usize {
        self.map.as_ref().map_or(0, |map| map.len())
    }

    fn type_names(&self) -> Vec<&'static str> {
        self.map
            .as_ref()
            .map(|map| map.values().map(|value| (**value).type_name()).collect())
            .unwrap_or_default()
    }
}

/// Typed metadata available during one tool execution.
///
/// Callers populate inbound values with [`insert`](Self::insert). A tool reads
/// them with [`get`](Self::get) or [`require`](Self::require), and attaches
/// outbound metadata with [`insert_result`](Self::insert_result). Each dispatch
/// receives an independent clone, so result metadata cannot leak between calls.
#[derive(Default, Clone)]
pub struct ToolContext {
    inbound: TypeMap,
    result: TypeMap,
}

impl ToolContext {
    /// Create an empty context without allocating.
    pub const fn new() -> Self {
        Self {
            inbound: TypeMap::EMPTY,
            result: TypeMap::EMPTY,
        }
    }

    /// Insert a caller-provided typed value.
    pub fn insert<T>(&mut self, value: T) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.inbound.insert(value)
    }

    /// Read a caller-provided typed value.
    pub fn get<T>(&self) -> Option<&T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.inbound.get::<T>()
    }

    /// Require a caller-provided typed value.
    pub fn require<T>(&self) -> Result<&T, MissingExtension>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.get::<T>().ok_or(MissingExtension(type_name::<T>()))
    }

    /// Mutably access a caller-provided typed value in this call's context.
    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.inbound.get_mut::<T>()
    }

    /// Remove a caller-provided typed value from this call's context.
    pub fn remove<T>(&mut self) -> Option<T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.inbound.remove::<T>()
    }

    /// Check whether an inbound value of type `T` is present.
    pub fn contains<T>(&self) -> bool
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.inbound.contains::<T>()
    }

    /// Insert typed metadata to accompany the tool result.
    pub fn insert_result<T>(&mut self, value: T) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.result.insert(value)
    }

    /// Read metadata attached to the tool result.
    pub fn result<T>(&self) -> Option<&T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.result.get::<T>()
    }

    /// Require metadata attached to the tool result.
    pub fn require_result<T>(&self) -> Result<&T, MissingExtension>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.result::<T>().ok_or(MissingExtension(type_name::<T>()))
    }

    /// Check whether result metadata of type `T` is present.
    pub fn contains_result<T>(&self) -> bool
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.result.contains::<T>()
    }

    /// Number of inbound values.
    pub fn len(&self) -> usize {
        self.inbound.len()
    }

    /// Whether the inbound context is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn clear_result_metadata(&mut self) {
        self.result = TypeMap::EMPTY;
    }
}

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("inbound_entries", &self.inbound.len())
            .field("inbound_types", &self.inbound.type_names())
            .field("result_entries", &self.result.len())
            .field("result_types", &self.result.type_names())
            .finish()
    }
}

/// Error returned when required typed tool context is absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("required tool context value of type `{0}` was not found")]
pub struct MissingExtension(pub &'static str);

#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ToolContext>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inbound_and_result_channels_are_distinct() {
        let mut context = ToolContext::new();
        context.insert(42u32);
        context.insert_result("request-1".to_string());

        assert_eq!(context.get::<u32>(), Some(&42));
        assert_eq!(
            context.result::<String>().map(String::as_str),
            Some("request-1")
        );
        assert_eq!(context.get::<String>(), None);
        assert_eq!(context.result::<u32>(), None);
    }

    #[test]
    fn clone_is_independent() {
        let mut context = ToolContext::new();
        context.insert(vec![1u8]);
        let mut cloned = context.clone();
        cloned.get_mut::<Vec<u8>>().unwrap().push(2);
        assert_eq!(context.get::<Vec<u8>>(), Some(&vec![1]));
        assert_eq!(cloned.get::<Vec<u8>>(), Some(&vec![1, 2]));
    }
}

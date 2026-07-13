//! Typed context passed through tool execution.

use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::{
    tool::result::ToolExecutionError,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

type AnyMap = HashMap<TypeId, Box<dyn AnyClone>, BuildHasherDefault<IdHasher>>;

#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    fn write_u64(&mut self, id: u64) {
        self.0 = id;
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(byte);
        }
    }

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

/// Internal type map shared by tool contexts and hook scratchpads.
#[derive(Default, Clone)]
pub(crate) struct TypeMap {
    map: Option<Box<AnyMap>>,
}

impl TypeMap {
    pub(crate) const EMPTY: Self = Self { map: None };

    pub(crate) fn insert<T>(&mut self, value: T) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .get_or_insert_with(Default::default)
            .insert(TypeId::of::<T>(), Box::new(value))
            .and_then(|previous| previous.into_any().downcast::<T>().ok())
            .map(|value| *value)
    }

    pub(crate) fn get<T>(&self) -> Option<&T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .as_ref()
            .and_then(|map| map.get(&TypeId::of::<T>()))
            .and_then(|value| (**value).as_any().downcast_ref::<T>())
    }

    pub(crate) fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .as_mut()
            .and_then(|map| map.get_mut(&TypeId::of::<T>()))
            .and_then(|value| (**value).as_any_mut().downcast_mut::<T>())
    }

    pub(crate) fn remove<T>(&mut self) -> Option<T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .as_mut()
            .and_then(|map| map.remove(&TypeId::of::<T>()))
            .and_then(|value| value.into_any().downcast::<T>().ok())
            .map(|value| *value)
    }

    pub(crate) fn contains<T>(&self) -> bool
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .as_ref()
            .is_some_and(|map| map.contains_key(&TypeId::of::<T>()))
    }

    pub(crate) fn len(&self) -> usize {
        self.map.as_ref().map_or(0, |map| map.len())
    }

    fn type_names(&self) -> Vec<&'static str> {
        self.map
            .as_ref()
            .map(|map| map.values().map(|value| (**value).type_name()).collect())
            .unwrap_or_default()
    }
}

/// Context passed to every tool execution.
///
/// Callers insert typed inbound values with [`insert`](Self::insert). Tools read
/// those values with [`get`](Self::get) or [`require`](Self::require), and attach
/// host-only result metadata with [`insert_result`](Self::insert_result). Result
/// hooks inspect that metadata through [`result`](Self::result). Neither inbound
/// values nor result metadata are sent to the model.
///
/// Registry, server, and agent dispatch clone inbound values once per call.
/// Mutating that snapshot affects only the current tool execution; the
/// dispatch surface returns result metadata without replacing the caller's
/// inbound values.
#[derive(Default, Clone)]
pub struct ToolContext {
    inbound: TypeMap,
    result: TypeMap,
}

impl ToolContext {
    /// Create an empty context.
    pub const fn new() -> Self {
        Self {
            inbound: TypeMap::EMPTY,
            result: TypeMap::EMPTY,
        }
    }

    /// Insert an inbound typed value, returning the displaced value if present.
    pub fn insert<T>(&mut self, value: T) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.inbound.insert(value)
    }

    /// Read an inbound typed value.
    pub fn get<T>(&self) -> Option<&T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.inbound.get::<T>()
    }

    /// Require an inbound typed value.
    pub fn require<T>(&self) -> Result<&T, MissingToolContext>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.get::<T>().ok_or(MissingToolContext(type_name::<T>()))
    }

    /// Mutably access an inbound typed value.
    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.inbound.get_mut::<T>()
    }

    /// Remove an inbound typed value.
    pub fn remove<T>(&mut self) -> Option<T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.inbound.remove::<T>()
    }

    /// Attach host-only metadata to this execution's result.
    pub fn insert_result<T>(&mut self, value: T) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.result.insert(value)
    }

    /// Read host-only result metadata.
    pub fn result<T>(&self) -> Option<&T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.result.get::<T>()
    }

    /// Require host-only result metadata.
    pub fn require_result<T>(&self) -> Result<&T, MissingToolContext>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.result::<T>()
            .ok_or(MissingToolContext(type_name::<T>()))
    }

    /// Whether this context contains the inbound type `T`.
    pub fn contains<T>(&self) -> bool
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.inbound.contains::<T>()
    }

    /// Build a fresh execution context with the same inbound values and no
    /// result metadata.
    ///
    /// Dispatch always runs against this snapshot. A tool may therefore mutate
    /// its local inbound values without changing the run-wide or caller-owned
    /// context that supplied them.
    pub(crate) fn for_dispatch(&self) -> Self {
        Self {
            inbound: self.inbound.clone(),
            result: TypeMap::EMPTY,
        }
    }

    /// Publish metadata produced by one dispatch while preserving the caller's
    /// inbound values.
    pub(crate) fn accept_dispatch_result(&mut self, dispatched: Self) {
        self.result = dispatched.result;
    }

    /// Clone only the inbound values, for nested tool execution.
    pub(crate) fn inbound_only(&self) -> Self {
        self.for_dispatch()
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

/// A required typed value was missing from a [`ToolContext`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("required tool context value of type `{0}` was not found")]
pub struct MissingToolContext(pub &'static str);

impl From<MissingToolContext> for ToolExecutionError {
    fn from(error: MissingToolContext) -> Self {
        ToolExecutionError::other(error.to_string()).with_source(error)
    }
}

#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ToolContext>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_separates_inbound_and_result_values() {
        let mut context = ToolContext::new();
        context.insert(42_u32);
        context.insert_result("request-1".to_string());
        assert_eq!(context.get::<u32>(), Some(&42));
        assert_eq!(
            context.result::<String>().map(String::as_str),
            Some("request-1")
        );

        let next = context.for_dispatch();
        assert_eq!(next.get::<u32>(), Some(&42));
        assert!(next.result::<String>().is_none());
    }

    #[test]
    fn missing_context_converts_into_a_tool_execution_error() {
        fn require_value(context: &ToolContext) -> Result<u32, ToolExecutionError> {
            Ok(*context.require::<u32>()?)
        }

        let error = require_value(&ToolContext::new()).unwrap_err();
        assert!(error.is::<MissingToolContext>());
        assert_eq!(error.model_feedback(), None);
    }
}

#[cfg(test)]
mod migrated_tests {
    use super::*;

    #[test]
    fn insert_and_get_returns_value() {
        let mut c = ToolContext::new();
        assert_eq!(c.insert(42u32), None);
        assert_eq!(c.get::<u32>(), Some(&42));
    }
    #[test]
    fn get_missing_type_returns_none() {
        assert_eq!(ToolContext::new().get::<u32>(), None);
    }
    #[test]
    fn insert_overwrites_and_returns_previous() {
        let mut c = ToolContext::new();
        c.insert(1u32);
        assert_eq!(c.insert(2u32), Some(1));
        assert_eq!(c.get::<u32>(), Some(&2));
    }
    #[test]
    fn different_types_are_independent() {
        let mut c = ToolContext::new();
        c.insert(42u32);
        c.insert("hello".to_string());
        assert_eq!(c.get::<u32>(), Some(&42));
        assert_eq!(c.get::<String>().map(String::as_str), Some("hello"));
    }
    #[test]
    fn contains_tracks_types() {
        let mut c = ToolContext::new();
        c.insert(42u32);
        assert!(c.contains::<u32>());
        assert!(!c.contains::<String>());
    }
    #[test]
    fn clone_produces_independent_copy() {
        let mut c = ToolContext::new();
        c.insert(42u32);
        let mut clone = c.clone();
        clone.insert(99u32);
        assert_eq!(c.get::<u32>(), Some(&42));
        assert_eq!(clone.get::<u32>(), Some(&99));
    }
    #[test]
    fn clone_deep_copies_heap_values() {
        let mut c = ToolContext::new();
        c.insert(vec![1u8, 2, 3]);
        let mut clone = c.clone();
        clone.get_mut::<Vec<u8>>().unwrap().push(4);
        assert_eq!(c.get::<Vec<u8>>(), Some(&vec![1, 2, 3]));
        assert_eq!(clone.get::<Vec<u8>>(), Some(&vec![1, 2, 3, 4]));
    }
    #[test]
    fn empty_context_is_default_and_allocation_free() {
        let c = ToolContext::default();
        assert!(!c.contains::<u32>());
        assert!(c.inbound.map.is_none());
        assert!(c.result.map.is_none());
    }
    #[test]
    fn get_mut_modifies_in_place() {
        let mut c = ToolContext::new();
        c.insert(42u32);
        *c.get_mut::<u32>().unwrap() = 99;
        assert_eq!(c.get::<u32>(), Some(&99));
    }
    #[test]
    fn remove_returns_value_and_clears_entry() {
        let mut c = ToolContext::new();
        c.insert(42u32);
        assert_eq!(c.remove::<u32>(), Some(42));
        assert!(!c.contains::<u32>());
    }
    #[test]
    fn remove_missing_type_returns_none() {
        assert_eq!(ToolContext::new().remove::<u32>(), None);
    }
    #[test]
    fn require_present_returns_value() {
        let mut c = ToolContext::new();
        c.insert(42u32);
        assert_eq!(c.require::<u32>().copied(), Ok(42));
    }
    #[test]
    fn require_missing_names_type() {
        let e = ToolContext::new().require::<u32>().unwrap_err();
        assert!(e.to_string().contains("u32"));
    }
    #[test]
    fn result_metadata_round_trips_and_requires() {
        #[derive(Clone, Debug, PartialEq)]
        struct Id(u32);
        let mut c = ToolContext::new();
        c.insert_result(Id(7));
        assert_eq!(c.result::<Id>(), Some(&Id(7)));
        assert_eq!(c.require_result::<Id>(), Ok(&Id(7)));
        assert!(c.get::<Id>().is_none());
    }
    #[test]
    fn debug_reports_types_without_values() {
        #[derive(Clone)]
        struct Secret(&'static str);
        let mut c = ToolContext::new();
        c.insert(42u32);
        c.insert_result(Secret("do-not-print"));
        let d = format!("{c:?}");
        assert!(d.contains("u32"));
        assert!(d.contains("Secret"));
        assert!(!d.contains("do-not-print"));
        assert_eq!(c.result::<Secret>().map(|s| s.0), Some("do-not-print"));
    }
    #[test]
    fn dispatch_snapshot_isolates_inbound_and_publishes_only_result_metadata() {
        let mut c = ToolContext::new();
        c.insert(7u32);
        c.insert_result("old".to_string());
        let mut d = c.for_dispatch();
        assert_eq!(d.get::<u32>(), Some(&7));
        assert!(d.result::<String>().is_none());
        *d.get_mut::<u32>().expect("snapshot value") = 8;
        d.insert_result("new".to_string());

        c.accept_dispatch_result(d);
        assert_eq!(c.get::<u32>(), Some(&7));
        assert_eq!(c.result::<String>().map(String::as_str), Some("new"));
    }
    #[test]
    fn many_distinct_types_round_trip_through_type_id_hasher() {
        #[derive(Clone, PartialEq, Debug)]
        struct A(u8);
        #[derive(Clone, PartialEq, Debug)]
        struct B(u16);
        let mut c = ToolContext::new();
        c.insert(A(1));
        c.insert(B(2));
        c.insert(3u32);
        c.insert("four".to_string());
        assert_eq!(c.get::<A>(), Some(&A(1)));
        assert_eq!(c.get::<B>(), Some(&B(2)));
        assert_eq!(c.get::<u32>(), Some(&3));
        assert_eq!(c.get::<String>().map(String::as_str), Some("four"));
    }
}

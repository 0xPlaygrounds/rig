//! Typed private context and result metadata for tool execution.
//!
//! [`ToolContext`] is the only metadata channel in the tool API. Callers place
//! private typed values in its inbound context map before dispatch; tools read
//! those values and can attach typed result metadata to the same context. Neither
//! map is rendered to the model.

use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

type AnyMap = HashMap<TypeId, Box<dyn AnyClone>, BuildHasherDefault<IdHasher>>;

/// `TypeId` is already a hash, so avoid hashing it a second time.
#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    fn write_u64(&mut self, id: u64) {
        self.0 = id;
    }

    fn write(&mut self, bytes: &[u8]) {
        // Keep the hasher correct if `TypeId` ever changes how it writes itself.
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

/// Allocation-free when empty, which is the common tool-call path.
#[derive(Default, Clone)]
struct TypeMap {
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
            .and_then(|old| old.into_any().downcast::<T>().ok())
            .map(|old| *old)
    }

    fn get<T>(&self) -> Option<&T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .as_ref()?
            .get(&TypeId::of::<T>())
            .and_then(|value| (**value).as_any().downcast_ref::<T>())
    }

    fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .as_mut()?
            .get_mut(&TypeId::of::<T>())
            .and_then(|value| (**value).as_any_mut().downcast_mut::<T>())
    }

    fn remove<T>(&mut self) -> Option<T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .as_mut()?
            .remove(&TypeId::of::<T>())
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

/// A required typed value was absent from a [`ToolContext`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("required tool context value of type `{0}` was not found")]
pub struct MissingContext(pub &'static str);

/// Private typed state for one tool invocation.
///
/// A context has two deliberately separate namespaces:
///
/// - callers configure inbound values with [`insert`](Self::insert), which tools
///   read with [`get`](Self::get) or [`require`](Self::require);
/// - tools attach outbound values with [`insert_metadata`](Self::insert_metadata),
///   which hooks and callers inspect on the resulting
///   [`ToolExecution`](crate::tool::ToolExecution).
///
/// Values are cloned when the context is cloned for parallel tool calls. They
/// must therefore be `Clone` and use Rig's WASM-compatible send/sync bounds.
/// Neither namespace is included in model-visible output.
#[derive(Default, Clone)]
pub struct ToolContext {
    input: TypeMap,
    metadata: TypeMap,
}

impl ToolContext {
    /// Create an empty context without allocating.
    pub const fn new() -> Self {
        Self {
            input: TypeMap::EMPTY,
            metadata: TypeMap::EMPTY,
        }
    }

    /// Insert an inbound typed value, returning the displaced value if present.
    pub fn insert<T>(&mut self, value: T) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.input.insert(value)
    }

    /// Read an inbound typed value.
    pub fn get<T>(&self) -> Option<&T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.input.get::<T>()
    }

    /// Mutably access an inbound typed value.
    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.input.get_mut::<T>()
    }

    /// Require an inbound typed value, reporting its Rust type when absent.
    pub fn require<T>(&self) -> Result<&T, MissingContext>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.get::<T>().ok_or(MissingContext(type_name::<T>()))
    }

    /// Remove an inbound typed value.
    pub fn remove<T>(&mut self) -> Option<T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.input.remove::<T>()
    }

    /// Whether an inbound value of type `T` is present.
    pub fn contains<T>(&self) -> bool
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.input.contains::<T>()
    }

    /// Number of inbound values.
    pub fn len(&self) -> usize {
        self.input.len()
    }

    /// Whether there are no inbound values.
    pub fn is_empty(&self) -> bool {
        self.input.len() == 0
    }

    /// Attach typed metadata to the execution result.
    pub fn insert_metadata<T>(&mut self, value: T) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.metadata.insert(value)
    }

    /// Read metadata already attached to this execution.
    pub fn metadata<T>(&self) -> Option<&T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.metadata.get::<T>()
    }

    /// Mutably access metadata already attached to this execution.
    pub fn metadata_mut<T>(&mut self) -> Option<&mut T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.metadata.get_mut::<T>()
    }

    /// Remove attached metadata.
    pub fn remove_metadata<T>(&mut self) -> Option<T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.metadata.remove::<T>()
    }

    /// Whether metadata of type `T` is attached.
    pub fn contains_metadata<T>(&self) -> bool
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.metadata.contains::<T>()
    }

    /// Number of attached metadata values.
    pub fn metadata_len(&self) -> usize {
        self.metadata.len()
    }

    /// Whether no result metadata is attached.
    pub fn metadata_is_empty(&self) -> bool {
        self.metadata.len() == 0
    }
}

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("input_entries", &self.input.len())
            .field("input_types", &self.input.type_names())
            .field("metadata_entries", &self.metadata.len())
            .field("metadata_types", &self.metadata.type_names())
            .finish()
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
    fn inbound_values_round_trip_by_type() {
        let mut context = ToolContext::new();
        assert!(context.is_empty());
        assert_eq!(context.insert(42_u32), None);
        assert_eq!(context.insert("hello".to_string()), None);
        assert_eq!(context.get::<u32>(), Some(&42));
        assert_eq!(context.require::<String>().map(String::as_str), Ok("hello"));
        assert!(context.contains::<u32>());
        assert_eq!(context.len(), 2);
    }

    #[test]
    fn replacement_mutation_and_removal_work() {
        let mut context = ToolContext::new();
        context.insert(vec![1_u8, 2]);
        context.get_mut::<Vec<u8>>().unwrap().push(3);
        assert_eq!(context.insert(vec![4_u8]), Some(vec![1, 2, 3]));
        assert_eq!(context.remove::<Vec<u8>>(), Some(vec![4]));
        assert!(!context.contains::<Vec<u8>>());
    }

    #[test]
    fn clone_deep_copies_values() {
        let mut context = ToolContext::new();
        context.insert(vec![1_u8, 2, 3]);
        let mut cloned = context.clone();
        cloned.get_mut::<Vec<u8>>().unwrap().push(4);
        assert_eq!(context.get::<Vec<u8>>(), Some(&vec![1, 2, 3]));
        assert_eq!(cloned.get::<Vec<u8>>(), Some(&vec![1, 2, 3, 4]));
    }

    #[test]
    fn missing_required_value_names_type() {
        let error = ToolContext::new().require::<u32>().unwrap_err();
        assert!(error.to_string().contains("u32"));
    }

    #[test]
    fn inbound_and_result_metadata_are_separate() {
        let mut context = ToolContext::new();
        context.insert(42_u32);
        context.insert_metadata(7_u32);
        assert_eq!(context.get::<u32>(), Some(&42));
        assert_eq!(context.metadata::<u32>(), Some(&7));
        assert_eq!(context.len(), 1);
        assert_eq!(context.metadata_len(), 1);

        *context.metadata_mut::<u32>().unwrap() = 8;
        assert_eq!(context.remove_metadata::<u32>(), Some(8));
        assert!(context.metadata_is_empty());
    }

    #[test]
    fn empty_context_allocates_no_maps() {
        let context = ToolContext::new();
        assert!(context.input.map.is_none());
        assert!(context.metadata.map.is_none());
    }

    #[test]
    fn debug_reports_namespaces_and_types() {
        let mut context = ToolContext::new();
        context.insert(42_u32);
        context.insert_metadata("request-id".to_string());
        let debug = format!("{context:?}");
        assert!(debug.contains("input_entries: 1"));
        assert!(debug.contains("metadata_entries: 1"));
        assert!(debug.contains("u32"));
        assert!(debug.contains("String"));
    }

    #[test]
    fn heterogeneous_values_survive_custom_hasher() {
        #[derive(Clone, Debug, PartialEq)]
        struct A(u8);
        #[derive(Clone, Debug, PartialEq)]
        struct B(u16);
        #[derive(Clone, Debug, PartialEq)]
        struct C(u32);

        let mut context = ToolContext::new();
        context.insert(A(1));
        context.insert(B(2));
        context.insert(C(3));
        context.insert(4_u64);
        context.insert("five".to_string());

        assert_eq!(context.get::<A>(), Some(&A(1)));
        assert_eq!(context.get::<B>(), Some(&B(2)));
        assert_eq!(context.get::<C>(), Some(&C(3)));
        assert_eq!(context.get::<u64>(), Some(&4));
        assert_eq!(context.get::<String>().map(String::as_str), Some("five"));
    }
}

//! Typed per-call context passed through tool execution.
//!
//! A runtime hands every tool call a [`ToolContext`]: a bag of typed inbound
//! values (auth tokens, session ids, request metadata the model never sees)
//! plus a typed result map a tool can publish host-only data into. rig-agent's
//! contextual tools, context-aware [`PortableDynamicTool`](super::PortableDynamicTool)s,
//! and companion adapters (e.g. MCP `_meta` passthrough in `rig-rmcp`) all
//! share this one type, so the same values flow regardless of runtime.

use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use super::ToolExecutionError;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

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
/// A clone-on-dispatch map of values keyed by type, the storage behind
/// [`ToolContext`]. Public so runtimes can build related per-call state on the
/// same primitive (rig-agent's hook state does).
#[derive(Default, Clone)]
pub struct TypeMap {
    map: Option<Box<AnyMap>>,
}

impl TypeMap {
    pub(crate) const EMPTY: Self = Self { map: None };

    pub fn insert<T>(&mut self, value: T) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .get_or_insert_with(Default::default)
            .insert(TypeId::of::<T>(), Box::new(value))
            .and_then(|previous| previous.into_any().downcast::<T>().ok())
            .map(|value| *value)
    }

    pub fn get<T>(&self) -> Option<&T>
    where
        T: 'static,
    {
        self.map
            .as_ref()
            .and_then(|map| map.get(&TypeId::of::<T>()))
            .and_then(|value| (**value).as_any().downcast_ref::<T>())
    }

    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: 'static,
    {
        self.map
            .as_mut()
            .and_then(|map| map.get_mut(&TypeId::of::<T>()))
            .and_then(|value| (**value).as_any_mut().downcast_mut::<T>())
    }

    pub fn remove<T>(&mut self) -> Option<T>
    where
        T: 'static,
    {
        self.map
            .as_mut()
            .and_then(|map| map.remove(&TypeId::of::<T>()))
            .and_then(|value| value.into_any().downcast::<T>().ok())
            .map(|value| *value)
    }

    pub fn contains<T>(&self) -> bool
    where
        T: 'static,
    {
        self.map
            .as_ref()
            .is_some_and(|map| map.contains_key(&TypeId::of::<T>()))
    }

    /// Number of values held.
    pub fn len(&self) -> usize {
        self.map.as_ref().map_or(0, |map| map.len())
    }

    /// Whether the map holds no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
/// Map-level mutations (inserting, replacing, or removing typed slots) affect
/// only that execution. Value-level isolation follows each value's [`Clone`]
/// semantics, so intentionally shared values such as `Arc<Mutex<_>>` continue
/// to share their referent across dispatches. The dispatch surface returns
/// result metadata without replacing the caller's inbound slots.
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
        T: 'static,
    {
        self.inbound.get::<T>()
    }

    /// Require an inbound typed value.
    pub fn require<T>(&self) -> Result<&T, MissingToolContext>
    where
        T: 'static,
    {
        self.get::<T>().ok_or(MissingToolContext(type_name::<T>()))
    }

    /// Mutably access an inbound typed value.
    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: 'static,
    {
        self.inbound.get_mut::<T>()
    }

    /// Remove an inbound typed value.
    pub fn remove<T>(&mut self) -> Option<T>
    where
        T: 'static,
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
        T: 'static,
    {
        self.result.get::<T>()
    }

    /// Require host-only result metadata.
    pub fn require_result<T>(&self) -> Result<&T, MissingToolContext>
    where
        T: 'static,
    {
        self.result::<T>()
            .ok_or(MissingToolContext(type_name::<T>()))
    }

    /// Whether this context contains the inbound type `T`.
    pub fn contains<T>(&self) -> bool
    where
        T: 'static,
    {
        self.inbound.contains::<T>()
    }

    /// Build a fresh execution context with the same inbound values and no
    /// result metadata.
    ///
    /// Dispatch always runs against this snapshot. Mutating its typed slots does
    /// not change the run-wide or caller-owned map. Values with shared/interior
    /// state remain shared according to their [`Clone`] implementation.
    /// Build the snapshot one dispatch runs against. Runtimes call this per
    /// tool call so map-level mutations inside the call cannot leak into the
    /// caller's context.
    pub fn for_dispatch(&self) -> Self {
        Self {
            inbound: self.inbound.clone(),
            result: TypeMap::EMPTY,
        }
    }

    /// Publish metadata produced by one dispatch while preserving the caller's
    /// inbound values.
    /// Publish the result metadata a dispatch produced (see
    /// [`Self::for_dispatch`]) while keeping the caller's inbound values.
    pub fn accept_dispatch_result(&mut self, dispatched: Self) {
        self.result = dispatched.result;
    }

    /// Clear metadata from the previous dispatch before starting another one.
    /// Drop the previous dispatch's result metadata before starting another.
    pub fn clear_dispatch_result(&mut self) {
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
mod tests;

#[cfg(test)]
mod migrated_tests;

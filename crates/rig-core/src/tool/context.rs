//! Typed per-call context passed through tool execution.
//!
//! A runtime hands every tool call a [`ToolContext`]: a serde map of typed
//! inbound values (auth tokens, session ids, request metadata the model never
//! sees) plus a serde result map a tool can publish host-only data into. rig-agent's
//! contextual tools, context-aware [`PortableDynamicTool`](super::PortableDynamicTool)s,
//! and companion adapters (e.g. MCP `_meta` passthrough in `rig-rmcp`) all
//! share this one type, so the same values flow regardless of runtime.

use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

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
}

impl Clone for Box<dyn AnyClone> {
    fn clone(&self) -> Self {
        (**self).clone_box()
    }
}

/// A clone-on-dispatch map of live values keyed by type. Not the storage
/// behind [`ToolContext`] (which is serde data); runtimes use it for
/// in-process per-run state that never crosses a wire (rig-agent's hook state
/// does).
#[derive(Default, Clone)]
pub struct TypeMap {
    map: Option<Box<AnyMap>>,
}

impl TypeMap {
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
}

/// Context passed to every tool execution.
///
/// Callers insert typed inbound values with [`insert`](Self::insert). Tools read
/// those values with [`get`](Self::get) or [`require`](Self::require), and attach
/// host-only result metadata with [`insert_result`](Self::insert_result). Result
/// hooks inspect that metadata through [`result`](Self::result). Neither inbound
/// values nor result metadata are sent to the model.
///
/// Every value is **data**: inserting serializes it, reading deserializes it,
/// and the whole context is itself `Serialize + Deserialize`, so it crosses
/// tasks, channels, scenes, and record/replay logs unchanged. Values with
/// interior sharing (`Arc<Mutex<_>>`, atomics, channels) do not belong here;
/// they belong to the tool instance.
///
/// Registry, server, and agent dispatch clone inbound values once per call.
/// Map-level mutations (inserting, replacing, or removing typed slots) affect
/// only that execution. The dispatch surface returns result metadata without
/// replacing the caller's inbound slots.
///
/// Slots are keyed by the value's type name, so the accessors stay keyless:
/// one value per type per map.
#[derive(Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolContext {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    inbound: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    result: BTreeMap<String, serde_json::Value>,
}

/// A value that may be stored in a [`ToolContext`]: serde data under a key
/// the type declares. The key is what survives a refactor, a persisted
/// [`EffectLog`](crate::effect::EffectLog), or a different toolchain —
/// unlike `std::any::type_name`, which changes with a rename or a module
/// move and is not stable across compiler versions.
///
/// Derive it (`#[derive(ContextValue)]`, key defaults to the type's name;
/// `#[context(key = "…")]` overrides) or write the one-line impl. Two
/// value types must not share a key.
#[diagnostic::on_unimplemented(
    message = "`{Self}` declares no `ToolContext` key",
    label = "not a `ContextValue`",
    note = "derive it (`#[derive(rig::ContextValue)]`, optionally `#[context(key = \"…\")]`) or write `impl ContextValue for {Self} {{ const KEY: &'static str = \"…\"; }}`; a bare `String`, integer or `serde_json::Value` cannot be stored — wrap it in a newtype"
)]
pub trait ContextValue: Serialize + DeserializeOwned + 'static {
    /// The slot this value lives under.
    const KEY: &'static str;
}

fn encode<T: ContextValue>(value: &T) -> Result<serde_json::Value, ToolContextError> {
    serde_json::to_value(value).map_err(|error| ToolContextError::Encode {
        key: T::KEY,
        message: error.to_string(),
    })
}

fn decode<T: ContextValue>(value: &serde_json::Value) -> Result<T, ToolContextError> {
    serde_json::from_value(value.clone()).map_err(|error| ToolContextError::Decode {
        key: T::KEY,
        message: error.to_string(),
    })
}

/// Store `value`; the value it displaced when that decodes as `T`. A slot
/// holding something that does not decode as `T` is simply replaced: the
/// write succeeded, and a shape mismatch is a defect at the writer, not a
/// reason to unwind the caller after the fact.
fn insert_slot<T: ContextValue>(
    map: &mut BTreeMap<String, serde_json::Value>,
    value: T,
) -> Result<Option<T>, ToolContextError> {
    let encoded = encode(&value)?;
    Ok(map
        .insert(T::KEY.to_owned(), encoded)
        .and_then(|previous| decode(&previous).ok()))
}

/// `Ok(None)` when the slot is empty, `Err(Decode)` when it holds something
/// that is not a `T`: absence and a shape mismatch are different facts.
fn get_slot<T: ContextValue>(
    map: &BTreeMap<String, serde_json::Value>,
) -> Result<Option<T>, ToolContextError> {
    map.get(T::KEY).map(decode).transpose()
}

fn require_slot<T: ContextValue>(
    map: &BTreeMap<String, serde_json::Value>,
) -> Result<T, ToolContextError> {
    get_slot(map)?.ok_or(ToolContextError::Missing(T::KEY))
}

impl ToolContext {
    /// Create an empty context.
    pub const fn new() -> Self {
        Self {
            inbound: BTreeMap::new(),
            result: BTreeMap::new(),
        }
    }

    /// Insert an inbound typed value, returning the displaced value if present.
    ///
    /// Fails only when `value` cannot be represented as JSON (a map with
    /// non-string keys, a float `NaN`).
    pub fn insert<T: ContextValue>(&mut self, value: T) -> Result<Option<T>, ToolContextError> {
        insert_slot(&mut self.inbound, value)
    }

    /// Read an inbound typed value. `None` when absent or not decodable as `T`.
    pub fn get<T: ContextValue>(&self) -> Result<Option<T>, ToolContextError> {
        get_slot(&self.inbound)
    }

    /// Require an inbound typed value.
    pub fn require<T: ContextValue>(&self) -> Result<T, ToolContextError> {
        require_slot(&self.inbound)
    }

    /// Remove an inbound typed value.
    pub fn remove<T: ContextValue>(&mut self) -> Result<Option<T>, ToolContextError> {
        self.inbound
            .remove(T::KEY)
            .map(|value| decode(&value))
            .transpose()
    }

    /// Attach host-only metadata to this execution's result.
    pub fn insert_result<T: ContextValue>(
        &mut self,
        value: T,
    ) -> Result<Option<T>, ToolContextError> {
        insert_slot(&mut self.result, value)
    }

    /// Read host-only result metadata.
    pub fn result<T: ContextValue>(&self) -> Result<Option<T>, ToolContextError> {
        get_slot(&self.result)
    }

    /// Require host-only result metadata.
    pub fn require_result<T: ContextValue>(&self) -> Result<T, ToolContextError> {
        require_slot(&self.result)
    }

    /// Whether this context contains the inbound type `T`.
    pub fn contains<T: ContextValue>(&self) -> bool {
        self.inbound.contains_key(T::KEY)
    }

    /// Whether both maps are empty.
    pub fn is_empty(&self) -> bool {
        self.inbound.is_empty() && self.result.is_empty()
    }

    /// Build the snapshot one dispatch runs against: the same inbound values
    /// and no result metadata. Runtimes call this per tool call so map-level
    /// mutations inside the call cannot leak into the caller's context.
    pub fn for_dispatch(&self) -> Self {
        Self {
            inbound: self.inbound.clone(),
            result: BTreeMap::new(),
        }
    }

    /// Publish the result metadata a dispatch produced (see
    /// [`Self::for_dispatch`]) while keeping the caller's inbound values.
    pub fn accept_dispatch_result(&mut self, dispatched: Self) {
        self.result = dispatched.result;
    }

    /// Drop the previous dispatch's result metadata before starting another.
    pub fn clear_dispatch_result(&mut self) {
        self.result.clear();
    }
}

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("inbound_types", &self.inbound.keys().collect::<Vec<_>>())
            .field("result_types", &self.result.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// A [`ToolContext`] slot could not be read or written.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolContextError {
    /// A required typed value was absent.
    #[error("required tool context value `{0}` was not found")]
    Missing(&'static str),
    /// A value could not be represented as JSON.
    #[error("tool context value `{key}` could not be encoded: {message}")]
    Encode {
        /// The slot's type name.
        key: &'static str,
        /// The serializer's message.
        message: String,
    },
    /// A stored value could not be decoded as the requested type.
    #[error("tool context value `{key}` could not be decoded: {message}")]
    Decode {
        /// The slot's type name.
        key: &'static str,
        /// The deserializer's message.
        message: String,
    },
}

impl From<ToolContextError> for ToolExecutionError {
    fn from(error: ToolContextError) -> Self {
        ToolExecutionError::other(error.to_string()).with_source(error)
    }
}

// The context crosses the effect wire: it must serialize and cross threads on
// every target, browser wasm included.
const _: fn() = || {
    fn assert_wire<T: Send + Sync + 'static + Serialize + DeserializeOwned>() {}
    assert_wire::<ToolContext>();
};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod migrated_tests;

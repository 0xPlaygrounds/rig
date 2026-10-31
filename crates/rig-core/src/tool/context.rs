//! Typed inbound context and explicitly published host-only tool-result metadata.
//!
//! ```
//! use rig_core::tool::{ContextValue, ToolContext};
//! #[derive(serde::Serialize, serde::Deserialize)]
//! struct Session(String);
//! impl ContextValue for Session { const KEY: &'static str = "session"; }
//! let mut context = ToolContext::new();
//! context.insert(Session("example".into()))?;
//! assert_eq!(context.require::<Session>()?.0, "example");
//! # Ok::<(), rig_core::tool::ToolContextError>(())
//! ```

use std::any::Any;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use super::ToolExecutionError;

/// Context passed to every tool execution.
///
/// Callers insert typed inbound values with [`insert`](Self::insert). Tools read
/// those values with [`get`](Self::get) or [`require`](Self::require), and attach
/// host-only result metadata with [`insert_result`](Self::insert_result). Result
/// hooks inspect that metadata through [`result`](Self::result). Neither inbound
/// values nor result metadata are sent to the model.
///
/// Every value is **data**: inserting serializes it, reading deserializes it,
/// and the whole context is itself `Serialize + Deserialize`. Explicit scene
/// serialization can therefore contain sensitive inbound values and must be
/// protected by the host. Effect logs record only [`ToolResultContext`]:
/// values explicitly published through `insert_result`, never inbound slots
/// or runtime scopes. Replay uses the current dispatch's inbound context;
/// code must not treat mutations to inbound slots as durable output. Values with
/// interior sharing (`Arc<Mutex<_>>`, atomics, channels) do not belong here;
/// they belong to the tool instance.
///
/// Registry, server, and agent dispatch clone inbound values once per call.
/// Map-level mutations (inserting, replacing, or removing typed slots) affect
/// only that execution. The dispatch surface returns result metadata without
/// replacing the caller's inbound slots.
///
/// Slots use the value type's declared [`ContextValue::KEY`]. Accessors need
/// no key argument; types must use distinct keys to occupy distinct slots.
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct ToolContext {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    inbound: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    result: BTreeMap<String, serde_json::Value>,
    /// Live runtime scopes, excluded from serialization and equality.
    /// Publication clears them before the result is resolved.
    #[serde(skip)]
    scopes: Vec<std::sync::Arc<dyn Any + Send + Sync>>,
}

/// Explicitly published, durable tool-result metadata. Unlike [`ToolContext`],
/// this contains no inbound credentials or live runtime scopes. Values are
/// keyed by [`ContextValue::KEY`]; applications own their schema versions.
#[derive(Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolResultContext(BTreeMap<String, serde_json::Value>);

impl std::fmt::Debug for ToolResultContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ToolResultContext")
            .field(&self.0.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// The most bytes of JSON a call's published values may take together to
/// be kept with a conversation ([`ToolResultContext::kept`]): room for the
/// diff of a large edit, while a value past it, such as the diff of a
/// rewritten generated file, is left out rather than grow the session by
/// as much.
pub const MAX_KEPT_FACTS_BYTES: usize = 256 * 1024;

impl ToolResultContext {
    /// Read a published value without exposing ambient execution inputs.
    pub fn get<T: ContextValue>(&self) -> Result<Option<T>, ToolContextError> {
        get_slot(&self.0)
    }

    /// Whether nothing was published.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// These values as a conversation keeps them with the call's result
    /// ([`ResultFacts`](crate::conversation::ResultFacts)), for a view to
    /// draw the call from after a resume: every secret in their strings
    /// redacted ([`redact::json`](crate::redact::json)), the whole old
    /// text of each changed file ([`FileChange::before_text`]) left out,
    /// since the hunks show the change, and, past [`MAX_KEPT_FACTS_BYTES`]
    /// of JSON, the largest values left out, so a small one such as an
    /// exit status stays.
    ///
    /// ```
    /// use rig_core::tool::{ExitStatus, FileChange, FilesChanged, ToolContext};
    ///
    /// let mut context = ToolContext::new();
    /// context.insert_result(ExitStatus::Exited(0))?;
    /// let change = FileChange::modified("a.rs", 1, 1).with_before_text("old\n");
    /// context.insert_result(FilesChanged(vec![change]))?;
    /// let kept = context.result_context().kept();
    /// assert_eq!(kept.get::<ExitStatus>()?, Some(ExitStatus::Exited(0)));
    /// let changed = kept.get::<FilesChanged>()?.unwrap_or_default();
    /// assert_eq!(changed.0.first().map(|change| change.before_text.is_none()), Some(true));
    /// # Ok::<(), rig_core::tool::ToolContextError>(())
    /// ```
    ///
    /// [`FileChange::before_text`]: super::FileChange::before_text
    #[must_use]
    pub fn kept(&self) -> Self {
        let mut kept = self.0.clone();
        if let Ok(Some(mut changed)) = get_slot::<super::FilesChanged>(&kept) {
            for change in &mut changed.0 {
                change.before_text = None;
            }
            // A value that decoded encodes again; one that did not stays.
            insert_slot(&mut kept, changed).ok();
        }
        let mut sized: Vec<(usize, String, serde_json::Value)> = kept
            .into_iter()
            .filter_map(|(key, mut value)| {
                crate::redact::json(&mut value);
                let size = serde_json::to_vec(&value)
                    .ok()?
                    .len()
                    .saturating_add(key.len());
                Some((size, key, value))
            })
            .collect();
        sized.sort_by_key(|(size, ..)| *size);
        let (mut room, mut fits) = (MAX_KEPT_FACTS_BYTES, BTreeMap::new());
        for (size, key, value) in sized {
            if size <= room {
                room = room.saturating_sub(size);
                fits.insert(key, value);
            }
        }
        Self(fits)
    }
}

impl PartialEq for ToolContext {
    fn eq(&self, other: &Self) -> bool {
        self.inbound == other.inbound && self.result == other.result
    }
}

impl Eq for ToolContext {}

/// Serializable context data with a stable slot key. Distinct value types must
/// use distinct keys, and persisted keys must remain stable across schema changes.
/// The derive macro defaults to the type name; `#[context(key = "…")]` overrides it.
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
        source: error,
    })
}

fn decode<T: ContextValue>(value: &serde_json::Value) -> Result<T, ToolContextError> {
    T::deserialize(value).map_err(|error| ToolContextError::Decode {
        key: T::KEY,
        source: error,
    })
}

/// Replaces a slot after successful encoding, returning the displaced value
/// only if it decodes as `T`.
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

impl ToolContext {
    /// Create an empty context.
    pub const fn new() -> Self {
        Self {
            inbound: BTreeMap::new(),
            result: BTreeMap::new(),
            scopes: Vec::new(),
        }
    }

    /// Insert an inbound typed value, returning the displaced value if it decodes
    /// as `T`. A differently shaped displaced value is replaced successfully and
    /// returns `None`; decoding the previous value cannot undo a successful write.
    ///
    /// Returns an encoding error if serialization fails, leaving the slot unchanged.
    pub fn insert<T: ContextValue>(&mut self, value: T) -> Result<Option<T>, ToolContextError> {
        insert_slot(&mut self.inbound, value)
    }

    /// Read an inbound typed value: `Ok(None)` when absent, a decode error when
    /// the stored value does not match `T`.
    pub fn get<T: ContextValue>(&self) -> Result<Option<T>, ToolContextError> {
        get_slot(&self.inbound)
    }

    /// Require an inbound typed value.
    pub fn require<T: ContextValue>(&self) -> Result<T, ToolContextError> {
        get_slot(&self.inbound)?.ok_or(ToolContextError::Missing(T::KEY))
    }

    /// Remove an inbound typed value. The slot is removed even if decoding its
    /// former value fails.
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
            scopes: self.scopes.clone(),
        }
    }

    /// Attach one of the driver's scopes for the call (see the field).
    pub fn with_scope(mut self, scope: std::sync::Arc<dyn Any + Send + Sync>) -> Self {
        self.scopes.push(scope);
        self
    }

    /// Attach the driver's scopes for the call, as the sink carries them.
    pub fn with_scopes(mut self, scopes: Vec<std::sync::Arc<dyn Any + Send + Sync>>) -> Self {
        self.scopes.extend(scopes);
        self
    }

    /// Returns the first attached scope of type `T`, or `None` if absent.
    pub fn scope<T: Any + Send + Sync>(&self) -> Option<std::sync::Arc<T>> {
        self.scopes
            .iter()
            .find_map(|scope| std::sync::Arc::downcast::<T>(scope.clone()).ok())
    }

    /// Drop the scopes: the call is over and the context is data again.
    pub fn clear_scope(&mut self) {
        self.scopes.clear();
    }

    /// Publish the result metadata a dispatch produced (see
    /// [`Self::for_dispatch`]) while keeping the caller's inbound values.
    pub fn accept_dispatch_result(&mut self, dispatched: Self) {
        self.result = dispatched.result;
    }

    /// Snapshot only the values explicitly published with `insert_result`.
    pub fn result_context(&self) -> ToolResultContext {
        ToolResultContext(self.result.clone())
    }

    /// The values explicitly published with `insert_result`, taken.
    pub fn into_result_context(self) -> ToolResultContext {
        ToolResultContext(self.result)
    }

    /// Restore durable output while preserving this dispatch's inbound values.
    pub fn with_result_context(mut self, result: ToolResultContext) -> Self {
        self.result = result.0;
        self
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

/// What one dispatch published beside its answer, passed back beside the
/// sink rather than on the wire. A tool's adapter fills it with the tool's
/// whole context (a second publish replaces the first); any other layer,
/// such as a stream rewrite
/// ([`ErasedHandler::map_stream`](crate::serve::ErasedHandler::map_stream)),
/// adds single facts with [`insert`](Self::insert).
#[derive(Debug, Default)]
pub struct PublishedContext(std::sync::Mutex<Option<ToolContext>>);

impl PublishedContext {
    /// An empty one, shared between the driver and the sink.
    pub fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self::default())
    }

    /// The recorder's non-consuming snapshot. Publish before resolving the
    /// sink; values published after resolution are not part of that outcome.
    pub fn result_context(&self) -> Option<ToolResultContext> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map(ToolContext::result_context)
    }

    /// The adapter's write: the context after the tool ran, scopes cleared.
    /// All publication must finish before resolving the sink. Publishing
    /// afterward violates the delivery contract: consumers and recorders may
    /// observe different versions. The slot does not enforce this ordering.
    pub fn publish(&self, mut context: ToolContext) {
        context.clear_scope();
        *self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(context);
    }

    /// Adds `value` to what is published, beside what is there, as
    /// [`ToolContext::insert_result`] does, and returns the value it
    /// replaced. Publish before the dispatch's answer is resolved, as for
    /// [`publish`](Self::publish).
    ///
    /// # Errors
    ///
    /// When `value` cannot be encoded; nothing changes then.
    pub fn insert<T: ContextValue>(&self, value: T) -> Result<Option<T>, ToolContextError> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_or_insert_with(ToolContext::new)
            .insert_result(value)
    }

    /// The driver's read: what was published, if anything, leaving nothing.
    pub fn take(&self) -> Option<ToolContext> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }
}

/// A [`ToolContext`] slot could not be read or written.
#[derive(Debug, thiserror::Error)]
pub enum ToolContextError {
    /// A required typed value was absent.
    #[error("required tool context value `{0}` was not found")]
    Missing(&'static str),
    /// A value could not be represented as JSON.
    #[error("tool context value `{key}` could not be encoded: {source}")]
    Encode {
        /// The slot's type name.
        key: &'static str,
        /// The serializer's failure.
        #[source]
        source: serde_json::Error,
    },
    /// A stored value could not be decoded as the requested type.
    #[error("tool context value `{key}` could not be decoded: {source}")]
    Decode {
        /// The slot's type name.
        key: &'static str,
        /// The deserializer's failure.
        #[source]
        source: serde_json::Error,
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
    fn assert_shared<T: Send + Sync + 'static>() {}
    assert_shared::<PublishedContext>();
};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod migrated_tests;

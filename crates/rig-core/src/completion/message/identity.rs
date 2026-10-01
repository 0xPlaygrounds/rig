//! Identities a message carries: a tool call's one [`CallId`], a tool's
//! [`ToolName`], and reasoning [`Sealed`] to the [`Issuer`] that produced it.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// A tool name, never empty.
///
/// ```
/// use rig_core::message::ToolName;
///
/// let name = ToolName::new("add")?;
/// assert_eq!(name.as_str(), "add");
/// assert!(ToolName::new("").is_err());
/// # Ok::<(), rig_core::message::EmptyToolName>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ToolName(String);

/// A tool name was empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("a tool name cannot be empty")]
pub struct EmptyToolName;

impl ToolName {
    /// `name`, or [`EmptyToolName`] when it is empty.
    pub fn new(name: impl Into<String>) -> Result<Self, EmptyToolName> {
        let name = name.into();
        if name.is_empty() {
            Err(EmptyToolName)
        } else {
            Ok(Self(name))
        }
    }

    /// `name` without the emptiness check. Callers prove `name` is not empty,
    /// for example with a compile-time assertion on a `Tool::NAME`.
    pub(crate) fn new_unchecked(name: &str) -> Self {
        debug_assert!(!name.is_empty());
        Self(name.to_owned())
    }

    /// The name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ToolName {
    type Error = EmptyToolName;

    fn try_from(name: String) -> Result<Self, EmptyToolName> {
        Self::new(name)
    }
}

impl TryFrom<&str> for ToolName {
    type Error = EmptyToolName;

    fn try_from(name: &str) -> Result<Self, EmptyToolName> {
        Self::new(name)
    }
}

impl From<ToolName> for String {
    fn from(name: ToolName) -> Self {
        name.0
    }
}

impl std::ops::Deref for ToolName {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for ToolName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::borrow::Borrow<str> for ToolName {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ToolName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl PartialEq<str> for ToolName {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for ToolName {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl PartialEq<String> for ToolName {
    fn eq(&self, other: &String) -> bool {
        &self.0 == other
    }
}

impl PartialEq<ToolName> for String {
    fn eq(&self, other: &ToolName) -> bool {
        *self == other.0
    }
}

impl PartialEq<ToolName> for str {
    fn eq(&self, other: &ToolName) -> bool {
        self == other.0
    }
}

/// A tool call's one identity: the id the provider issued, or one rig issued
/// because the provider sent none.
///
/// A result copies the id of the call it answers ([`ToolCall::result`]), so
/// the two always match.
///
/// ```
/// use rig_core::message::{CallId, ProviderCallId};
///
/// let id = CallId::from(ProviderCallId::new("call_1").ok_or("empty id")?);
/// assert_eq!(id.to_string(), "call_1");
/// assert_eq!(id.provider().map(|provider| provider.call_id.as_str()), Some("call_1"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// [`ToolCall::result`]: super::ToolCall::result
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallId {
    /// The provider's identifiers.
    Provider(ProviderCallId),
    /// An identifier rig issued for a call the provider sent without one.
    Local(LocalCallId),
}

impl CallId {
    /// The provider's id `call_id`, or a rig-issued id when it is empty.
    pub fn from_wire(call_id: impl Into<String>) -> Self {
        ProviderCallId::new(call_id).map_or_else(|| Self::Local(LocalCallId::new()), Self::Provider)
    }

    /// A dual-identifier wire's ids (OpenAI Responses): `call_id` is the
    /// correlator and `item_id` the output item. A rig-issued id when
    /// `call_id` is empty.
    pub fn from_dual_wire(item_id: impl Into<String>, call_id: impl Into<String>) -> Self {
        match ProviderCallId::new(call_id) {
            Some(provider) => Self::Provider(provider.with_item_id(item_id)),
            None => Self::Local(LocalCallId::new()),
        }
    }

    /// The provider's identifiers, when the provider issued them.
    pub fn provider(&self) -> Option<&ProviderCallId> {
        match self {
            Self::Provider(provider) => Some(provider),
            Self::Local(_) => None,
        }
    }

    /// Whether rig issued this id.
    pub fn is_local(&self) -> bool {
        matches!(self, Self::Local(_))
    }

    /// The id as a wire sends it: the provider's call id, or the rig-issued
    /// UUID.
    pub fn wire(&self) -> Cow<'_, str> {
        match self {
            Self::Provider(provider) => Cow::Borrowed(provider.call_id.as_str()),
            Self::Local(local) => Cow::Owned(local.to_string()),
        }
    }
}

impl From<ProviderCallId> for CallId {
    fn from(provider: ProviderCallId) -> Self {
        Self::Provider(provider)
    }
}

impl From<LocalCallId> for CallId {
    fn from(local: LocalCallId) -> Self {
        Self::Local(local)
    }
}

impl std::fmt::Display for CallId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.wire())
    }
}

/// A call id rig issued: a random (v4) UUID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LocalCallId(uuid::Uuid);

impl LocalCallId {
    /// A fresh id.
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl Default for LocalCallId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for LocalCallId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Wire shape for [`ProviderCallId`], so deserialization enforces the
/// non-empty `call_id` invariant.
#[derive(Deserialize)]
struct ProviderCallIdWire {
    call_id: String,
    #[serde(default)]
    item_id: Option<String>,
}

/// Provider-issued identifiers for replay. Single-id protocols use `call_id`;
/// dual-id protocols also use `item_id`. Keep each identifier in its protocol
/// slot.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "ProviderCallIdWire")]
pub struct ProviderCallId {
    /// The call-correlation identifier the provider expects echoed back.
    pub call_id: String,
    /// The output-item id issued alongside `call_id` on dual-identifier
    /// wires (OpenAI Responses `fc_…`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
}

/// A provider call id was empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("a provider call id cannot be empty")]
pub struct EmptyCallId;

impl ProviderCallId {
    /// Adopt a provider-issued call identifier. `None` for the empty string:
    /// absence is not an id.
    pub fn new(call_id: impl Into<String>) -> Option<Self> {
        let call_id = call_id.into();
        (!call_id.is_empty()).then_some(Self {
            call_id,
            item_id: None,
        })
    }

    /// Attach the dual-wire output-item id (empty strings are dropped).
    pub fn with_item_id(mut self, item_id: impl Into<String>) -> Self {
        let item_id = item_id.into();
        self.item_id = (!item_id.is_empty()).then_some(item_id);
        self
    }
}

impl TryFrom<ProviderCallIdWire> for ProviderCallId {
    type Error = EmptyCallId;

    fn try_from(wire: ProviderCallIdWire) -> Result<Self, EmptyCallId> {
        let provider = Self::new(wire.call_id).ok_or(EmptyCallId)?;
        Ok(match wire.item_id {
            Some(item_id) => provider.with_item_id(item_id),
            None => provider,
        })
    }
}

/// The service that issued reasoning: the descriptor name of the wire that
/// decoded it (`"anthropic"`), or the model vendor where a gateway relays
/// several (`"openrouter/openai"`).
///
/// An issuer ending in `/` names a family: it accepts every issuer under it,
/// so `openrouter/` accepts `openrouter/openai`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Issuer(Cow<'static, str>);

impl Issuer {
    /// The issuer named `name`.
    pub fn new(name: impl Into<Cow<'static, str>>) -> Self {
        Self(name.into())
    }

    /// The issuer named `name`, in a const context.
    pub const fn from_static(name: &'static str) -> Self {
        Self(Cow::Borrowed(name))
    }

    /// The issuer's name.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether a value `issued` by that issuer may be opened by this one:
    /// the same issuer, or a family `issued` belongs to.
    pub fn accepts(&self, issued: &Issuer) -> bool {
        self.0 == issued.0 || (self.0.ends_with('/') && issued.0.starts_with(self.as_str()))
    }
}

impl std::fmt::Display for Issuer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&'static str> for Issuer {
    fn from(name: &'static str) -> Self {
        Self(Cow::Borrowed(name))
    }
}

impl From<String> for Issuer {
    fn from(name: String) -> Self {
        Self(Cow::Owned(name))
    }
}

/// A value only its issuer may read: reasoning, whose signatures, encrypted
/// payloads and ids mean something only to the service that produced them.
///
/// A wire encoding a request opens each sealed value with the issuers it
/// replays; a value another service issued does not open, so it is not
/// sent.
///
/// ```
/// use rig_core::message::{Issuer, Reasoning, Sealed};
///
/// let sealed = Sealed::new(Issuer::from("anthropic"), Reasoning::new("thinking"));
/// assert!(sealed.open(&Issuer::from("anthropic")).is_some());
/// assert!(sealed.open(&Issuer::from("openai")).is_none());
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sealed<T> {
    issuer: Issuer,
    #[serde(flatten)]
    value: T,
}

impl<T> Sealed<T> {
    /// `value`, readable only by `issuer`.
    pub fn new(issuer: impl Into<Issuer>, value: T) -> Self {
        Self {
            issuer: issuer.into(),
            value,
        }
    }

    /// Who issued the value.
    pub fn issuer(&self) -> &Issuer {
        &self.issuer
    }

    /// The value, when `to` accepts its issuer ([`Issuer::accepts`]).
    pub fn open(&self, to: &Issuer) -> Option<&T> {
        to.accepts(&self.issuer).then_some(&self.value)
    }

    /// The value, when any of `issuers` accepts its issuer.
    pub fn open_for(&self, issuers: &[Issuer]) -> Option<&T> {
        issuers
            .iter()
            .any(|to| to.accepts(&self.issuer))
            .then_some(&self.value)
    }

    /// The same value, readable only by `issuer`.
    pub(crate) fn reseal(self, issuer: impl Into<Issuer>) -> Self {
        Self::new(issuer, self.value)
    }

    /// The value, for the completion writer that assembles it.
    pub(crate) fn value(&self) -> &T {
        &self.value
    }

    /// The value, unsealed, for the completion writer that reseals it.
    pub(crate) fn into_value(self) -> T {
        self.value
    }

    /// The value, for the completion writer that reseals what it carries.
    pub(crate) fn value_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

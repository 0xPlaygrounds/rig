//! [`Id`]: a name or identifier that is never empty.

use std::fmt;
use std::marker::PhantomData;

/// A non-empty name or identifier of kind `K`.
///
/// Construction and deserialization refuse an empty string, so an unreported
/// value is `None` rather than `""`. Serialized as a plain string.
pub struct Id<K> {
    value: String,
    kind: PhantomData<fn() -> K>,
}

/// A provider's descriptor name (`"openai"`).
pub type ProviderName = Id<kind::Provider>;
/// A model identifier as the provider reported it.
pub type ModelName = Id<kind::Model>;
/// A provider-assigned response identifier. Never replayed as a message id.
pub type ResponseId = Id<kind::Response>;
/// A transport request identifier from reply headers or SDK metadata.
pub type RequestId = Id<kind::Request>;
/// A provider-assigned assistant message identifier, suitable for replay.
pub type MessageId = Id<kind::Message>;

/// What each kind of [`Id`] is called.
pub mod kind {
    /// A kind of [`Id`](super::Id).
    pub trait Kind: 'static {
        /// The kind's name, as an error spells it.
        const NAME: &'static str;
    }

    /// [`ProviderName`](super::ProviderName).
    #[derive(Debug)]
    pub enum Provider {}
    impl Kind for Provider {
        const NAME: &'static str = "provider name";
    }

    /// [`ModelName`](super::ModelName).
    #[derive(Debug)]
    pub enum Model {}
    impl Kind for Model {
        const NAME: &'static str = "model name";
    }

    /// [`ResponseId`](super::ResponseId).
    #[derive(Debug)]
    pub enum Response {}
    impl Kind for Response {
        const NAME: &'static str = "response id";
    }

    /// [`RequestId`](super::RequestId).
    #[derive(Debug)]
    pub enum Request {}
    impl Kind for Request {
        const NAME: &'static str = "request id";
    }

    /// [`MessageId`](super::MessageId).
    #[derive(Debug)]
    pub enum Message {}
    impl Kind for Message {
        const NAME: &'static str = "message id";
    }
}

/// An empty string where a non-empty [`Id`] was required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a {0} must not be empty")]
pub struct EmptyId(&'static str);

impl<K: kind::Kind> Id<K> {
    /// The id, or [`EmptyId`] when `value` is empty.
    pub fn new(value: impl Into<String>) -> Result<Self, EmptyId> {
        let value = value.into();
        if value.is_empty() {
            Err(EmptyId(K::NAME))
        } else {
            Ok(Self {
                value,
                kind: PhantomData,
            })
        }
    }

    /// The id, or `None` when `value` is empty: how a decoder reads an id
    /// the provider may leave blank.
    pub fn non_empty(value: impl Into<String>) -> Option<Self> {
        Self::new(value).ok()
    }

    /// The id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.value
    }

    /// The id as an owned string.
    pub fn into_string(self) -> String {
        self.value
    }
}

impl<K> Clone for Id<K> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            kind: PhantomData,
        }
    }
}

impl<K> PartialEq for Id<K> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<K> Eq for Id<K> {}

impl<K> std::hash::Hash for Id<K> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<K> PartialEq<str> for Id<K> {
    fn eq(&self, other: &str) -> bool {
        self.value == other
    }
}

impl<K> PartialEq<&str> for Id<K> {
    fn eq(&self, other: &&str) -> bool {
        self.value == *other
    }
}

impl<K> AsRef<str> for Id<K> {
    fn as_ref(&self) -> &str {
        &self.value
    }
}

impl<K> std::ops::Deref for Id<K> {
    type Target = str;

    fn deref(&self) -> &str {
        &self.value
    }
}

impl<K> fmt::Display for Id<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.value)
    }
}

impl<K: kind::Kind> fmt::Debug for Id<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({:?})", K::NAME, self.value)
    }
}

impl<K> From<Id<K>> for String {
    fn from(id: Id<K>) -> Self {
        id.value
    }
}

impl<K: kind::Kind> TryFrom<String> for Id<K> {
    type Error = EmptyId;

    fn try_from(value: String) -> Result<Self, EmptyId> {
        Self::new(value)
    }
}

impl<K: kind::Kind> TryFrom<&str> for Id<K> {
    type Error = EmptyId;

    fn try_from(value: &str) -> Result<Self, EmptyId> {
        Self::new(value)
    }
}

impl<K> serde::Serialize for Id<K> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.value)
    }
}

impl<'de, K: kind::Kind> serde::Deserialize<'de> for Id<K> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests;

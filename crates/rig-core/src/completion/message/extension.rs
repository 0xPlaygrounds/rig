//! Typed provider data on message blocks, and the [`Opaque`] item that holds
//! output with no canonical meaning.
//!
//! A provider declares each piece of data it keeps as its own [`Extension`]
//! type and reads it back by type. Core stores the serialized value under
//! the type's key and never interprets it, so a provider adds data without
//! changing core, and a wire that cannot name another provider's type cannot
//! read it.
//!
//! ```
//! use rig_core::message::{Extension, Text};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Debug, PartialEq, Serialize, Deserialize)]
//! struct Phase(String);
//!
//! impl Extension for Phase {
//!     const KEY: &'static str = "example_phase";
//! }
//!
//! let text = Text::new("hello").with_extension(&Phase("final".into()))?;
//! assert_eq!(text.extension::<Phase>()?, Some(Phase("final".into())));
//! # Ok::<(), rig_core::message::ExtensionError>(())
//! ```

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use super::{AdditionalParams, Sealed};

/// Provider data stored on a message block under [`Self::KEY`].
///
/// The value is persisted as its serde form. Only code that names the type
/// reads it, so the type is the access control: keep it private to the wire
/// that owns it when other wires must not read it.
pub trait Extension: Serialize + DeserializeOwned + 'static {
    /// The storage key. Prefix it with the owning wire format
    /// (`"anthropic_"`, `"openai_responses"`), because keys share one map.
    const KEY: &'static str;

    /// Keys earlier releases stored the same value under. Reads fall back to
    /// them in order; writes use [`Self::KEY`] only.
    const LEGACY_KEYS: &'static [&'static str] = &[];
}

/// A stored extension value did not match its type, or a value did not
/// serialize to JSON.
#[derive(Debug, thiserror::Error)]
#[error("extension `{key}`: {source}")]
pub struct ExtensionError {
    /// The key the value is stored under.
    pub key: &'static str,
    /// Why the value did not convert.
    #[source]
    pub source: serde_json::Error,
}

impl AdditionalParams {
    /// The `T` stored here, `None` when absent. A stored value that does not
    /// deserialize as `T` is an error, never silently absent.
    pub fn extension<T: Extension>(&self) -> Result<Option<T>, ExtensionError> {
        let stored = std::iter::once(T::KEY)
            .chain(T::LEGACY_KEYS.iter().copied())
            .find_map(|key| self.0.get(key).map(|value| (key, value)));
        match stored {
            None => Ok(None),
            Some((key, value)) => T::deserialize(value)
                .map(Some)
                .map_err(|source| ExtensionError { key, source }),
        }
    }

    /// Store `value` under `T`'s key, replacing any value there or under a
    /// legacy key.
    pub fn insert_extension<T: Extension>(&mut self, value: &T) -> Result<(), ExtensionError> {
        let value = serde_json::to_value(value).map_err(|source| ExtensionError {
            key: T::KEY,
            source,
        })?;
        for legacy in T::LEGACY_KEYS {
            self.0.remove(*legacy);
        }
        self.0.insert(T::KEY.to_owned(), value);
        Ok(())
    }

    /// These params without `T`, or `None` when nothing else is left,
    /// because empty params are absent.
    pub fn without_extension<T: Extension>(mut self) -> Option<Self> {
        self.0.remove(T::KEY);
        for legacy in T::LEGACY_KEYS {
            self.0.remove(*legacy);
        }
        Self::new(self.0)
    }

    /// Params holding only `value`.
    pub fn of<T: Extension>(value: &T) -> Result<Self, ExtensionError> {
        let value = serde_json::to_value(value).map_err(|source| ExtensionError {
            key: T::KEY,
            source,
        })?;
        let mut map = serde_json::Map::new();
        map.insert(T::KEY.to_owned(), value);
        Ok(Self(map))
    }
}

/// The `T` stored in optional params; see [`AdditionalParams::extension`].
pub fn extension_of<T: Extension>(
    params: Option<&AdditionalParams>,
) -> Result<Option<T>, ExtensionError> {
    params.map_or(Ok(None), AdditionalParams::extension)
}

/// `params` with `value` stored as `T`.
pub fn with_extension<T: Extension>(
    params: Option<AdditionalParams>,
    value: &T,
) -> Result<AdditionalParams, ExtensionError> {
    match params {
        Some(mut params) => {
            params.insert_extension(value)?;
            Ok(params)
        }
        None => AdditionalParams::of(value),
    }
}

/// An assistant item with no canonical meaning: a hosted-tool step, a
/// compaction marker, or an item type Rig does not model. It holds the
/// provider's own [`Extension`] types and keeps its place among its
/// siblings.
///
/// It is always [`Sealed`] to the service that produced it. A wire replays
/// it when a replay issuer opens it and it holds a type that wire owns, and
/// leaves it out otherwise.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Opaque {
    /// The provider data, never empty.
    pub extensions: AdditionalParams,
}

impl Opaque {
    /// An item holding only `value`.
    pub fn of<T: Extension>(value: &T) -> Result<Self, ExtensionError> {
        Ok(Self {
            extensions: AdditionalParams::of(value)?,
        })
    }

    /// The `T` this item holds, `None` when absent.
    pub fn extension<T: Extension>(&self) -> Result<Option<T>, ExtensionError> {
        self.extensions.extension()
    }
}

impl Sealed<Opaque> {
    /// The `T` the item holds, when one of `issuers` opens it.
    pub fn extension_for<T: Extension>(
        &self,
        issuers: &[super::Issuer],
    ) -> Result<Option<T>, ExtensionError> {
        self.open_for(issuers)
            .map_or(Ok(None), |opaque| opaque.extension())
    }
}

/// A stable 64-bit digest of a block's canonical payload (FNV-1a), the same
/// on every build and target.
///
/// An extension that describes its block's payload, such as citations into
/// a text, records the fingerprint of the payload it was decoded with. The
/// owning encoder drops that data when the payload has since been edited.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fingerprint(u64);

impl Fingerprint {
    /// The fingerprint of `payload`.
    pub fn of(payload: &str) -> Self {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        Self(payload.bytes().fold(OFFSET, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(PRIME)
        }))
    }

    /// Whether `payload` is the payload this fingerprint was taken of.
    pub fn matches(&self, payload: &str) -> bool {
        *self == Self::of(payload)
    }
}

// Hex text, because JSON numbers above 2^53 lose precision in many readers.
impl Serialize for Fingerprint {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&format!("{:016x}", self.0))
    }
}

impl<'de> Deserialize<'de> for Fingerprint {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        u64::from_str_radix(&text, 16)
            .map(Self)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests;

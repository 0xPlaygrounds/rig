//! Google's v1beta Gemini REST schema as Rust, generated from the pinned
//! `discovery.json` by `cargo xtask gemini-api`. Every struct keeps the fields
//! it does not type in [`Unmodeled`] and every enum keeps the values it does
//! not know in `Unknown`, so a body read here re-serializes to the same JSON.
//!
//! The settings templates ([`RequestSettings`], [`GenerationSettings`],
//! [`HostedTool`], [`ToolConfigSettings`]) are the request mirrors without the
//! fields rig owns: contents, system instruction, function declarations, tool
//! choice, sampling temperature, output limit and response schema.
//!
//! ```
//! use rig_core::providers::gemini::api;
//!
//! let settings = api::RequestSettings {
//!     generation_config: api::GenerationSettings {
//!         thinking_config: Some(api::ThinkingConfig {
//!             thinking_level: Some(api::ThinkingLevel::Low),
//!             ..Default::default()
//!         }),
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! assert!(api::Unmodeled::<api::GenerationSettings>::new()
//!     .with("thinking_config", serde_json::json!({}))
//!     .is_err());
//! # let _ = settings;
//! ```

use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::message::{Issuer, NativePart, Sealed};

mod generated;

pub use generated::*;

/// The schema a native GenerateContent part records.
pub const PART_SCHEMA: &str = "google.ai.generativelanguage.v1beta.Part";

/// A generated mirror of one Google schema.
pub trait Mirrored {
    /// The mirror's Rust name.
    const NAME: &'static str;
    /// Every wire field Google documents for the schema, including fields a
    /// settings template removed because rig owns them.
    const FIELDS: &'static [&'static str];

    /// Push the path of every unmodeled field in this value, recursively.
    fn unmodeled_fields(&self, path: &str, out: &mut Vec<String>);
}

/// Fields Google sends or accepts that this mirror does not type yet.
pub struct Unmodeled<T> {
    fields: serde_json::Map<String, serde_json::Value>,
    mirror: PhantomData<fn() -> T>,
}

/// A key refused by [`Unmodeled::with`].
#[derive(Debug, thiserror::Error)]
pub enum UnmodeledError {
    /// The key names a documented field of the mirror, in either spelling
    /// Google accepts, typed or owned by rig.
    #[error("`{field}` is a typed field of {mirror}")]
    Modeled {
        /// The mirror the key belongs to.
        mirror: &'static str,
        /// The refused key.
        field: String,
    },
}

impl<T: Mirrored> Unmodeled<T> {
    /// No unmodeled fields.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `key`, refusing any key the mirror documents. Google also accepts
    /// each field's snake_case proto name and rejects a body that spells a
    /// field both ways, so both spellings are refused.
    pub fn with(
        mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Result<Self, UnmodeledError> {
        let key = key.into();
        if T::FIELDS.iter().any(|field| same_field(field, &key)) {
            return Err(UnmodeledError::Modeled {
                mirror: T::NAME,
                field: key,
            });
        }
        self.fields.insert(key, value);
        Ok(self)
    }
}

/// Whether `key` spells `field` as Google reads it: its camelCase JSON name or
/// its snake_case proto name.
fn same_field(field: &str, key: &str) -> bool {
    let fold = |name: &str| -> String {
        name.chars()
            .filter(|c| *c != '_')
            .map(|c| c.to_ascii_lowercase())
            .collect()
    };
    fold(field) == fold(key)
}

impl<T> Unmodeled<T> {
    /// The unmodeled keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.fields.keys().map(String::as_str)
    }

    /// The unmodeled value under `key`.
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.fields.get(key)
    }

    /// Whether every field was typed.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

impl<T> Default for Unmodeled<T> {
    fn default() -> Self {
        Self {
            fields: serde_json::Map::new(),
            mirror: PhantomData,
        }
    }
}

impl<T> Clone for Unmodeled<T> {
    fn clone(&self) -> Self {
        Self {
            fields: self.fields.clone(),
            mirror: PhantomData,
        }
    }
}

impl<T> PartialEq for Unmodeled<T> {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}

impl<T> std::fmt::Debug for Unmodeled<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fields.fmt(f)
    }
}

impl<T> Serialize for Unmodeled<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.fields.serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for Unmodeled<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self {
            fields: serde_json::Map::deserialize(deserializer)?,
            mirror: PhantomData,
        })
    }
}

/// An enum with Google's wire spelling per variant and an `Unknown` arm for
/// values published after generation.
macro_rules! mirror_enum {
    ($name:ident { $($variant:ident => $wire:literal,)* }) => {
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        #[allow(missing_docs)]
        pub enum $name {
            $($variant,)*
            /// A value this mirror does not know yet.
            Unknown(String),
        }

        impl $name {
            /// Google's spelling.
            pub fn as_str(&self) -> &str {
                match self {
                    $(Self::$variant => $wire,)*
                    Self::Unknown(value) => value,
                }
            }
        }

        impl Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(self.as_str())
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                match value.as_str() {
                    $($wire => Self::$variant,)*
                    _ => Self::Unknown(value),
                }
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                String::deserialize(deserializer).map(Self::from)
            }
        }
    };
}

pub(crate) use mirror_enum;

/// A mirror type a whole reply is recognized by: a field every such document
/// carries, which the mirror's all-optional fields cannot require.
pub(crate) trait KeyedBy: serde::de::DeserializeOwned {
    /// The field.
    const KEY: &'static str;
}

/// `T` from a document that carries `T::KEY`. Any other document fails to
/// decode, so an untyped classifier reports it as unknown.
pub(crate) struct Recognized<T>(pub(crate) T);

impl<'de, T: KeyedBy> Deserialize<'de> for Recognized<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let value = serde_json::Value::deserialize(deserializer)?;
        if value.get(T::KEY).is_none() {
            return Err(D::Error::custom(format_args!("no `{}`", T::KEY)));
        }
        T::deserialize(value).map(Self).map_err(D::Error::custom)
    }
}

macro_rules! keyed_by {
    ($($name:ident => $key:literal,)*) => {
        $(impl KeyedBy for $name {
            const KEY: &'static str = $key;
        })*
    };
}

keyed_by! {
    CachedContent => "name",
    ListCachedContentsResponse => "cachedContents",
    File => "name",
    ListFilesResponse => "files",
    Operation => "name",
    ListOperationsResponse => "operations",
}

/// A native part that does not read as a GenerateContent [`Part`].
#[derive(Debug, thiserror::Error)]
pub enum NativeError {
    /// Another service issued the part.
    #[error("native part was issued by `{0}`")]
    OtherIssuer(Issuer),
    /// The part follows another schema.
    #[error("native part follows `{0}`")]
    OtherSchema(String),
    /// The JSON is not a part.
    #[error(transparent)]
    Decode(#[from] serde_json::Error),
}

impl TryFrom<&Sealed<NativePart>> for Part {
    type Error = NativeError;

    fn try_from(native: &Sealed<NativePart>) -> Result<Self, Self::Error> {
        let part = native
            .open(&super::ISSUER)
            .ok_or_else(|| NativeError::OtherIssuer(native.issuer().clone()))?;
        if part.schema != PART_SCHEMA {
            return Err(NativeError::OtherSchema(part.schema.to_string()));
        }
        Ok(serde_json::from_str(part.json())?)
    }
}

/// Proto3 JSON reads `null` as the field's default.
fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

#[cfg(test)]
mod tests;

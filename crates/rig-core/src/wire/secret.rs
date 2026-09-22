//! Redacted credentials for serializable provider configuration.
//!
//! ```
//! use rig_core::wire::Secret;
//!
//! let secret = Secret::from("api-key");
//! assert_eq!(format!("{secret:?}"), "[redacted]");
//! ```

/// Credential whose `Debug` and serialized form contain only `[redacted]`.
/// Equality compares the stored value; [`Self::expose`] returns it unredacted.
/// Deserializing `[redacted]` produces an empty credential. Callers must supply
/// credentials again after reloading serialized configuration.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct Secret(String);

/// Redaction sentinel, deserialized as an empty credential.
const REDACTED: &str = "[redacted]";

impl Secret {
    /// The credential itself. Every call site is a place a secret can leak.
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// Whether no credential was supplied.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<S: Into<String>> From<S> for Secret {
    fn from(value: S) -> Self {
        Self(value.into())
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(REDACTED)
    }
}

impl serde::Serialize for Secret {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(REDACTED)
    }
}

impl<'de> serde::Deserialize<'de> for Secret {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut value = <String as serde::Deserialize>::deserialize(deserializer)?;
        if value == REDACTED {
            value.clear();
        }
        Ok(Self(value))
    }
}

#[cfg(test)]
pub(crate) mod tests;

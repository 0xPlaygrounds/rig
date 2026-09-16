//! A credential inside provider data.

/// A credential held inside a [`Wire`](super::Wire)'s data.
///
/// A wire is plain data a host may serialize into a scene, a component or a
/// config file, so the credential must not travel with it: `Debug` prints
/// `[redacted]`, `Serialize` writes `"[redacted]"`, and only
/// [`Self::expose`] returns the value. Equality is by value, so two wires
/// built from the same key compare equal.
///
/// The round trip is lossy by contract: deserializing the `"[redacted]"`
/// sentinel yields the *empty* `Secret`, so a reloaded wire reports
/// [`Self::is_empty`] and gets its credential from the environment again
/// rather than sending the sentinel as a key.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct Secret(String);

/// What a redacted secret renders and serializes as — and the one string
/// that deserializes to no credential.
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

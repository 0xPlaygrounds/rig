//! A credential inside provider data.

/// A credential held inside a [`Wire`](super::Wire)'s data.
///
/// A wire is plain data a host may serialize into a scene, a component or a
/// config file, so the credential must not travel with it: `Debug` prints
/// `[redacted]`, `Serialize` writes `"[redacted]"`, and only
/// [`Self::expose`] returns the value. Equality is by value, so two wires
/// built from the same key compare equal.
#[derive(Clone, Default, PartialEq, Eq, Hash, serde::Deserialize)]
#[serde(transparent)]
pub struct Secret(String);

/// What a redacted secret renders and serializes as.
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

#[cfg(test)]
mod tests;

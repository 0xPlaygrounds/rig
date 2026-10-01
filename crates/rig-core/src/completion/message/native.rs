//! Provider output Rig has no canonical field for, kept in history and
//! replayed only to the wire that produced it.
//!
//! A [`Native`] is a value shaped for one wire [`Format`]. It travels sealed
//! to the service that issued it, either as a whole item
//! ([`AssistantContent::Native`](super::AssistantContent::Native)) or as the
//! residue of a canonical block ([`Text::native`](super::Text::native)). At
//! rest it is plain JSON, so it serializes, compares and reflects like any
//! other content. Code reads and writes it through a [`NativeData`] type,
//! which names its format once, so no provider code names a key or lifts
//! JSON by hand.
//!
//! ```
//! use rig_core::message::{Format, Native, NativeData};
//!
//! #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
//! struct Phase {
//!     phase: String,
//! }
//!
//! impl NativeData for Phase {
//!     const FORMAT: Format = Format::from_static("example-wire");
//! }
//!
//! let native = Native::new(&Phase { phase: "final".into() })?;
//! assert_eq!(native.decode::<Phase>().transpose()?, Some(Phase { phase: "final".into() }));
//! # Ok::<(), serde_json::Error>(())
//! ```

use std::borrow::Cow;

use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// The wire schema a native value is shaped for, such as
/// `"anthropic-messages"`. Two wires share a format only when they accept
/// each other's items byte for byte.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Format(Cow<'static, str>);

impl Format {
    /// The format named `name`, in a const context.
    pub const fn from_static(name: &'static str) -> Self {
        Self(Cow::Borrowed(name))
    }

    /// The format named `name`.
    pub fn new(name: impl Into<Cow<'static, str>>) -> Self {
        Self(name.into())
    }

    /// The format's name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A typed view of native data one wire format owns. The format is named
/// here and nowhere else.
pub trait NativeData: Serialize + DeserializeOwned {
    /// The format this data is shaped for.
    const FORMAT: Format;
}

/// A value shaped for one wire format: an output item Rig does not model,
/// or the fields of a block its canonical type has no place for.
///
/// A message carries it [`Sealed`](super::Sealed) to the service that
/// issued it. A wire replays it only when it speaks the same format and
/// accepts that issuer; every other wire leaves it out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Native {
    format: Format,
    value: serde_json::Value,
}

impl Native {
    /// `data` in its format.
    pub fn new<T: NativeData>(data: &T) -> Result<Self, serde_json::Error> {
        Ok(Self {
            format: T::FORMAT,
            value: serde_json::to_value(data)?,
        })
    }

    /// A value the decoder of `format` received and does not model, kept
    /// exactly as it arrived.
    pub fn verbatim(format: Format, value: serde_json::Value) -> Self {
        Self { format, value }
    }

    /// The format the value is shaped for.
    pub fn format(&self) -> &Format {
        &self.format
    }

    /// Whether the value is shaped for `format`.
    pub fn is(&self, format: &Format) -> bool {
        &self.format == format
    }

    /// The value as JSON, for inspection.
    pub fn value(&self) -> &serde_json::Value {
        &self.value
    }

    /// The value, consumed.
    pub fn into_value(self) -> serde_json::Value {
        self.value
    }

    /// The value as `T`, or `None` when it is shaped for another format. An
    /// error means the value is in `T`'s format but not `T`'s shape.
    pub fn decode<T: NativeData>(&self) -> Option<Result<T, serde_json::Error>> {
        self.is(&T::FORMAT)
            .then(|| serde_json::from_value(self.value.clone()))
    }
}

/// A stable fingerprint of a block's text: FNV-1a over its UTF-8 bytes.
/// The same text has the same fingerprint on every platform and under every
/// serde feature set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(into = "String", try_from = "String")]
pub struct Fingerprint(u64);

impl Fingerprint {
    /// The fingerprint of `text`.
    pub fn of(text: &str) -> Self {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        Self(text.bytes().fold(OFFSET, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(PRIME)
        }))
    }
}

impl From<Fingerprint> for String {
    fn from(fingerprint: Fingerprint) -> Self {
        format!("{:016x}", fingerprint.0)
    }
}

/// A fingerprint was not 16 hexadecimal digits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("a text fingerprint must be 16 hexadecimal digits")]
pub struct InvalidFingerprint;

impl TryFrom<String> for Fingerprint {
    type Error = InvalidFingerprint;

    fn try_from(hex: String) -> Result<Self, InvalidFingerprint> {
        if hex.len() != 16 {
            return Err(InvalidFingerprint);
        }
        u64::from_str_radix(&hex, 16)
            .map(Self)
            .map_err(|_| InvalidFingerprint)
    }
}

/// Native data that describes a block's own text, such as annotations that
/// index into it. It is valid only for the text it was decoded with: an
/// edit to the text makes it stale, and a stale value is never read.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextBound<T> {
    text: Fingerprint,
    value: T,
}

impl<T> TextBound<T> {
    /// `value`, bound to `text`.
    pub fn new(text: &str, value: T) -> Self {
        Self {
            text: Fingerprint::of(text),
            value,
        }
    }

    /// The value, when `text` is still the text it was bound to.
    pub fn for_text(&self, text: &str) -> Option<&T> {
        (self.text == Fingerprint::of(text)).then_some(&self.value)
    }

    /// The value, when `text` is still the text it was bound to.
    pub fn into_for_text(self, text: &str) -> Option<T> {
        (self.text == Fingerprint::of(text)).then_some(self.value)
    }
}

#[cfg(test)]
mod tests;

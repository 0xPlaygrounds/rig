//! Provider items with no canonical meaning, kept verbatim for replay to the
//! wire format that produced them.
//!
//! A decoder lifts every output item it receives. An item that is not text,
//! reasoning, a tool call or an image becomes a [`Native`] item, sealed to its
//! issuer like reasoning. A request replays it only when the encoding wire
//! speaks the same [`WireFormat`] and accepts the issuer; any other wire leaves
//! it out.
//!
//! ```
//! use rig_core::message::{Native, NativeDialect, WireFormat};
//!
//! struct Example;
//! impl NativeDialect for Example {
//!     const FORMAT: WireFormat = WireFormat::from_static("example.v1");
//!     type Item = serde_json::Value;
//! }
//!
//! let item = Native::new::<Example>(&serde_json::json!({"type": "novel"}))?;
//! assert_eq!(item.format(), &Example::FORMAT);
//! assert!(item.decode::<Example>().is_some());
//! # Ok::<(), serde_json::Error>(())
//! ```

use std::borrow::Cow;

use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// A wire format: the shape of a [`Native`] item, independent of the vendor
/// that issued it. Anthropic and Z.AI share `anthropic.messages`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WireFormat(Cow<'static, str>);

impl WireFormat {
    /// The format named `name`, in a const context.
    pub const fn from_static(name: &'static str) -> Self {
        Self(Cow::Borrowed(name))
    }

    /// The format's name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for WireFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Binds a [`WireFormat`] to the wire type its items decode as, so an item
/// of one format cannot be read as another's.
pub trait NativeDialect {
    /// The format's name.
    const FORMAT: WireFormat;
    /// The provider's wire item type. Its deserializer must accept every
    /// item the provider can send, including ones it does not model.
    type Item: Serialize + DeserializeOwned;
}

/// One provider output item with no canonical meaning, held verbatim.
///
/// It holds no copy of any other block, so editing sibling content never makes
/// it stale. Only its own wire format reads it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Native {
    format: WireFormat,
    item: serde_json::Value,
}

impl Native {
    /// `item`, held for replay to `D`'s wire format.
    ///
    /// # Errors
    /// When `item` does not serialize to JSON.
    pub fn new<D: NativeDialect>(item: &D::Item) -> Result<Self, serde_json::Error> {
        Ok(Self {
            format: D::FORMAT,
            item: serde_json::to_value(item)?,
        })
    }

    /// The item as `D`'s wire type, or `None` when it belongs to another
    /// format.
    pub fn decode<D: NativeDialect>(&self) -> Option<Result<D::Item, serde_json::Error>> {
        (self.format == D::FORMAT).then(|| D::Item::deserialize(&self.item))
    }

    /// The item as the provider sent it, or `None` when it belongs to another
    /// format. Encoders replay this, verbatim.
    pub fn item_for<D: NativeDialect>(&self) -> Option<&serde_json::Value> {
        (self.format == D::FORMAT).then_some(&self.item)
    }

    /// The wire format the item belongs to.
    pub fn format(&self) -> &WireFormat {
        &self.format
    }

    /// The item as the provider sent it, for display and inspection.
    pub fn item(&self) -> &serde_json::Value {
        &self.item
    }

    /// The item's `type` tag, when it has one.
    pub fn kind(&self) -> Option<&str> {
        self.item.get("type").and_then(serde_json::Value::as_str)
    }
}

#[cfg(test)]
mod tests;

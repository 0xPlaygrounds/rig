use serde::{Deserialize, Serialize};

/// Supported encoding types for media
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Base64(pub String);

impl<S> From<S> for Base64
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Base64(value.into())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Url(pub url::Url);

impl From<url::Url> for Url {
    fn from(value: url::Url) -> Self {
        Self(value)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Raw(pub Vec<u8>);

impl From<Vec<u8>> for Raw {
    fn from(value: Vec<u8>) -> Self {
        Raw(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnyEncoding {
    Base64(Base64),
    Url(Url),
    Raw(Raw),
}

impl Default for AnyEncoding {
    fn default() -> Self {
        Self::Raw(Default::default())
    }
}

impl From<Base64> for AnyEncoding {
    fn from(value: Base64) -> Self {
        Self::Base64(value)
    }
}

impl From<Raw> for AnyEncoding {
    fn from(value: Raw) -> Self {
        Self::Raw(value)
    }
}

impl From<Url> for AnyEncoding {
    fn from(value: Url) -> Self {
        Self::Url(value)
    }
}

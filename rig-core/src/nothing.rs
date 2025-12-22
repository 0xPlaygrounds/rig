use std::any::Any;

use crate::client;
use serde::{Deserialize, Serialize, de::Visitor};

/// A type containing nothing at all. For `Option`-like behavior on the type level, i.e. to describe
/// the lack of a capability or field (an API key, for instance)
#[derive(Debug, Default, Clone, Copy)]
pub struct Nothing;

impl client::ApiKey for Nothing {}

impl TryFrom<String> for Nothing {
    type Error = &'static str;

    fn try_from(_: String) -> Result<Self, Self::Error> {
        Err(
            "Tried to create a Nothing from a string - this should not happen, please file an issue",
        )
    }
}

impl Serialize for Nothing {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_unit()
    }
}

impl<'de> Deserialize<'de> for Nothing {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_ignored_any(IgnoredAnyVisitor);

        Ok(Nothing)
    }
}

struct IgnoredAnyVisitor;

impl<'de> Visitor<'de> for IgnoredAnyVisitor {
    type Value = Nothing;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("nothing")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(Nothing)
    }
}

pub fn is_nothing<T: ?Sized + Any + 'static>(_: &T) -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<Nothing>()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn nothing_is_nothing() {
        assert!(is_nothing(&Nothing))
    }

    #[test]
    fn something_is_not_nothing() {
        assert!(!is_nothing(&String::new()))
    }
}

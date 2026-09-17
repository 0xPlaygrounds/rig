//! Inspect wire object keys without collecting question definitions into a map.
use crate::{Error, types::Question};
use serde::Deserializer;
use serde::de::{DeserializeOwned, MapAccess, Visitor};
use serde_json::value::RawValue;
use std::{collections::BTreeSet, fmt, marker::PhantomData};

fn object_ids<T: DeserializeOwned>(
    raw: &RawValue,
    validate: fn(&str, &T) -> Result<(), Error>,
) -> Result<BTreeSet<String>, serde_json::Error> {
    struct Keys<T> {
        validate: fn(&str, &T) -> Result<(), Error>,
        marker: PhantomData<T>,
    }
    impl<'de, T: DeserializeOwned> Visitor<'de> for Keys<T> {
        type Value = BTreeSet<String>;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("an object keyed by unique question IDs")
        }
        fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
            let mut ids = BTreeSet::new();
            while let Some((id, value)) = map.next_entry::<String, T>()? {
                if id.is_empty() || !ids.insert(id.clone()) {
                    return Err(serde::de::Error::custom(
                        "question IDs must be nonempty and unique",
                    ));
                }
                (self.validate)(&id, &value).map_err(serde::de::Error::custom)?;
            }
            Ok(ids)
        }
    }
    let mut deserializer = serde_json::Deserializer::from_str(raw.get());
    let ids = deserializer.deserialize_map(Keys {
        validate,
        marker: PhantomData,
    })?;
    deserializer.end()?;
    Ok(ids)
}

pub(crate) fn request_ids(raw: &RawValue) -> Result<BTreeSet<String>, Error> {
    let ids = object_ids::<Question>(raw, crate::questions::validate_definition)
        .map_err(|error| Error::InvalidRequest(error.to_string()))?;
    if ids.is_empty() {
        return Err(Error::InvalidRequest(
            "at least one question is required".into(),
        ));
    }
    Ok(ids)
}
pub(crate) fn response_ids(raw: &RawValue) -> Result<BTreeSet<String>, Error> {
    object_ids::<serde::de::IgnoredAny>(raw, |_, _| Ok(()))
        .map_err(|error| Error::InvalidResponse(error.to_string()))
}

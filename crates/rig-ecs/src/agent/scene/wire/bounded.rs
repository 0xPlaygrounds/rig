//! Limit the intermediate JSON tree even when callers use generic serde APIs.
use std::fmt;

use serde::{
    Deserializer,
    de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor},
};
use serde_json::{Map, Number, Value};

use super::Budget;

pub(super) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Value, D::Error> {
    Seed {
        budget: &mut Budget::new(),
        depth: 0,
    }
    .deserialize(deserializer)
}

/// Match the reader's accounting on the transformed envelope without copying it.
pub(super) fn validate(value: &Value) -> Result<(), String> {
    fn visit(value: &Value, budget: &mut Budget, depth: usize) -> Result<(), String> {
        budget.take(8, depth)?;
        match value {
            Value::String(text) => budget.take(text.len(), depth)?,
            Value::Array(items) => {
                for item in items {
                    visit(item, budget, depth + 1)?;
                }
            }
            Value::Object(map) => {
                for (key, value) in map {
                    budget.take(key.len(), depth)?;
                    visit(value, budget, depth + 1)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    visit(value, &mut Budget::new(), 0)
}

struct Seed<'a> {
    budget: &'a mut Budget,
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for Seed<'_> {
    type Value = Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        self.budget.take(8, self.depth).map_err(de::Error::custom)?;
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Seed<'_> {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded scene value")
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }
    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }
    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }
    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Value, E> {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("non-finite scene number"))
    }
    fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_none<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_str<E: de::Error>(self, value: &str) -> Result<Value, E> {
        self.budget
            .take(value.len(), self.depth)
            .map_err(E::custom)?;
        Ok(Value::String(value.to_owned()))
    }
    fn visit_string<E: de::Error>(self, value: String) -> Result<Value, E> {
        self.budget
            .take(value.len(), self.depth)
            .map_err(E::custom)?;
        Ok(Value::String(value))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Value, A::Error> {
        // Do not trust size_hint: a hostile input must not reserve arbitrary memory.
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(Seed {
            budget: self.budget,
            depth: self.depth + 1,
        })? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut values = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            self.budget
                .take(key.len(), self.depth)
                .map_err(de::Error::custom)?;
            if values.contains_key(&key) {
                return Err(de::Error::custom("duplicate scene object key"));
            }
            let value = map.next_value_seed(Seed {
                budget: self.budget,
                depth: self.depth + 1,
            })?;
            values.insert(key, value);
        }
        Ok(Value::Object(values))
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "controlled validation fixtures"
)]
mod tests;

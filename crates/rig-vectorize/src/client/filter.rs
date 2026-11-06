//! Filter implementation for Cloudflare Vectorize.

use rig_core::vector_store::request::SearchFilter;
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value, json};

use super::VectorizeError;

/// Key of the marker that makes [`VectorizeFilter::validate`] fail.
const UNSUPPORTED: &str = "$unsupported";

/// Metadata filter for Vectorize queries, supporting the `$eq`, `$ne`, `$gt`,
/// `$lt`, `$gte`, `$lte`, `$in`, and `$nin` operators over indexed fields.
///
/// Vectorize has no disjunction, so `or`, like an `and` that repeats an
/// operator with another value, produces a filter that
/// [`VectorizeFilter::validate`] rejects.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorizeFilter(Value);

impl VectorizeFilter {
    /// Creates an empty filter.
    pub fn new() -> Self {
        Self(json!({}))
    }

    /// Returns the inner JSON value.
    pub fn into_inner(self) -> Value {
        self.0
    }

    /// Returns a reference to the inner JSON value.
    pub fn as_value(&self) -> &Value {
        &self.0
    }

    /// Whether the filter constrains nothing.
    pub fn is_empty(&self) -> bool {
        self.0.as_object().is_none_or(serde_json::Map::is_empty)
    }
}

impl SearchFilter for VectorizeFilter {
    type Value = Value;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(json!({ key.as_ref(): { "$eq": value } }))
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(json!({ key.as_ref(): { "$gt": value } }))
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(json!({ key.as_ref(): { "$lt": value } }))
    }

    /// Merges both filters into one object, which Vectorize evaluates as a
    /// conjunction. Operators on the same field are combined; an operator
    /// repeated with a different value produces a filter that fails validation.
    /// Numbers compare by value, so `5` and `5.0` do not conflict.
    fn and(self, rhs: Self) -> Self {
        let (Some(mut merged), Some(rhs)) = (self.into_object(), rhs.into_object()) else {
            return Self::unsupported("AND operands must be filter objects");
        };

        for (field, rhs_ops) in rhs {
            let Some(lhs_ops) = merged.get_mut(&field) else {
                merged.insert(field, rhs_ops);
                continue;
            };
            if *lhs_ops == rhs_ops {
                continue;
            }
            let (Value::Object(lhs_ops), Value::Object(rhs_ops)) = (lhs_ops, rhs_ops) else {
                return Self::unsupported(format!("AND cannot combine conditions on `{field}`"));
            };
            for (op, value) in rhs_ops {
                if lhs_ops
                    .get(&op)
                    .is_some_and(|existing| !same_value(existing, &value))
                {
                    return Self::unsupported(format!(
                        "AND cannot repeat `{op}` on `{field}` with different values"
                    ));
                }
                lhs_ops.insert(op, value);
            }
        }

        Self(Value::Object(merged))
    }

    /// Discards both operands and returns a filter that fails validation, since
    /// Vectorize has no disjunction and this method cannot fail.
    fn or(self, _rhs: Self) -> Self {
        tracing::error!("Vectorize does not support OR filters. This filter will fail.");
        Self::unsupported("OR filters are not supported by Vectorize")
    }
}

impl VectorizeFilter {
    /// Matches records whose value at `key` differs from `value`.
    pub fn ne(key: impl AsRef<str>, value: &Value) -> Self {
        Self(json!({ key.as_ref(): { "$ne": value } }))
    }

    /// Matches records whose value at `key` is at least `value`.
    pub fn gte(key: impl AsRef<str>, value: &Value) -> Self {
        Self(json!({ key.as_ref(): { "$gte": value } }))
    }

    /// Matches records whose value at `key` is at most `value`.
    pub fn lte(key: impl AsRef<str>, value: &Value) -> Self {
        Self(json!({ key.as_ref(): { "$lte": value } }))
    }

    /// Matches records whose value at `key` is one of `values`.
    pub fn in_values(key: impl AsRef<str>, values: &[Value]) -> Self {
        Self(json!({ key.as_ref(): { "$in": values } }))
    }

    /// Matches records whose value at `key` is none of `values`.
    pub fn nin(key: impl AsRef<str>, values: &[Value]) -> Self {
        Self(json!({ key.as_ref(): { "$nin": values } }))
    }

    /// Rejects filters that `or`, or a conflicting `and`, made unrepresentable.
    pub fn validate(&self) -> Result<(), VectorizeError> {
        match self.0.get(UNSUPPORTED) {
            Some(reason) => Err(VectorizeError::UnsupportedFilterOperation(
                reason.as_str().unwrap_or_default().to_string(),
            )),
            None => Ok(()),
        }
    }

    /// A filter that [`Self::validate`] rejects with `reason`.
    fn unsupported(reason: impl Into<String>) -> Self {
        Self(json!({ UNSUPPORTED: reason.into() }))
    }

    /// The filter's top-level object; the default `null` filter is empty.
    fn into_object(self) -> Option<serde_json::Map<String, Value>> {
        match self.0 {
            Value::Object(object) => Some(object),
            Value::Null => Some(serde_json::Map::new()),
            _ => None,
        }
    }
}

/// Whether two operator values are equal, comparing numbers by value.
fn same_value(lhs: &Value, rhs: &Value) -> bool {
    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => same_number(lhs, rhs),
        (Value::Array(lhs), Value::Array(rhs)) => {
            lhs.len() == rhs.len() && lhs.iter().zip(rhs).all(|(l, r)| same_value(l, r))
        }
        _ => lhs == rhs,
    }
}

/// Compares exactly: integers past 2^53 are not rounded to `f64` first.
fn same_number(lhs: &Number, rhs: &Number) -> bool {
    let integer = |n: &Number| {
        n.as_i64()
            .map(i128::from)
            .or_else(|| n.as_u64().map(i128::from))
    };
    let equals_float = |int: i128, float: Option<f64>| {
        float.is_some_and(|float| float.fract() == 0.0 && float as i128 == int)
    };

    match (integer(lhs), integer(rhs)) {
        (Some(lhs), Some(rhs)) => lhs == rhs,
        (Some(int), None) => equals_float(int, rhs.as_f64()),
        (None, Some(int)) => equals_float(int, lhs.as_f64()),
        (None, None) => lhs.as_f64() == rhs.as_f64(),
    }
}

#[cfg(test)]
mod tests;

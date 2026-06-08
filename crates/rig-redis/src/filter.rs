//! Redis-specific filter types for RediSearch `FT.SEARCH` queries.
//!
//! Provides [`Filter`] which implements [`SearchFilter`] and translates
//! Rig's generic filter expressions into RediSearch query syntax.
//!
//! # Field Type Expectations
//!
//! - **Numeric fields**: `eq`, `gt`, `lt`, `gte`, `lte`, `range` produce range syntax (`@field:[min max]`)
//! - **Tag fields**: String equality uses tag syntax (`@field:{value}`)
//! - **Bool fields**: Treated as numeric TAG (1/0) with tag syntax
//!
//! Ensure your RediSearch schema matches the filter types you use.

use rig_core::vector_store::request::{Filter as CoreFilter, FilterError, SearchFilter};
use serde::{Deserialize, Serialize};

/// Typed value for Redis filter expressions.
///
/// Determines how the value is formatted in the RediSearch query syntax.
#[derive(Debug, Clone, PartialEq)]
pub enum RedisValue {
    /// Numeric value — produces range syntax for all comparisons.
    Number(f64),
    /// String/tag value — produces tag syntax `{value}`.
    String(String),
    /// Boolean value — treated as numeric TAG (`1` or `0`).
    Bool(bool),
}

impl RedisValue {
    /// Formats value for tag-style filters (`@field:{value}` or `@field:{1}`).
    fn to_tag_expr(&self) -> String {
        match self {
            RedisValue::Number(n) => n.to_string(),
            RedisValue::String(s) => s.clone(),
            RedisValue::Bool(b) => {
                if *b {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            }
        }
    }
}

impl From<i64> for RedisValue {
    fn from(value: i64) -> Self {
        Self::Number(value as f64)
    }
}

impl From<u64> for RedisValue {
    fn from(value: u64) -> Self {
        Self::Number(value as f64)
    }
}

impl From<f64> for RedisValue {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

impl From<bool> for RedisValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<String> for RedisValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for RedisValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl TryFrom<serde_json::Value> for RedisValue {
    type Error = FilterError;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        match value {
            serde_json::Value::Bool(b) => Ok(RedisValue::Bool(b)),
            serde_json::Value::Number(n) => {
                let num = n.as_f64().ok_or_else(|| FilterError::Expected {
                    expected: "Valid 64-bit float".into(),
                    got: "Invalid 64-bit float".into(),
                })?;
                Ok(RedisValue::Number(num))
            }
            serde_json::Value::String(s) => Ok(RedisValue::String(s)),
            serde_json::Value::Null
            | serde_json::Value::Array(_)
            | serde_json::Value::Object(_) => Err(FilterError::TypeError(
                "Redis filter does not currently support null values, arrays or objects".into(),
            )),
        }
    }
}

/// Redis filter for FT.SEARCH queries.
///
/// Wraps a raw RediSearch query string. Combine filters with [`SearchFilter::and`]
/// and [`SearchFilter::or`], or use the additional helpers like [`Filter::range`]
/// and [`Filter::tag_in`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Filter(String);

impl SearchFilter for Filter {
    type Value = RedisValue;

    /// Equality filter.
    ///
    /// - Numeric: `@field:[val val]` (exact range match)
    /// - String: `@field:{value}` (tag match)
    /// - Bool: `@field:{1}` or `@field:{0}` (tag match)
    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        match value {
            RedisValue::Number(n) => Self(format!("@{}:[{} {}]", key.as_ref(), n, n)),
            RedisValue::String(ref s) => Self(format!("@{}:{{{}}}", key.as_ref(), s)),
            RedisValue::Bool(b) => {
                let v = if b { "1" } else { "0" };
                Self(format!("@{}:{{{}}}", key.as_ref(), v))
            }
        }
    }

    /// Greater-than filter (exclusive).
    ///
    /// Numeric: `@field:[(val +inf]`. Non-numeric falls back to tag syntax.
    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        match value {
            RedisValue::Number(n) => Self(format!("@{}:[({} +inf]", key.as_ref(), n)),
            _ => Self(format!("@{}:{{{}}}", key.as_ref(), value.to_tag_expr())),
        }
    }

    /// Less-than filter (exclusive).
    ///
    /// Numeric: `@field:[-inf (val]`. Non-numeric falls back to tag syntax.
    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        match value {
            RedisValue::Number(n) => Self(format!("@{}:[-inf ({}]", key.as_ref(), n)),
            _ => Self(format!("@{}:{{{}}}", key.as_ref(), value.to_tag_expr())),
        }
    }

    fn and(self, rhs: Self) -> Self {
        Self(format!("({} {})", self.0, rhs.0))
    }

    fn or(self, rhs: Self) -> Self {
        Self(format!("({} | {})", self.0, rhs.0))
    }
}

impl Filter {
    /// Negates this filter expression.
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self(format!("-{}", self.0))
    }

    /// Greater than or equal (inclusive).
    ///
    /// Numeric: `@field:[val +inf]`.
    pub fn gte(key: impl AsRef<str>, value: <Self as SearchFilter>::Value) -> Self {
        match value {
            RedisValue::Number(n) => Self(format!("@{}:[{} +inf]", key.as_ref(), n)),
            _ => Self(format!("@{}:{{{}}}", key.as_ref(), value.to_tag_expr())),
        }
    }

    /// Less than or equal (inclusive).
    ///
    /// Numeric: `@field:[-inf val]`.
    pub fn lte(key: impl AsRef<str>, value: <Self as SearchFilter>::Value) -> Self {
        match value {
            RedisValue::Number(n) => Self(format!("@{}:[-inf {}]", key.as_ref(), n)),
            _ => Self(format!("@{}:{{{}}}", key.as_ref(), value.to_tag_expr())),
        }
    }

    /// Numeric range filter (inclusive on both ends).
    pub fn range(key: impl AsRef<str>, min: f64, max: f64) -> Self {
        Self(format!("@{}:[{} {}]", key.as_ref(), min, max))
    }

    /// Numeric range filter (exclusive on both ends).
    pub fn range_exclusive(key: impl AsRef<str>, min: f64, max: f64) -> Self {
        Self(format!("@{}:[({} ({}]", key.as_ref(), min, max))
    }

    /// Tag filter for multiple values (OR).
    ///
    /// Produces `@field:{val1 | val2 | val3}`.
    pub fn tag_in(key: impl AsRef<str>, values: Vec<String>) -> Self {
        let tags = values.join(" | ");
        Self(format!("@{}:{{{}}}", key.as_ref(), tags))
    }

    /// Full-text search within a TEXT field.
    pub fn text_contains(key: impl AsRef<str>, text: impl AsRef<str>) -> Self {
        Self(format!("@{}:{}", key.as_ref(), text.as_ref()))
    }

    /// Consumes the filter and returns the raw RediSearch query string.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl TryFrom<CoreFilter<serde_json::Value>> for Filter {
    type Error = FilterError;

    fn try_from(value: CoreFilter<serde_json::Value>) -> Result<Self, Self::Error> {
        let filter = match value {
            CoreFilter::Eq(k, val) => Filter::eq(k, val.try_into()?),
            CoreFilter::Gt(k, val) => Filter::gt(k, val.try_into()?),
            CoreFilter::Lt(k, val) => Filter::lt(k, val.try_into()?),
            CoreFilter::And(l, r) => Self::try_from(*l)?.and(Self::try_from(*r)?),
            CoreFilter::Or(l, r) => Self::try_from(*l)?.or(Self::try_from(*r)?),
        };

        Ok(filter)
    }
}

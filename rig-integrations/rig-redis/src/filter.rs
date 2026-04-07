//! Redis-specific filter types for RediSearch `FT.SEARCH` queries.
//!
//! Provides [`Filter`] which implements [`SearchFilter`] and translates
//! Rig's generic filter expressions into RediSearch query syntax.

use rig::vector_store::request::{Filter as CoreFilter, FilterError, SearchFilter};
use serde::{Deserialize, Serialize};

/// Redis filter value type.
#[derive(Debug, Clone, PartialEq)]
pub enum RedisValue {
    Number(f64),
    String(String),
    Bool(bool),
}

impl RedisValue {
    fn to_redis_expr(&self) -> String {
        match self {
            RedisValue::Number(n) => n.to_string(),
            RedisValue::String(s) => format!("{{{}}}", s),
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

/// Redis filter for FT.SEARCH queries
///
/// Redis uses a query syntax like: `@field:[min max]` for numeric ranges,
/// `@field:{value}` for tags, etc.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Filter(String);

impl SearchFilter for Filter {
    type Value = RedisValue;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(format!("@{}:{}", key.as_ref(), value.to_redis_expr()))
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        match value {
            RedisValue::Number(n) => Self(format!("@{}:[({} +inf]", key.as_ref(), n)),
            _ => Self(format!("@{}:{}", key.as_ref(), value.to_redis_expr())),
        }
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        match value {
            RedisValue::Number(n) => Self(format!("@{}:[-inf ({}]", key.as_ref(), n)),
            _ => Self(format!("@{}:{}", key.as_ref(), value.to_redis_expr())),
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

    /// Greater than or equal
    pub fn gte(key: impl AsRef<str>, value: <Self as SearchFilter>::Value) -> Self {
        match value {
            RedisValue::Number(n) => Self(format!("@{}:[{} +inf]", key.as_ref(), n)),
            _ => Self(format!("@{}:{}", key.as_ref(), value.to_redis_expr())),
        }
    }

    /// Less than or equal
    pub fn lte(key: impl AsRef<str>, value: <Self as SearchFilter>::Value) -> Self {
        match value {
            RedisValue::Number(n) => Self(format!("@{}:[-inf {}]", key.as_ref(), n)),
            _ => Self(format!("@{}:{}", key.as_ref(), value.to_redis_expr())),
        }
    }

    /// Range filter (inclusive)
    pub fn range(key: impl AsRef<str>, min: f64, max: f64) -> Self {
        Self(format!("@{}:[{} {}]", key.as_ref(), min, max))
    }

    /// Range filter (exclusive)
    pub fn range_exclusive(key: impl AsRef<str>, min: f64, max: f64) -> Self {
        Self(format!("@{}:[({} ({}]", key.as_ref(), min, max))
    }

    /// Tag filter for multiple values (OR)
    pub fn tag_in(key: impl AsRef<str>, values: Vec<String>) -> Self {
        let tags = values.join(" | ");
        Self(format!("@{}:{{{}}}", key.as_ref(), tags))
    }

    /// Text search in field
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

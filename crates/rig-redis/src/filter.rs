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
//!
//! # Filtering Limitations
//!
//! Filters apply to fields that exist in your RediSearch index schema and are
//! present in the stored hash keys. Documents inserted via [`InsertDocuments`](crate::RedisVectorStore)
//! only store `document`, `embedded_text`, and the vector field. To use filters,
//! you must either write additional hash fields out-of-band or use a custom
//! insertion approach that includes the filterable fields in the hash.
//!
//! # Escaping
//!
//! Tag values are automatically escaped for RediSearch special characters.
//! See [`escape_tag_value`] for the list of escaped characters.

use rig_core::vector_store::request::{Filter as CoreFilter, FilterError, SearchFilter};
use serde::{Deserialize, Serialize};

/// Characters that must be escaped with a backslash in RediSearch tag queries.
///
/// Per the [Redis documentation](https://redis.io/docs/latest/develop/ai/search-and-query/advanced-concepts/tags/),
/// these characters have special meaning in tag query syntax:
/// - `'` (single quote)
/// - `-` (hyphen / negation)
/// - `(`, `)` (parentheses / grouping)
/// - `[`, `]`, `{`, `}` (brackets / range/tag delimiters)
/// - `|` (pipe / OR operator)
/// - `@` (field prefix)
/// - `\\` (backslash / escape character itself)
const TAG_ESCAPE_CHARS: &[char] = &['\'', '-', '(', ')', '[', ']', '{', '}', '|', '@', '\\'];

/// Escapes special characters in a value for use in RediSearch tag queries.
///
/// Prepends a backslash to characters that have special meaning in RediSearch
/// tag syntax to prevent query injection or parse failures.
///
/// # Example
/// ```
/// use rig_redis::filter::escape_tag_value;
///
/// assert_eq!(escape_tag_value("hello-world"), r"hello\-world");
/// assert_eq!(escape_tag_value("it's"), r"it\'s");
/// assert_eq!(escape_tag_value("foo|bar"), r"foo\|bar");
/// ```
pub fn escape_tag_value(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        if TAG_ESCAPE_CHARS.contains(&ch) {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

/// Characters that must be escaped with a backslash in RediSearch text field queries.
///
/// Per the [Redis tokenization documentation](https://redis.io/docs/latest/develop/ai/search-and-query/advanced-concepts/escaping/),
/// these punctuation marks are token separators in TEXT fields:
/// `,.<>{}[]"':;!@#$%^&*()-+=~`
const TEXT_ESCAPE_CHARS: &[char] = &[
    ',', '.', '<', '>', '{', '}', '[', ']', '"', '\'', ':', ';', '!', '@', '#', '$', '%', '^', '&',
    '*', '(', ')', '-', '+', '=', '~', '|', '\\',
];

/// Escapes special characters in a value for use in RediSearch TEXT field queries.
///
/// Prepends a backslash to token-separator characters.
pub fn escape_text_value(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        if TEXT_ESCAPE_CHARS.contains(&ch) {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

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
    ///
    /// Values are escaped for RediSearch special characters.
    fn to_tag_expr(&self) -> String {
        match self {
            RedisValue::Number(n) => n.to_string(),
            RedisValue::String(s) => escape_tag_value(s),
            RedisValue::Bool(b) => {
                if *b {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            }
        }
    }

    /// Returns `true` if this value is numeric.
    fn is_numeric(&self) -> bool {
        matches!(self, RedisValue::Number(_))
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
    /// - String: `@field:{value}` (tag match, value is escaped)
    /// - Bool: `@field:{1}` or `@field:{0}` (tag match)
    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        match value {
            RedisValue::Number(n) => Self(format!("@{}:[{} {}]", key.as_ref(), n, n)),
            RedisValue::String(ref s) => {
                Self(format!("@{}:{{{}}}", key.as_ref(), escape_tag_value(s)))
            }
            RedisValue::Bool(b) => {
                let v = if b { "1" } else { "0" };
                Self(format!("@{}:{{{}}}", key.as_ref(), v))
            }
        }
    }

    /// Greater-than filter (exclusive).
    ///
    /// Numeric: `@field:[(val +inf]`.
    ///
    /// Non-numeric values are not meaningful for range comparisons and will
    /// produce a tag-equality filter with a warning log. Prefer using [`Filter::eq`]
    /// for non-numeric values.
    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        if !value.is_numeric() {
            tracing::warn!(
                target: "rig",
                field = %key.as_ref(),
                "gt() called with non-numeric value; falling back to tag-equality semantics. \
                 Use eq() for string/bool comparisons."
            );
        }
        match value {
            RedisValue::Number(n) => Self(format!("@{}:[({} +inf]", key.as_ref(), n)),
            _ => Self(format!("@{}:{{{}}}", key.as_ref(), value.to_tag_expr())),
        }
    }

    /// Less-than filter (exclusive).
    ///
    /// Numeric: `@field:[-inf (val]`.
    ///
    /// Non-numeric values are not meaningful for range comparisons and will
    /// produce a tag-equality filter with a warning log. Prefer using [`Filter::eq`]
    /// for non-numeric values.
    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        if !value.is_numeric() {
            tracing::warn!(
                target: "rig",
                field = %key.as_ref(),
                "lt() called with non-numeric value; falling back to tag-equality semantics. \
                 Use eq() for string/bool comparisons."
            );
        }
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
    ///
    /// Non-numeric values produce a tag-equality filter with a warning.
    pub fn gte(key: impl AsRef<str>, value: <Self as SearchFilter>::Value) -> Self {
        if !value.is_numeric() {
            tracing::warn!(
                target: "rig",
                field = %key.as_ref(),
                "gte() called with non-numeric value; falling back to tag-equality semantics."
            );
        }
        match value {
            RedisValue::Number(n) => Self(format!("@{}:[{} +inf]", key.as_ref(), n)),
            _ => Self(format!("@{}:{{{}}}", key.as_ref(), value.to_tag_expr())),
        }
    }

    /// Less than or equal (inclusive).
    ///
    /// Numeric: `@field:[-inf val]`.
    ///
    /// Non-numeric values produce a tag-equality filter with a warning.
    pub fn lte(key: impl AsRef<str>, value: <Self as SearchFilter>::Value) -> Self {
        if !value.is_numeric() {
            tracing::warn!(
                target: "rig",
                field = %key.as_ref(),
                "lte() called with non-numeric value; falling back to tag-equality semantics."
            );
        }
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
    /// Produces `@field:{val1 | val2 | val3}`. Values are escaped for special characters.
    pub fn tag_in(key: impl AsRef<str>, values: Vec<String>) -> Self {
        let tags = values
            .iter()
            .map(|v| escape_tag_value(v))
            .collect::<Vec<_>>()
            .join(" | ");
        Self(format!("@{}:{{{}}}", key.as_ref(), tags))
    }

    /// Full-text search within a TEXT field.
    ///
    /// The text value is escaped for RediSearch text-field special characters.
    pub fn text_contains(key: impl AsRef<str>, text: impl AsRef<str>) -> Self {
        Self(format!(
            "@{}:{}",
            key.as_ref(),
            escape_text_value(text.as_ref())
        ))
    }

    /// Creates a filter from a raw RediSearch query string.
    ///
    /// No escaping is applied — the caller is responsible for correct syntax.
    pub fn raw(query: impl Into<String>) -> Self {
        Self(query.into())
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

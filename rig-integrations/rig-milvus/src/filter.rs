use rig::vector_store::request::{Filter as CoreFilter, FilterError, SearchFilter};
use serde::{Deserialize, Serialize};

pub enum MilvusValue {
    Number(f64),
    Bool(bool),
    String(String),
    Array(Vec<Self>),
}

impl MilvusValue {
    fn escaped(self) -> String {
        use MilvusValue::*;

        match self {
            Number(n) => n.to_string(),
            Bool(b) => b.to_string(),
            String(s) => format!("\"{}\"", s.replace("\\", "\\\\").replace("\"", "\\\"")),
            Array(arr) => format!(
                "[{}]",
                arr.into_iter()
                    .map(MilvusValue::escaped)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

impl From<i64> for MilvusValue {
    fn from(value: i64) -> Self {
        Self::Number(value as f64)
    }
}

impl From<u64> for MilvusValue {
    fn from(value: u64) -> Self {
        Self::Number(value as f64)
    }
}

impl From<f64> for MilvusValue {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

impl From<bool> for MilvusValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<String> for MilvusValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl TryFrom<serde_json::Value> for MilvusValue {
    type Error = FilterError;
    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        match value {
            serde_json::Value::Bool(b) => Ok(MilvusValue::Bool(b)),
            serde_json::Value::Number(n) => {
                Ok(MilvusValue::Number(n.as_f64().ok_or_else(|| {
                    FilterError::Expected {
                        expected: "Valid 64-bit float".into(),
                        got: "Invalid 64-bit float".into(),
                    }
                })?))
            }
            serde_json::Value::String(s) => Ok(MilvusValue::String(s)),
            serde_json::Value::Array(arr) => Ok(MilvusValue::Array(
                arr.into_iter()
                    .map(MilvusValue::try_from)
                    .collect::<Result<_, _>>()?,
            )),
            serde_json::Value::Null | serde_json::Value::Object(_) => Err(FilterError::TypeError(
                "Milvus filter does not currently support null values or objects".into(),
            )),
        }
    }
}

impl<T> From<Vec<T>> for MilvusValue
where
    Self: From<T>,
{
    fn from(value: Vec<T>) -> Self {
        Self::Array(value.into_iter().map(Self::from).collect())
    }
}

impl<T> From<&[T]> for MilvusValue
where
    MilvusValue: From<T>,
    T: Clone,
{
    fn from(value: &[T]) -> Self {
        Self::Array(value.iter().cloned().map(Self::from).collect())
    }
}

impl<T, const N: usize> From<[T; N]> for MilvusValue
where
    Self: From<T>,
{
    fn from(value: [T; N]) -> Self {
        Self::Array(value.into_iter().map(Self::from).collect())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Filter(String);

impl SearchFilter for Filter {
    type Value = MilvusValue;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(format!("{} == {}", key.as_ref(), value.escaped()))
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(format!("{} > {}", key.as_ref(), value.escaped()))
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(format!("{} < {}", key.as_ref(), value.escaped()))
    }

    fn and(self, rhs: Self) -> Self {
        Self(format!("({}) AND ({})", self.0, rhs.0))
    }

    fn or(self, rhs: Self) -> Self {
        Self(format!("({}) OR ({})", self.0, rhs.0))
    }
}

impl Filter {
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self(format!("NOT ({})", self.0))
    }

    pub fn gte(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} >= {}", value.escaped()))
    }

    pub fn lte(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} <= {}", value.escaped()))
    }

    /// IN operator
    pub fn in_values(key: String, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        let values_str = values
            .into_iter()
            .map(|v| v.escaped())
            .collect::<Vec<_>>()
            .join(", ");
        Self(format!("{} in [{}]", key, values_str))
    }

    /// NOT IN operator
    pub fn not_in(key: String, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        let values_str = values
            .into_iter()
            .map(|v| v.escaped())
            .collect::<Vec<_>>()
            .join(", ");
        Self(format!("{} not in [{}]", key, values_str))
    }

    /// LIKE operator (string pattern matching)
    pub fn like(key: String, pattern: String) -> Self {
        Self(format!("{} like '{}'", key, pattern))
    }

    /// Array contains
    pub fn array_contains(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self(format!("array_contains({}, {})", key, value.escaped()))
    }

    /// Array contains all
    pub fn array_contains_all(key: String, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        let values_str = values
            .into_iter()
            .map(|v| v.escaped())
            .collect::<Vec<_>>()
            .join(", ");
        Self(format!("array_contains_all({}, [{}])", key, values_str))
    }

    /// Array contains any
    pub fn array_contains_any(key: String, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        let values_str = values
            .into_iter()
            .map(|v| v.escaped())
            .collect::<Vec<_>>()
            .join(", ");
        Self(format!("array_contains_any({}, [{}])", key, values_str))
    }

    /// Array length comparison
    pub fn array_length_eq(key: String, length: i32) -> Self {
        Self(format!("array_length({}) == {}", key, length))
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl TryFrom<CoreFilter<serde_json::Value>> for Filter {
    type Error = FilterError;
    fn try_from(value: CoreFilter<serde_json::Value>) -> Result<Self, Self::Error> {
        let value = match value {
            CoreFilter::Eq(k, val) => Filter::eq(k, val.try_into()?),
            CoreFilter::Gt(k, val) => Filter::gt(k, val.try_into()?),
            CoreFilter::Lt(k, val) => Filter::lt(k, val.try_into()?),
            CoreFilter::And(l, r) => Self::try_from(*l)?.and(Self::try_from(*r)?),
            CoreFilter::Or(l, r) => Self::try_from(*l)?.or(Self::try_from(*r)?),
        };

        Ok(value)
    }
}

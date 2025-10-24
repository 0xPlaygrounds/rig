use rig::vector_store::request::SearchFilter;

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

#[derive(Clone, Debug)]
pub struct Filter(String);

impl SearchFilter for Filter {
    type Value = MilvusValue;

    fn eq(key: String, value: Self::Value) -> Self {
        Self(format!("{key} == {}", value.escaped()))
    }

    fn gt(key: String, value: Self::Value) -> Self {
        Self(format!("{key} > {}", value.escaped()))
    }

    fn lt(key: String, value: Self::Value) -> Self {
        Self(format!("{key} < {}", value.escaped()))
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

    pub fn into_inner(self) -> String {
        self.0
    }
}

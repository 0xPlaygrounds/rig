use rig_core::vector_store::request::{
    DynamicSearchFilter, Filter as CoreFilter, FilterError, SearchFilter,
};
use serde::{Deserialize, Serialize};

/// A literal compared against a Weaviate property. Weaviate checks the
/// variant against the property's data type, so an `int` property needs
/// [`WeaviateValue::Int`] and a `number` property [`WeaviateValue::Number`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum WeaviateValue {
    /// A `text` or `uuid` value, sent as `valueText`.
    Text(String),
    /// An `int` value, sent as `valueInt`.
    Int(i64),
    /// A `number` value, sent as `valueNumber`.
    Number(f64),
    /// A `boolean` value, sent as `valueBoolean`.
    Boolean(bool),
    /// An RFC 3339 `date` value, sent as `valueDate`.
    Date(String),
}

impl WeaviateValue {
    fn key(&self) -> &'static str {
        match self {
            Self::Text(_) => "valueText",
            Self::Int(_) => "valueInt",
            Self::Number(_) => "valueNumber",
            Self::Boolean(_) => "valueBoolean",
            Self::Date(_) => "valueDate",
        }
    }

    /// The value as a GraphQL literal. Strings use JSON escaping, which is
    /// also valid GraphQL string syntax.
    fn literal(&self) -> Result<String, FilterError> {
        match self {
            Self::Text(text) | Self::Date(text) => serde_json::to_string(text)
                .map_err(|error| FilterError::Serialization(error.to_string())),
            Self::Int(int) => Ok(int.to_string()),
            Self::Number(number) if number.is_finite() => Ok(number.to_string()),
            Self::Number(number) => Err(FilterError::TypeError(format!(
                "Weaviate filters take finite numbers, got {number}"
            ))),
            Self::Boolean(boolean) => Ok(boolean.to_string()),
        }
    }
}

impl From<&str> for WeaviateValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for WeaviateValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<i64> for WeaviateValue {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<f64> for WeaviateValue {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

impl From<bool> for WeaviateValue {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

/// JSON integers become [`WeaviateValue::Int`] and other numbers
/// [`WeaviateValue::Number`]. Null, arrays and objects are rejected.
impl TryFrom<serde_json::Value> for WeaviateValue {
    type Error = FilterError;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        use serde_json::Value;

        match value {
            Value::String(text) => Ok(Self::Text(text)),
            Value::Bool(boolean) => Ok(Self::Boolean(boolean)),
            Value::Number(number) => match (number.as_i64(), number.as_f64()) {
                (Some(int), _) => Ok(Self::Int(int)),
                (None, Some(float)) => Ok(Self::Number(float)),
                (None, None) => Err(FilterError::TypeError(format!(
                    "{number} is outside Weaviate's number range"
                ))),
            },
            other => Err(FilterError::TypeError(format!(
                "Weaviate filters compare against text, numbers and booleans, got {other}"
            ))),
        }
    }
}

/// Comparison operators of a single-property condition.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum Operator {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanEqual,
    LessThan,
    LessThanEqual,
    Like,
    IsNull,
    ContainsAny,
    ContainsAll,
    ContainsNone,
}

impl Operator {
    fn name(self) -> &'static str {
        match self {
            Self::Equal => "Equal",
            Self::NotEqual => "NotEqual",
            Self::GreaterThan => "GreaterThan",
            Self::GreaterThanEqual => "GreaterThanEqual",
            Self::LessThan => "LessThan",
            Self::LessThanEqual => "LessThanEqual",
            Self::Like => "Like",
            Self::IsNull => "IsNull",
            Self::ContainsAny => "ContainsAny",
            Self::ContainsAll => "ContainsAll",
            Self::ContainsNone => "ContainsNone",
        }
    }
}

/// Operators that combine conditions.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum Combinator {
    And,
    Or,
    Not,
}

impl Combinator {
    fn name(self) -> &'static str {
        match self {
            Self::And => "And",
            Self::Or => "Or",
            Self::Not => "Not",
        }
    }
}

/// What a condition compares the property against.
#[derive(Clone, Debug, Serialize, Deserialize)]
enum Operand {
    One(WeaviateValue),
    Many(Vec<WeaviateValue>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Node {
    Condition {
        property: String,
        operator: Operator,
        operand: Operand,
    },
    Combination {
        combinator: Combinator,
        operands: Vec<Node>,
    },
}

impl Node {
    fn render(&self) -> Result<String, FilterError> {
        match self {
            Self::Condition {
                property,
                operator,
                operand,
            } => {
                let path = serde_json::to_string(property)
                    .map_err(|error| FilterError::Serialization(error.to_string()))?;
                let (key, literal) = match operand {
                    Operand::One(value) => (value.key(), value.literal()?),
                    Operand::Many(values) => list(values)?,
                };
                Ok(format!(
                    "{{path: [{path}], operator: {}, {key}: {literal}}}",
                    operator.name()
                ))
            }
            Self::Combination {
                combinator,
                operands,
            } => {
                let operands = operands
                    .iter()
                    .map(Node::render)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(format!(
                    "{{operator: {}, operands: [{}]}}",
                    combinator.name(),
                    operands.join(", ")
                ))
            }
        }
    }
}

/// The value key and literal of a list operand, whose values must share one
/// variant. An empty list is sent as an empty text list.
fn list(values: &[WeaviateValue]) -> Result<(&'static str, String), FilterError> {
    let key = values.first().map_or("valueText", WeaviateValue::key);
    if values.iter().any(|value| value.key() != key) {
        return Err(FilterError::TypeError(
            "a Weaviate list filter takes values of one type".into(),
        ));
    }
    let literals = values
        .iter()
        .map(WeaviateValue::literal)
        .collect::<Result<Vec<_>, _>>()?;
    Ok((key, format!("[{}]", literals.join(", "))))
}

/// A Weaviate `where` filter over a collection's properties.
///
/// [`InsertDocuments`](rig_core::vector_store::InsertDocuments) stores each
/// document as JSON text in the `document` property, so conditions on other
/// properties apply to collections whose objects carry them. Property names
/// and values are escaped when the filter is sent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeaviateFilter(Node);

impl WeaviateFilter {
    fn condition(property: impl AsRef<str>, operator: Operator, operand: Operand) -> Self {
        Self(Node::Condition {
            property: property.as_ref().to_owned(),
            operator,
            operand,
        })
    }

    fn combine(combinator: Combinator, operands: Vec<Self>) -> Self {
        Self(Node::Combination {
            combinator,
            operands: operands.into_iter().map(|filter| filter.0).collect(),
        })
    }

    /// Matches objects whose property differs from `value`.
    pub fn ne(property: impl AsRef<str>, value: WeaviateValue) -> Self {
        Self::condition(property, Operator::NotEqual, Operand::One(value))
    }

    /// Matches objects whose property is at least `value`.
    pub fn gte(property: impl AsRef<str>, value: WeaviateValue) -> Self {
        Self::condition(property, Operator::GreaterThanEqual, Operand::One(value))
    }

    /// Matches objects whose property is at most `value`.
    pub fn lte(property: impl AsRef<str>, value: WeaviateValue) -> Self {
        Self::condition(property, Operator::LessThanEqual, Operand::One(value))
    }

    /// Matches text against a pattern where `*` is any run of characters and
    /// `?` one character.
    pub fn like(property: impl AsRef<str>, pattern: impl Into<String>) -> Self {
        Self::condition(
            property,
            Operator::Like,
            Operand::One(WeaviateValue::Text(pattern.into())),
        )
    }

    /// Matches objects whose property is null (`true`) or set (`false`). The
    /// collection must index null state.
    pub fn is_null(property: impl AsRef<str>, null: bool) -> Self {
        Self::condition(
            property,
            Operator::IsNull,
            Operand::One(WeaviateValue::Boolean(null)),
        )
    }

    /// Matches objects whose property contains at least one of `values`.
    pub fn contains_any(property: impl AsRef<str>, values: Vec<WeaviateValue>) -> Self {
        Self::condition(property, Operator::ContainsAny, Operand::Many(values))
    }

    /// Matches objects whose property contains every one of `values`.
    pub fn contains_all(property: impl AsRef<str>, values: Vec<WeaviateValue>) -> Self {
        Self::condition(property, Operator::ContainsAll, Operand::Many(values))
    }

    /// Matches objects whose property contains none of `values`.
    pub fn contains_none(property: impl AsRef<str>, values: Vec<WeaviateValue>) -> Self {
        Self::condition(property, Operator::ContainsNone, Operand::Many(values))
    }

    /// Matches objects this filter does not match.
    pub fn not(self) -> Self {
        Self::combine(Combinator::Not, vec![self])
    }

    /// The filter as a GraphQL `where` argument value.
    ///
    /// # Errors
    ///
    /// When a number is not finite or a list mixes value types.
    pub fn to_graphql(&self) -> Result<String, FilterError> {
        self.0.render()
    }
}

impl SearchFilter for WeaviateFilter {
    type Value = WeaviateValue;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self::condition(key, Operator::Equal, Operand::One(value))
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self::condition(key, Operator::GreaterThan, Operand::One(value))
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self::condition(key, Operator::LessThan, Operand::One(value))
    }

    fn and(self, rhs: Self) -> Self {
        Self::combine(Combinator::And, vec![self, rhs])
    }

    fn or(self, rhs: Self) -> Self {
        Self::combine(Combinator::Or, vec![self, rhs])
    }
}

impl DynamicSearchFilter for WeaviateFilter {
    fn from_dynamic_filter(filter: CoreFilter<serde_json::Value>) -> Result<Self, FilterError> {
        filter.try_interpret(WeaviateValue::try_from)
    }
}

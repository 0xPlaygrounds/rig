use qdrant_client::qdrant::{
    Condition, FieldCondition, Filter, IsEmptyCondition, IsNullCondition, Match, Range,
    condition::ConditionOneOf, r#match::MatchValue,
};
use rig_core::vector_store::request::{FilterError, SearchFilter};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Qdrant metadata filter built as a JSON condition tree and translated into a
/// Qdrant `Filter` by [`QdrantFilter::interpret`].
///
/// Use this as the filter type for [`rig_core::vector_store::request::VectorSearchRequest`]
/// when querying [`crate::QdrantVectorStore`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantFilter(serde_json::Value);

impl SearchFilter for QdrantFilter {
    type Value = serde_json::Value;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        let key = key.as_ref().to_owned();

        Self(json!({
            "key": key,
            "match": {
                "value": value
            }
        }))
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        let key = key.as_ref().to_owned();

        Self(json!({
            "key": key,
            "range": {
                "gt": value
            }
        }))
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        let key = key.as_ref().to_owned();

        Self(json!({
            "key": key,
            "range": {
                "lt": value
            }
        }))
    }

    fn and(self, rhs: Self) -> Self {
        Self(json!({ "must": [ self.0, rhs.0 ]}))
    }

    fn or(self, rhs: Self) -> Self {
        Self(json!({ "should": [ self.0, rhs.0 ]}))
    }
}

impl QdrantFilter {
    pub fn not(self) -> Self {
        Self(json!({ "must_not": [ self.0 ]}))
    }
    pub fn into_inner(self) -> serde_json::Value {
        self.0
    }

    pub fn exists(key: &str) -> Self {
        Self(json!({ "key": key, "is_null": { "value": false } }))
    }

    pub fn is_null(key: &str) -> Self {
        Self(json!({ "key": key, "is_null": { "value": true } }))
    }

    pub fn is_empty(key: &str) -> Self {
        Self(json!({ "is_empty": { "key": key } }))
    }

    /// Range filter matching `lo < value < hi`.
    pub fn range_exclusive(key: &str, lo: &serde_json::Value, hi: &serde_json::Value) -> Self {
        Self(json!({
            "key": key,
            "range": {
                "gt": lo,
                "lt": hi
            }
        }))
    }

    /// Range filter matching `lo < value <= hi`.
    pub fn range_lower_inclusive(
        key: &str,
        lo: &serde_json::Value,
        hi: &serde_json::Value,
    ) -> Self {
        Self(json!({
            "key": key,
            "range": {
                "gt": lo,
                "lte": hi
            }
        }))
    }

    /// Range filter matching `lo <= value < hi`.
    pub fn range_higher_inclusive(
        key: &str,
        lo: &serde_json::Value,
        hi: &serde_json::Value,
    ) -> Self {
        Self(json!({
            "key": key,
            "range": {
                "gte": lo,
                "lt": hi
            }
        }))
    }

    /// Range filter matching `lo <= value <= hi`.
    pub fn range_inclusive(key: &str, lo: &serde_json::Value, hi: &serde_json::Value) -> Self {
        Self(json!({
            "key": key,
            "range": {
                "gte": lo,
                "lte": hi
            }
        }))
    }

    /// Translates the condition tree into a Qdrant filter, returning `None` when
    /// the filter is null, empty, or contains no conditions. Range bounds are
    /// compared as `f64`, and match values accept strings, booleans, and integers.
    pub fn interpret(self) -> Result<Option<Filter>, FilterError> {
        use serde_json::Value::*;

        let value = self.into_inner();

        if let Null = value {
            Ok(None)
        } else if json!({}) == value {
            Ok(None)
        } else {
            fn to_match(value: serde_json::Value) -> Result<MatchValue, FilterError> {
                match value {
                    String(s) => Ok(MatchValue::Keyword(s)),
                    Bool(b) => Ok(MatchValue::Boolean(b)),
                    Number(n) => {
                        if let Some(as_int) = n.as_i64() {
                            Ok(MatchValue::Integer(as_int))
                        } else {
                            Err(FilterError::Expected {
                                expected: "Integer".into(),
                                got: n.to_string(),
                            })
                        }
                    }
                    _ => Err(FilterError::TypeError(value.to_string())),
                }
            }

            fn to_condition(value: serde_json::Value) -> Result<Condition, FilterError> {
                if let Some(is_empty) = value.get("is_empty") {
                    let key = is_empty
                        .get("key")
                        .and_then(|k| k.as_str())
                        .ok_or_else(|| FilterError::MissingField("key".into()))?
                        .to_string();

                    Ok(Condition {
                        condition_one_of: Some(
                            qdrant_client::qdrant::condition::ConditionOneOf::IsEmpty(
                                IsEmptyCondition { key },
                            ),
                        ),
                    })
                } else if let Some(is_null) = value.get("is_null") {
                    let is_null_value = is_null
                        .get("value")
                        .and_then(serde_json::Value::as_bool)
                        .ok_or_else(|| {
                        FilterError::Must("is_null".into(), "have a 'value' field".into())
                    })?;

                    let key = value
                        .get("key")
                        .and_then(|k| k.as_str())
                        .ok_or_else(|| {
                            FilterError::Must("is_null".into(), "have a 'key' field".into())
                        })?
                        .to_string();

                    if is_null_value {
                        Ok(Condition {
                            condition_one_of: Some(
                                qdrant_client::qdrant::condition::ConditionOneOf::IsNull(
                                    IsNullCondition { key },
                                ),
                            ),
                        })
                    } else {
                        let is_empty_condition = Condition {
                            condition_one_of: Some(ConditionOneOf::IsEmpty(IsEmptyCondition {
                                key,
                            })),
                        };

                        let filter = Filter {
                            must_not: vec![is_empty_condition],
                            ..Default::default()
                        };

                        Ok(Condition {
                            condition_one_of: Some(ConditionOneOf::Filter(filter)),
                        })
                    }
                } else if value.as_object().is_some_and(|o| {
                    o.contains_key("must") || o.contains_key("must_not") || o.contains_key("should")
                }) {
                    let filter = QdrantFilter(value).interpret()?;

                    Ok(Condition {
                        condition_one_of: filter.map(ConditionOneOf::Filter),
                    })
                } else if let Some(key) = value.get("key").and_then(|k| k.as_str()) {
                    let mut field_condition = FieldCondition {
                        key: key.to_string(),
                        ..Default::default()
                    };

                    if let Some(match_obj) = value.get("match")
                        && let Some(val) = match_obj.get("value")
                    {
                        field_condition.r#match = Some(Match {
                            match_value: Some(to_match(val.clone())?),
                        });
                    }

                    if let Some(range_obj) = value.get("range") {
                        let mut range = Range::default();

                        if let Some(gt) = range_obj.get("gt") {
                            range.gt = gt.as_f64();
                        }
                        if let Some(gte) = range_obj.get("gte") {
                            range.gte = gte.as_f64();
                        }
                        if let Some(lt) = range_obj.get("lt") {
                            range.lt = lt.as_f64();
                        }
                        if let Some(lte) = range_obj.get("lte") {
                            range.lte = lte.as_f64();
                        }

                        field_condition.range = Some(range);
                    }

                    Ok(Condition {
                        condition_one_of: Some(
                            qdrant_client::qdrant::condition::ConditionOneOf::Field(
                                field_condition,
                            ),
                        ),
                    })
                } else {
                    Err(FilterError::TypeError(value.to_string()))
                }
            }

            fn to_filter(value: serde_json::Value) -> Result<Option<Filter>, FilterError> {
                let mut filter = Filter::default();

                if value.get("key").or(value.get("is_empty")).is_some() {
                    let condition = to_condition(value)?;
                    filter.must.push(condition);
                    Ok(Some(filter))
                } else {
                    if let Some(must) = value.get("must")
                        && let Some(arr) = must.as_array()
                    {
                        let conditions: Vec<Condition> = arr
                            .iter()
                            .cloned()
                            .map(to_condition)
                            .collect::<Result<_, _>>()?;
                        filter.must.extend(conditions);
                    }

                    if let Some(should) = value.get("should")
                        && let Some(arr) = should.as_array()
                    {
                        let conditions: Vec<Condition> = arr
                            .iter()
                            .cloned()
                            .map(to_condition)
                            .collect::<Result<_, _>>()?;
                        filter.should.extend(conditions);
                    }

                    if let Some(must_not) = value.get("must_not")
                        && let Some(arr) = must_not.as_array()
                    {
                        let conditions: Vec<Condition> = arr
                            .iter()
                            .cloned()
                            .map(to_condition)
                            .collect::<Result<_, _>>()?;
                        filter.must_not.extend(conditions);
                    }

                    if filter.must.is_empty()
                        && filter.should.is_empty()
                        && filter.must_not.is_empty()
                    {
                        Ok(None)
                    } else {
                        Ok(Some(filter))
                    }
                }
            }

            to_filter(value)
        }
    }
}

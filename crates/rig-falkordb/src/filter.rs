use falkordb::FalkorValue;
use rig_core::vector_store::{
    VectorStoreError,
    request::{Filter, FilterError, SearchFilter},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A composable predicate on the matched node's top-level properties.
///
/// Values are bound as driver parameters. Property names are quoted and invalid
/// names are rejected when executing a search.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FalkorDBSearchFilter(Filter<Value>);

impl SearchFilter for FalkorDBSearchFilter {
    type Value = Value;

    fn eq(key: impl AsRef<str>, value: Value) -> Self {
        Self(Filter::eq(key, value))
    }

    fn gt(key: impl AsRef<str>, value: Value) -> Self {
        Self(Filter::gt(key, value))
    }

    fn lt(key: impl AsRef<str>, value: Value) -> Self {
        Self(Filter::lt(key, value))
    }

    fn and(self, rhs: Self) -> Self {
        Self(self.0.and(rhs.0))
    }

    fn or(self, rhs: Self) -> Self {
        Self(self.0.or(rhs.0))
    }
}

impl FalkorDBSearchFilter {
    pub(crate) fn compile(&self) -> Result<(String, Vec<(String, FalkorValue)>), VectorStoreError> {
        let mut params = Vec::new();
        let condition = compile(&self.0, &mut params)?;
        Ok((condition, params))
    }
}

fn compile(
    filter: &Filter<Value>,
    params: &mut Vec<(String, FalkorValue)>,
) -> Result<String, VectorStoreError> {
    match filter {
        Filter::Eq(key, value) | Filter::Gt(key, value) | Filter::Lt(key, value) => {
            let key = crate::quote_identifier(key)
                .map_err(|error| FilterError::TypeError(error.to_string()))?;
            if matches!(filter, Filter::Eq(_, Value::Null)) {
                return Ok(format!("node.{key} IS NULL"));
            }
            let op = match filter {
                Filter::Eq(..) => "=",
                Filter::Gt(..) => ">",
                _ => "<",
            };
            let name = format!("filter_{}", params.len());
            params.push((name.clone(), crate::json_to_falkor(value.clone())?));
            Ok(format!("node.{key} {op} ${name}"))
        }
        Filter::And(left, right) | Filter::Or(left, right) => {
            let left = compile(left, params)?;
            let right = compile(right, params)?;
            let op = if matches!(filter, Filter::And(..)) {
                "AND"
            } else {
                "OR"
            };
            Ok(format!("({left}) {op} ({right})"))
        }
    }
}

#[cfg(test)]
mod tests;

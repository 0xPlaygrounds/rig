//! TypeSafe's JSON protocol. Prefer the typed question constructors for application code.
use rig_core::completion::Cost;
use rig_core::json_utils::Lenient;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::collections::BTreeMap;

/// Model-independent input to an evaluation. [`crate::Jev`] supplies the model
/// from its configuration when encoding the full HTTP request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// The shared state evaluated independently by every question.
    pub state: Value,
    /// Serialized question object. Keeping the JSON intact preserves duplicate
    /// keys for validation and allows named structs to supply the shape.
    pub questions: Box<serde_json::value::RawValue>,
}

/// A question as represented by the HTTP API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Choice {
        instructions: Value,
        criteria: BTreeMap<String, Option<Value>>,
    },
    Score {
        instructions: Value,
        criteria: Vec<Value>,
    },
    Noul {
        instructions: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<BTreeMap<String, Value>>,
    },
}

/// An answer as represented by the HTTP API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    Score {
        score: f64,
        probabilities: BTreeMap<String, f64>,
        legend: BTreeMap<String, Value>,
        confidence: f64,
    },
    Noul {
        noul: f64,
    },
}

/// Rig token accounting; counters absent from Jev remain unknown.
pub use rig_core::completion::Usage;

/// Raw response, including the transport's request identifier. It
/// serializes in Jev's shape, which reads back as the same response, without
/// the transport's request identifier or the whole reply document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// The model identifier returned by the provider.
    pub model: String,
    /// Answer object kept intact for validation and typed Serde decoding.
    pub answers: Box<serde_json::value::RawValue>,
    /// Token accounting when reported by the provider.
    #[serde(
        default,
        deserialize_with = "reported_usage",
        serialize_with = "jev_usage"
    )]
    pub usage: Option<Usage>,
    #[serde(skip)]
    /// Transport request identifier for diagnostics.
    pub provider_request_id: Option<String>,
    #[serde(skip)]
    /// The whole reply document, including fields this type does not model.
    pub raw: Value,
}

/// Jev's `usage`: its token counters and, through OpenRouter, a `cost` in
/// USD. A counter or cost of another shape is unreported rather than failing
/// the reply, whose answers do not depend on it.
fn reported_usage<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Usage>, D::Error> {
    let usage = Option::<Value>::deserialize(deserializer)?;
    Ok(usage.filter(Value::is_object).map(|usage| {
        Usage::new()
            .input_tokens(usage.u64("input_tokens"))
            .output_tokens(usage.u64("output_tokens"))
            .cost(
                usage
                    .f64("cost")
                    .filter(|cost| cost.is_finite())
                    .map(Cost::from_total),
            )
    }))
}

/// `usage` in Jev's shape: its token counters and its cost's total.
fn jev_usage<S: Serializer>(usage: &Option<Usage>, serializer: S) -> Result<S::Ok, S::Error> {
    #[derive(Serialize)]
    struct JevUsage {
        #[serde(skip_serializing_if = "Option::is_none")]
        input_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cost: Option<f64>,
    }
    usage
        .map(|usage| JevUsage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cost: usage.cost.map(|cost| cost.total),
        })
        .serialize(serializer)
}

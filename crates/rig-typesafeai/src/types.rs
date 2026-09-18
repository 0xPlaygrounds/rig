//! TypeSafe's JSON protocol. Prefer the typed question constructors for application code.
use serde::{Deserialize, Serialize};
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

/// Raw response, including the transport's request identifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// The model identifier returned by the provider.
    pub model: String,
    /// Answer object kept intact for validation and typed Serde decoding.
    pub answers: Box<serde_json::value::RawValue>,
    #[serde(default)]
    /// Token accounting when reported by the provider.
    pub usage: Option<Usage>,
    #[serde(skip)]
    /// Transport request identifier for diagnostics.
    pub provider_request_id: Option<String>,
}

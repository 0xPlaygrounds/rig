//! A custom agent runtime companion crate for Rig, demonstrating how to build an orchestrator
//! with heavy telemetry and explicit tracing for maximum observability.

pub mod agent;

pub use agent::{CustomAgent, CustomAgentBuilder, CustomAgentError};

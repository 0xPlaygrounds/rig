//! This module primarily concerns being able to orchestrate telemetry across a given pipeline or workflow.
//! This includes tracing, being able to send traces to an OpenTelemetry collector, setting up your
//! agents with the correct tracing style so you can emit the right traces for platforms like Langfuse,
//! and more.

use serde::Serialize;

pub trait ProviderRequestExt {
    type InputMessage: Serialize;

    fn get_input_messages(&self) -> Vec<Self::InputMessage>;
    fn get_model_name(&self) -> String;
    fn get_prompt(&self) -> Option<String>;
}

pub trait ProviderResponseExt {
    type OutputMessage: Serialize;
    type Usage: Serialize;

    fn get_output_messages(&self) -> Vec<Self::OutputMessage>;
    fn get_text_response(&self) -> Option<String>;
    fn get_usage(&self) -> Option<Self::Usage>;
}

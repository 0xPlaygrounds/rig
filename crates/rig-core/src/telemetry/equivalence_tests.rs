//! Every span the unified builder opens matches what the replaced completion
//! and modality builders opened, captured before they were removed: name,
//! target, parent, declared fields, and recorded values, case by case.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tracing::Subscriber;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::{Layer, Registry, registry::LookupSpan};

use super::*;
use crate::completion::{CompletionRequestBuilder, CompletionResponse};
use crate::embeddings::EmbeddingResponse;
use crate::error::ProviderError;
use crate::operation::{Completion, Embedding, Rerank, RerankRequest, Transcription};
use crate::rerank::RerankResponse;
use crate::streaming::{StreamEvent, StreamFinal};
use crate::transcription::{TranscriptionRequest, TranscriptionResponse};
use crate::wire::Operation;

/// One JSON line per case, in the order [`cases`] runs them.
const EXPECTED: &str = r#"{"case":"fresh completion chat","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"chat","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"chat","gen_ai.provider.name":"prov","gen_ai.request.model":"model"}}]}
{"case":"fresh completion chat + system recorded","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"chat","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"chat","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.system_instructions":"[{\"type\":\"text\",\"content\":\"sys\"}]"}}]}
{"case":"adopted completion chat","spans":[{"fields":["rig.completion_parent","gen_ai.operation.name","gen_ai.system_instructions","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"agent_chat","parent":null,"target":"runtime","values":{"gen_ai.operation.name":"chat","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.system_instructions":"[{\"type\":\"text\",\"content\":\"sys\"}]","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3,"rig.completion_parent":true}}]}
{"case":"fresh completion chat_streaming","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"chat_streaming","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"chat_streaming","gen_ai.provider.name":"prov","gen_ai.request.model":"model"}}]}
{"case":"fresh completion chat_streaming + system recorded","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"chat_streaming","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"chat_streaming","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.system_instructions":"[{\"type\":\"text\",\"content\":\"sys\"}]"}}]}
{"case":"adopted completion chat_streaming","spans":[{"fields":["rig.completion_parent","gen_ai.operation.name","gen_ai.system_instructions","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"agent_chat","parent":null,"target":"runtime","values":{"gen_ai.operation.name":"chat_streaming","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.system_instructions":"[{\"type\":\"text\",\"content\":\"sys\"}]","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3,"rig.completion_parent":true}}]}
{"case":"fresh completion generate_content","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"generate_content","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"generate_content","gen_ai.provider.name":"prov","gen_ai.request.model":"model"}}]}
{"case":"fresh completion generate_content + system recorded","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"generate_content","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"generate_content","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.system_instructions":"[{\"type\":\"text\",\"content\":\"sys\"}]"}}]}
{"case":"adopted completion generate_content","spans":[{"fields":["rig.completion_parent","gen_ai.operation.name","gen_ai.system_instructions","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"agent_chat","parent":null,"target":"runtime","values":{"gen_ai.operation.name":"generate_content","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.system_instructions":"[{\"type\":\"text\",\"content\":\"sys\"}]","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3,"rig.completion_parent":true}}]}
{"case":"fresh completion interactions","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"interactions","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"interactions","gen_ai.provider.name":"prov","gen_ai.request.model":"model"}}]}
{"case":"fresh completion interactions + system recorded","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"interactions","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"interactions","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.system_instructions":"[{\"type\":\"text\",\"content\":\"sys\"}]"}}]}
{"case":"adopted completion interactions","spans":[{"fields":["rig.completion_parent","gen_ai.operation.name","gen_ai.system_instructions","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"agent_chat","parent":null,"target":"runtime","values":{"gen_ai.operation.name":"interactions","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.system_instructions":"[{\"type\":\"text\",\"content\":\"sys\"}]","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3,"rig.completion_parent":true}}]}
{"case":"fresh completion interactions_streaming","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"interactions_streaming","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"interactions_streaming","gen_ai.provider.name":"prov","gen_ai.request.model":"model"}}]}
{"case":"fresh completion interactions_streaming + system recorded","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"interactions_streaming","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"interactions_streaming","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.system_instructions":"[{\"type\":\"text\",\"content\":\"sys\"}]"}}]}
{"case":"adopted completion interactions_streaming","spans":[{"fields":["rig.completion_parent","gen_ai.operation.name","gen_ai.system_instructions","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"agent_chat","parent":null,"target":"runtime","values":{"gen_ai.operation.name":"interactions_streaming","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.system_instructions":"[{\"type\":\"text\",\"content\":\"sys\"}]","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3,"rig.completion_parent":true}}]}
{"case":"fresh completion under an ordinary ambient span is its child","spans":[{"fields":[],"name":"ambient","parent":null,"target":"runtime","values":{}},{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"chat","parent":"ambient","target":"rig::completions","values":{"gen_ai.operation.name":"chat","gen_ai.provider.name":"prov","gen_ai.request.model":"model"}}]}
{"case":"fresh completion chat + system not recorded","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"chat","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"chat","gen_ai.provider.name":"prov","gen_ai.request.model":"model"}}]}
{"case":"modality embeddings","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens"],"name":"embeddings","parent":null,"target":"rig::modalities","values":{"gen_ai.operation.name":"embeddings","gen_ai.provider.name":"prov","gen_ai.request.model":"model"}}]}
{"case":"modality rerank","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens"],"name":"rerank","parent":null,"target":"rig::modalities","values":{"gen_ai.operation.name":"rerank","gen_ai.provider.name":"prov","gen_ai.request.model":"model"}}]}
{"case":"modality transcription","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens"],"name":"transcription","parent":null,"target":"rig::modalities","values":{"gen_ai.operation.name":"transcription","gen_ai.provider.name":"prov","gen_ai.request.model":"model"}}]}
{"case":"modality image_generation","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens"],"name":"image_generation","parent":null,"target":"rig::modalities","values":{"gen_ai.operation.name":"image_generation","gen_ai.provider.name":"prov","gen_ai.request.model":"model"}}]}
{"case":"modality audio_generation","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens"],"name":"audio_generation","parent":null,"target":"rig::modalities","values":{"gen_ai.operation.name":"audio_generation","gen_ai.provider.name":"prov","gen_ai.request.model":"model"}}]}
{"case":"modality span under a completion parent stays fresh","spans":[{"fields":["rig.completion_parent","gen_ai.operation.name","gen_ai.system_instructions","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"agent_chat","parent":null,"target":"runtime","values":{"gen_ai.operation.name":"chat","rig.completion_parent":true}},{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens"],"name":"embeddings","parent":"agent_chat","target":"rig::modalities","values":{"gen_ai.operation.name":"embeddings","gen_ai.provider.name":"prov","gen_ai.request.model":"model"}}]}
{"case":"instrument_modality ok","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens"],"name":"embeddings","parent":null,"target":"rig::modalities","values":{"gen_ai.operation.name":"embeddings","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.response.id":"emb_id","gen_ai.response.model":"emb_model","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3}}]}
{"case":"instrument_modality err","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens"],"name":"embeddings","parent":null,"target":"rig::modalities","values":{"gen_ai.operation.name":"embeddings","gen_ai.provider.name":"prov","gen_ai.request.model":"model"}}]}
{"case":"completion operation span+record (message id fallback)","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"chat","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"chat","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.response.id":"msg_1","gen_ai.response.model":"resp_model","gen_ai.system_instructions":"[{\"type\":\"text\",\"content\":\"sys\"}]","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3}}]}
{"case":"completion operation streaming span+record_event","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"chat_streaming","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"chat_streaming","gen_ai.provider.name":"prov","gen_ai.request.model":"override","gen_ai.response.id":"resp_1","gen_ai.response.model":"m2","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3}}]}
{"case":"embedding operation span+record","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens"],"name":"embeddings","parent":null,"target":"rig::modalities","values":{"gen_ai.operation.name":"embeddings","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.response.id":"emb_id","gen_ai.response.model":"emb_model","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3}}]}
{"case":"rerank operation span+record","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens"],"name":"rerank","parent":null,"target":"rig::modalities","values":{"gen_ai.operation.name":"rerank","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.response.id":"rr_id","gen_ai.response.model":"rr_model","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3}}]}
{"case":"transcription operation span+record","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens"],"name":"transcription","parent":null,"target":"rig::modalities","values":{"gen_ai.operation.name":"transcription","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.response.id":"tr_id","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3}}]}
{"case":"span combinator on a native response","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"chat","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"chat","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.response.id":"native_id","gen_ai.response.model":"native_model","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3}}]}"#;

#[derive(Clone, Default)]
struct Spans {
    spans: Arc<Mutex<Vec<Value>>>,
    index: Arc<Mutex<BTreeMap<u64, usize>>>,
}

struct Values<'a>(&'a mut serde_json::Map<String, Value>);

impl Visit for Values<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().into(), json!(format!("{value:?}")));
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().into(), json!(value));
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().into(), json!(value));
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.0.insert(field.name().into(), json!(value));
    }
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.0.insert(field.name().into(), json!(value));
    }
}

impl<S> Layer<S> for Spans
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let metadata = attrs.metadata();
        let parent = match attrs.parent() {
            Some(parent) => ctx.span(parent).map(|span| span.name()),
            None if attrs.is_contextual() => ctx.lookup_current().map(|span| span.name()),
            None => None,
        };
        let mut values = serde_json::Map::new();
        attrs.record(&mut Values(&mut values));
        let fields: Vec<&str> = metadata.fields().iter().map(|field| field.name()).collect();
        let mut spans = self.spans.lock().expect("spans");
        self.index
            .lock()
            .expect("index")
            .insert(id.into_u64(), spans.len());
        spans.push(json!({
            "name": metadata.name(),
            "target": metadata.target(),
            "fields": fields,
            "parent": parent,
            "values": Value::Object(values),
        }));
    }

    fn on_record(&self, id: &Id, record: &Record<'_>, _: Context<'_, S>) {
        let Some(&index) = self.index.lock().expect("index").get(&id.into_u64()) else {
            return;
        };
        let mut spans = self.spans.lock().expect("spans");
        if let Some(values) = spans[index]["values"].as_object_mut() {
            record.record(&mut Values(values));
        }
    }
}

fn usage() -> Usage {
    Usage {
        input_tokens: Some(10),
        output_tokens: Some(0),
        reasoning_tokens: Some(3),
        ..Usage::default()
    }
}

fn run(case: &str, out: &mut Vec<Value>, body: impl FnOnce()) {
    let spans = Spans::default();
    tracing::subscriber::with_default(Registry::default().with(spans.clone()), body);
    let captured = spans.spans.lock().expect("spans").clone();
    out.push(json!({ "case": case, "spans": captured }));
}

fn completion_parent() -> tracing::Span {
    crate::completion_parent_span!(
        target: "runtime",
        name: "agent_chat",
        operation: "chat",
        system_instructions: Option::<&str>::None,
    )
}

fn cases() -> Vec<Value> {
    let mut out = Vec::new();
    let completions = [
        ("chat", GenAiOperation::Chat),
        ("chat_streaming", GenAiOperation::ChatStreaming),
        ("generate_content", GenAiOperation::GenerateContent),
        ("interactions", GenAiOperation::Interactions),
        (
            "interactions_streaming",
            GenAiOperation::InteractionsStreaming,
        ),
    ];
    for (name, operation) in completions {
        run(&format!("fresh completion {name}"), &mut out, || {
            SpanBuilder::new("prov", "model", operation).build();
        });
        run(
            &format!("fresh completion {name} + system recorded"),
            &mut out,
            || {
                SpanBuilder::new("prov", "model", operation)
                    .system_instructions(Some("sys"), true)
                    .build();
            },
        );
        run(&format!("adopted completion {name}"), &mut out, || {
            let parent = completion_parent();
            let _entered = parent.enter();
            SpanBuilder::new("prov", "model", operation)
                .system_instructions(Some("sys"), true)
                .build()
                .record_token_usage(&usage());
        });
    }
    run(
        "fresh completion under an ordinary ambient span is its child",
        &mut out,
        || {
            let ambient = tracing::info_span!(target: "runtime", "ambient");
            let _entered = ambient.enter();
            SpanBuilder::new("prov", "model", GenAiOperation::Chat).build();
        },
    );
    run(
        "fresh completion chat + system not recorded",
        &mut out,
        || {
            SpanBuilder::new("prov", "model", GenAiOperation::Chat)
                .system_instructions(Some("sys"), false)
                .build();
        },
    );
    let modalities = [
        ("embeddings", GenAiOperation::Embeddings),
        ("rerank", GenAiOperation::Rerank),
        ("transcription", GenAiOperation::Transcription),
        ("image_generation", GenAiOperation::ImageGeneration),
        ("audio_generation", GenAiOperation::AudioGeneration),
    ];
    for (name, operation) in modalities {
        run(&format!("modality {name}"), &mut out, || {
            SpanBuilder::new("prov", "model", operation).build();
        });
    }
    run(
        "modality span under a completion parent stays fresh",
        &mut out,
        || {
            let parent = completion_parent();
            let _entered = parent.enter();
            SpanBuilder::new("prov", "model", GenAiOperation::Embeddings).build();
        },
    );
    run("instrument_modality ok", &mut out, || {
        let response = EmbeddingResponse::new(vec![], "prov")
            .with_response_id("emb_id")
            .with_model("emb_model")
            .with_usage(usage());
        let result = futures::executor::block_on(instrument_modality::<Embedding, _>(
            "prov",
            "model",
            async move { Ok::<_, ProviderError>(response) },
        ));
        assert!(result.is_ok());
    });
    run("instrument_modality err", &mut out, || {
        let result = futures::executor::block_on(instrument_modality::<Embedding, _>(
            "prov",
            "model",
            async move { Err::<EmbeddingResponse, _>(ProviderError::Provider("no".into())) },
        ));
        assert!(result.is_err());
    });
    run(
        "completion operation span+record (message id fallback)",
        &mut out,
        || {
            let request = CompletionRequestBuilder::unbound("hi")
                .preamble("sys".into())
                .record_content_telemetry(true)
                .build();
            let span = Completion::span(
                "prov",
                Some("model"),
                Completion::telemetry(false),
                &request,
            );
            let response = CompletionResponse::new(vec![], usage(), "prov", Value::Null)
                .with_message_id("msg_1")
                .with_model("resp_model");
            Completion::record(&span, &response);
        },
    );
    run(
        "completion operation streaming span+record_event",
        &mut out,
        || {
            let request = CompletionRequestBuilder::unbound("hi")
                .model("override")
                .build();
            let span =
                Completion::span("prov", Some("model"), Completion::telemetry(true), &request);
            let terminal = StreamFinal::new("prov", usage(), Value::Null)
                .with_response_id("resp_1")
                .with_message_id("msg_1")
                .with_model("m2");
            Completion::record_event(&span, &StreamEvent::Final(terminal));
        },
    );
    run("embedding operation span+record", &mut out, || {
        let span = Embedding::span(
            "prov",
            Some("model"),
            Embedding::telemetry(false),
            &vec!["a".to_owned()],
        );
        let response = EmbeddingResponse::new(vec![], "prov")
            .with_response_id("emb_id")
            .with_model("emb_model")
            .with_usage(usage());
        Embedding::record(&span, &response);
    });
    run("rerank operation span+record", &mut out, || {
        let request = RerankRequest {
            query: "q".into(),
            documents: vec!["d".into()],
        };
        let span = Rerank::span("prov", Some("model"), Rerank::telemetry(false), &request);
        let response = RerankResponse::new(vec![], "prov")
            .with_response_id("rr_id")
            .with_model("rr_model")
            .with_usage(usage());
        Rerank::record(&span, &response);
    });
    run("transcription operation span+record", &mut out, || {
        let request = TranscriptionRequest {
            data: vec![1],
            filename: "a.wav".into(),
            language: None,
            prompt: None,
            temperature: None,
            additional_params: None,
        };
        let span = Transcription::span(
            "prov",
            Some("model"),
            Transcription::telemetry(false),
            &request,
        );
        let response = TranscriptionResponse::new("text", "prov")
            .with_response_id("tr_id")
            .with_usage(usage());
        Transcription::record(&span, &response);
    });
    run("span combinator on a native response", &mut out, || {
        SpanBuilder::new("prov", "model", GenAiOperation::Chat)
            .build()
            .record_response(Some("native_id"), Some("native_model"), &usage());
    });
    out
}

#[test]
fn spans_match_the_replaced_builders() {
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    let actual = cases();
    let expected: Vec<Value> = EXPECTED
        .lines()
        .map(|line| serde_json::from_str(line).expect("expected case"))
        .collect();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual, expected, "{}", expected["case"]);
    }
}

/// System instructions only belong on completion spans; a modality span
/// ignores them.
#[test]
fn modality_spans_ignore_system_instructions() {
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    let mut with = Vec::new();
    run("with", &mut with, || {
        SpanBuilder::new("prov", "model", GenAiOperation::Rerank)
            .system_instructions(Some("sys"), true)
            .build();
    });
    let mut without = Vec::new();
    run("with", &mut without, || {
        SpanBuilder::new("prov", "model", GenAiOperation::Rerank).build();
    });
    assert_eq!(with, without);
}

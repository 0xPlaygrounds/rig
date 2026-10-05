//! Every span the unified builder opens matches what the replaced completion
//! and modality builders opened, captured before they were removed: name,
//! target, parent, declared fields, and recorded values, case by case.

use serde_json::{Value, json};

use super::*;
use crate::completion::CompletionRequest;
use crate::driver::{Exchange, Local, Model, Opened, Opening, Transport};
use crate::embeddings::EmbeddingResponse;
use crate::error::{EncodeError, ProviderError};
use crate::operation::{Completion, Embedding, Finish, Rerank, RerankRequest, Transcription};
use crate::rerank::RerankResponse;
use crate::test_utils::{CapturedSpan, TraceCapture};
use crate::transcription::{TranscriptionRequest, TranscriptionResponse};
use crate::wire::{Decoder, Descriptor, Flow, Fold, Mode, Out, Wire, WireEvent};
use futures::StreamExt;

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
{"case":"completion operation span+record","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"chat","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"chat","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.response.model":"resp_model","gen_ai.system_instructions":"[{\"type\":\"text\",\"content\":\"sys\"}]","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3}}]}
{"case":"completion operation streaming span+record_event","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"chat_streaming","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"chat_streaming","gen_ai.provider.name":"prov","gen_ai.request.model":"override","gen_ai.response.id":"resp_1","gen_ai.response.model":"m2","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3}}]}
{"case":"embedding operation span+record","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens"],"name":"embeddings","parent":null,"target":"rig::modalities","values":{"gen_ai.operation.name":"embeddings","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.response.id":"emb_id","gen_ai.response.model":"emb_model","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3}}]}
{"case":"rerank operation span+record","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens"],"name":"rerank","parent":null,"target":"rig::modalities","values":{"gen_ai.operation.name":"rerank","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.response.id":"rr_id","gen_ai.response.model":"rr_model","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3}}]}
{"case":"transcription operation span+record","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens"],"name":"transcription","parent":null,"target":"rig::modalities","values":{"gen_ai.operation.name":"transcription","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.response.id":"tr_id","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3}}]}
{"case":"span combinator on a native response","spans":[{"fields":["gen_ai.operation.name","gen_ai.provider.name","gen_ai.request.model","gen_ai.system_instructions","gen_ai.response.id","gen_ai.response.model","rig.provider_request_id","gen_ai.usage.input_tokens","gen_ai.usage.output_tokens","gen_ai.usage.cache_read.input_tokens","gen_ai.usage.cache_creation.input_tokens","gen_ai.usage.tool_use_prompt_tokens","gen_ai.usage.reasoning_tokens","gen_ai.input.messages","gen_ai.output.messages"],"name":"chat","parent":null,"target":"rig::completions","values":{"gen_ai.operation.name":"chat","gen_ai.provider.name":"prov","gen_ai.request.model":"model","gen_ai.response.id":"native_id","gen_ai.response.model":"native_model","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":0,"gen_ai.usage.reasoning_tokens":3}}]}"#;

fn usage() -> Usage {
    Usage {
        input_tokens: Some(10),
        output_tokens: Some(0),
        reasoning_tokens: Some(3),
        ..Usage::default()
    }
}

fn run(case: &str, out: &mut Vec<Value>, body: impl FnOnce()) {
    let capture = TraceCapture::default();
    tracing::subscriber::with_default(capture.subscriber(), body);
    let spans: Vec<Value> = capture.spans().iter().map(CapturedSpan::summary).collect();
    out.push(json!({ "case": case, "spans": spans }));
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
    run("completion operation span+record", &mut out, || {
        let request = CompletionRequest::new("hi")
            .preamble("sys")
            .record_content_telemetry(true);
        let end = Finish {
            usage: usage(),
            model: Some("resp_model".into()),
            ..Finish::default()
        };
        let fold = crate::test_utils::fold_for(&request, &Scripted(end.clone()), Mode::Unary);
        fold.finish(end, reply())
            .expect("the fold records its response");
    });
    run(
        "completion operation streaming span+record_event",
        &mut out,
        || {
            let request = CompletionRequest::new("hi").model("override");
            let end = Finish {
                usage: usage(),
                response_id: Some("resp_1".into()),
                model: Some("m2".into()),
                ..Finish::default()
            };
            // The streamed call's span records the response it finishes with.
            let mut stream = Model::new(Scripted(end.clone()), Scripted(end))
                .stream(request)
                .expect("a scripted stream opens");
            futures::executor::block_on(async {
                while stream.next().await.is_some() {}
                stream.finish().await.expect("the scripted reply ends")
            });
        },
    );
    run("embedding operation span+record", &mut out, || {
        let wire = Local::<Embedding>::new("prov").with_id("model");
        let fold = crate::test_utils::fold_for(&Vec::new(), &wire, Mode::Unary);
        let response = EmbeddingResponse {
            provider: "prov".into(),
            response_id: Some("emb_id".into()),
            model: Some("emb_model".into()),
            usage: usage(),
            ..EmbeddingResponse::new(vec![])
        };
        fold.finish(response, reply())
            .expect("the fold records its response");
    });
    run("rerank operation span+record", &mut out, || {
        let request = RerankRequest {
            query: "q".into(),
            documents: vec!["d".into()],
        };
        let wire = Local::<Rerank>::new("prov").with_id("model");
        let fold = crate::test_utils::fold_for(&request, &wire, Mode::Unary);
        let response = RerankResponse {
            provider: "prov".into(),
            response_id: Some("rr_id".into()),
            model: Some("rr_model".into()),
            usage: usage(),
            ..RerankResponse::new(vec![])
        };
        fold.finish(response, reply())
            .expect("the fold records its response");
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
        let wire = Local::<Transcription>::new("prov").with_id("model");
        let fold = crate::test_utils::fold_for(&request, &wire, Mode::Unary);
        let response = TranscriptionResponse {
            provider: "prov".into(),
            response_id: Some("tr_id".into()),
            usage: usage(),
            ..TranscriptionResponse::new("text")
        };
        fold.finish(response, reply())
            .expect("the fold records its response");
    });
    run("span combinator on a native response", &mut out, || {
        SpanBuilder::new("prov", "model", GenAiOperation::Chat)
            .build()
            .record_response(Some("native_id"), Some("native_model"), &usage());
    });
    out
}

/// A reply that carried nothing beyond its events.
fn reply() -> crate::wire::Reply {
    crate::wire::Reply {
        provider: "prov".to_owned(),
        raw: Value::Null,
        provider_request_id: None,
    }
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

/// A completion wire named `prov` for `model`, whose transport answers with
/// one scripted end.
#[derive(Clone, Debug)]
struct Scripted(Finish);

impl crate::completion::ReplayTarget for Scripted {
    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static("prov.chat")
    }

    fn provider(&self) -> &str {
        "prov"
    }

    fn model(&self) -> &str {
        "model"
    }

    fn accepts(&self, _model: &str) -> crate::completion::Accepts {
        crate::completion::Accepts::ALL
    }
}

impl Wire for Scripted {
    type Op = Completion;
    type Payload = ();
    type Frame = Finish;
    type Decoder<'id> = Ends;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new("prov").model("model").replay(self)
    }

    fn encode(&self, _request: CompletionRequest, _mode: Mode) -> Result<(), EncodeError> {
        Ok(())
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        Ends
    }
}

impl Transport<Scripted> for Scripted {
    fn send(&self, _payload: (), _exchange: Exchange) -> Opening<Finish> {
        Opening::ready(Opened::new(futures::stream::iter([Ok(self.0.clone())])))
    }
}

/// Ends the reply with the scripted end.
struct Ends;

impl<'id> Decoder<'id, Completion, Finish> for Ends {
    type Event = Finish;

    fn classify(&self, frame: Finish) -> WireEvent<Finish> {
        WireEvent::Known(frame)
    }

    fn decode(&mut self, end: Finish, out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        Ok(out.end(end))
    }
}

//! Transport-boundary middleware on the erased HTTP client.
//!
//! Demonstrates the three `HttpMiddleware` moments on a real provider call:
//! injecting a per-request `anthropic-beta` header, logging the serialized
//! request body, and reading rate-limit / request-id response headers — on a
//! streaming call, before the stream is consumed. Requires `ANTHROPIC_API_KEY`.

use anyhow::{Context, Result};
use rig::http_client::{BoxedHttpClient, HeaderMap, HeaderValue, HttpMiddleware, Method, Uri};
use rig::prelude::*;
use rig::providers::anthropic;
use rig::wasm_compat::WasmBoxedFuture;

/// Adds a beta header to every outgoing request and prints the wire traffic
/// the semantic layer never shows: the serialized body and the response's
/// transport metadata.
struct WireLogger;

impl HttpMiddleware for WireLogger {
    fn before_request_headers<'a>(
        &'a self,
        _method: &'a Method,
        _uri: &'a Uri,
        headers: &'a mut HeaderMap,
    ) -> WasmBoxedFuture<'a, rig::http_client::Result<()>> {
        Box::pin(async move {
            headers.insert(
                "anthropic-beta",
                HeaderValue::from_static("token-efficient-tools-2025-02-19"),
            );
            Ok(())
        })
    }

    fn before_request_body<'a>(
        &'a self,
        method: &'a Method,
        uri: &'a Uri,
        _headers: &'a HeaderMap,
        body: bytes::Bytes,
    ) -> WasmBoxedFuture<'a, rig::http_client::Result<bytes::Bytes>> {
        Box::pin(async move {
            println!("→ {method} {uri} ({} byte payload)", body.len());
            Ok(body)
        })
    }

    fn after_response<'a>(
        &'a self,
        _method: &'a Method,
        _uri: &'a Uri,
        status: http::StatusCode,
        headers: &'a HeaderMap,
    ) -> WasmBoxedFuture<'a, rig::http_client::Result<()>> {
        Box::pin(async move {
            // Runs before any of the (possibly streaming) body is consumed.
            let request_id = headers
                .get("request-id")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("<none>");
            println!("← {status}, request-id: {request_id}");
            for (name, value) in headers {
                if name.as_str().starts_with("anthropic-ratelimit-") {
                    println!("  {name}: {}", value.to_str().unwrap_or("<binary>"));
                }
            }
            Ok(())
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").context("ANTHROPIC_API_KEY is not set")?;

    // Erase the default transport, then attach the middleware — no provider
    // code involved; the same handle could back every provider a host builds.
    let http_client = BoxedHttpClient::new(rig::http_client::ReqwestClient::default())
        .with_middleware(WireLogger);

    let client = anthropic::Client::builder()
        .http_client(http_client)
        .api_key(api_key)
        .build()?;

    let agent = client
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent.prompt("What is 2 + 2?").await?.output;
    println!("Response: {response}");

    Ok(())
}

//! Real provider adapters over the existing recorder. The transport budget and
//! destination checks run before sending bytes, including streaming requests.

use bytes::Bytes;
use futures::FutureExt;
use rig::{
    client::CompletionClient,
    http_client::{BoxedHttpClient, HeaderMap, HttpMiddleware, Method, Uri},
    providers::{anthropic, gemini, openai},
};
use rig_core::{serve::adapters::CompletionAdapter, wasm_compat::WasmBoxedFuture};
use serde::Serialize;
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use super::{Case, Error, Evidence, Provider, execute};
use crate::cassettes::{CassetteMode, CassetteSpec, ProviderCassette};

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Limits {
    pub requests: usize,
    pub output_tokens: u64,
    pub seconds: u64,
    pub retries: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            requests: 32,
            output_tokens: 512,
            seconds: 300,
            retries: 0,
        }
    }
}

#[derive(Clone)]
pub(crate) struct Budget {
    pub limits: Limits,
    pub used: Arc<AtomicUsize>,
    pub deadline: Instant,
}

impl Budget {
    pub fn new(limits: Limits) -> Self {
        Self {
            deadline: Instant::now() + Duration::from_secs(limits.seconds),
            limits,
            used: Arc::default(),
        }
    }
    pub fn used(&self) -> usize {
        self.used.load(Ordering::SeqCst)
    }
}

struct Guard {
    budget: Budget,
    authority: String,
}

fn refuse(message: &str) -> rig_core::http_client::Error {
    rig_core::http_client::Error::Instance(Box::new(Error::Invariant(message.into())))
}

impl HttpMiddleware for Guard {
    fn before_request_headers<'a>(
        &'a self,
        _: &'a Method,
        uri: &'a Uri,
        _: &'a mut HeaderMap,
    ) -> WasmBoxedFuture<'a, rig_core::http_client::Result<()>> {
        Box::pin(async move {
            if uri.scheme_str() != Some("http")
                || uri.authority().map(|a| a.as_str()) != Some(self.authority.as_str())
            {
                return Err(refuse(
                    "transport destination is not the selected local cassette server",
                ));
            }
            if Instant::now() >= self.budget.deadline {
                return Err(refuse("provider elapsed-time budget exhausted"));
            }
            self.budget
                .used
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                    (used < self.budget.limits.requests).then_some(used + 1)
                })
                .map_err(|_| refuse("provider request budget exhausted"))?;
            Ok(())
        })
    }
    fn before_request_body<'a>(
        &'a self,
        _: &'a Method,
        _: &'a Uri,
        _: &'a HeaderMap,
        body: Bytes,
    ) -> WasmBoxedFuture<'a, rig_core::http_client::Result<Bytes>> {
        Box::pin(async move {
            let value: serde_json::Value = serde_json::from_slice(&body)
                .map_err(|_| refuse("provider request is not JSON"))?;
            let tokens = value
                .get("max_tokens")
                .or_else(|| value.get("max_completion_tokens"))
                .or_else(|| value.get("max_output_tokens"))
                .or_else(|| value.pointer("/generationConfig/maxOutputTokens"))
                .and_then(serde_json::Value::as_u64);
            if tokens.is_none_or(|tokens| tokens == 0 || tokens > self.budget.limits.output_tokens)
            {
                return Err(refuse(
                    "provider output token limit absent or exceeds budget",
                ));
            }
            Ok(body)
        })
    }
}

pub(crate) fn identity(
    provider: Provider,
) -> Result<(&'static str, &'static str, &'static str, &'static str), Error> {
    Ok(match provider {
        Provider::Anthropic => (
            "anthropic",
            "https://api.anthropic.com",
            "ANTHROPIC_API_KEY",
            anthropic::completion::CLAUDE_HAIKU_4_5,
        ),
        Provider::Openai => (
            "openai",
            "https://api.openai.com/v1",
            "OPENAI_API_KEY",
            openai::GPT_4_1_MINI,
        ),
        Provider::Gemini => (
            "gemini",
            "https://generativelanguage.googleapis.com",
            "GEMINI_API_KEY",
            gemini::completion::GEMINI_2_5_FLASH,
        ),
        Provider::Synthetic => {
            return Err(Error::Invocation(
                "synthetic cases do not have provider traffic".into(),
            ));
        }
    })
}

pub(crate) async fn run(
    case: &Case,
    mode: CassetteMode,
    path: &Path,
    budget: &Budget,
) -> Result<Evidence, Error> {
    let (provider, upstream, variable, model) = identity(case.provider)?;
    if mode == CassetteMode::Record && std::env::var_os(variable).is_none() {
        return Err(Error::Invocation(format!("missing {variable}")));
    }
    if mode == CassetteMode::Replay && !path.is_file() {
        return Err(Error::Invocation(format!(
            "missing cassette {}",
            path.display()
        )));
    }
    if mode == CassetteMode::Replay {
        super::artifacts::safe_cassette(path)?;
    }
    let cassette = ProviderCassette::start_at(
        provider,
        CassetteSpec::new(case.id),
        upstream,
        mode,
        path.to_owned(),
    )
    .await;
    let base = cassette.base_url();
    let authority = base
        .parse::<Uri>()
        .map_err(|_| Error::Invariant("cassette URI invalid".into()))?
        .authority()
        .ok_or_else(|| Error::Invariant("cassette URI has no authority".into()))?
        .as_str()
        .to_owned();
    let reqwest = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .timeout(Duration::from_secs(90))
        .build()
        .map_err(|e| Error::Invariant(format!("build transport: {e}")))?;
    let http: BoxedHttpClient = rig::http_client::ReqwestClient::new(reqwest)
        .boxed()
        .with_middleware(Guard {
            budget: budget.clone(),
            authority,
        });
    let result = match case.provider {
        Provider::Anthropic => {
            let client = anthropic::Client::builder()
                .api_key(cassette.api_key(variable))
                .base_url(&base)
                .http_client(http)
                .build()?;
            execute(
                case,
                CompletionAdapter::new(model, client.completion_model(model)),
            )
            .await
        }
        Provider::Openai => {
            let client = openai::Client::builder()
                .api_key(cassette.api_key(variable))
                .base_url(&base)
                .http_client(http)
                .build()?;
            execute(
                case,
                CompletionAdapter::new(model, client.completion_model(model)),
            )
            .await
        }
        Provider::Gemini => {
            let client = gemini::Client::builder()
                .api_key(cassette.api_key(variable))
                .base_url(&base)
                .http_client(http)
                .build()?;
            execute(
                case,
                CompletionAdapter::new(model, client.completion_model(model)),
            )
            .await
        }
        Provider::Synthetic => return Err(Error::Invocation("synthetic cassette mode".into())),
    };
    let finalized = std::panic::AssertUnwindSafe(cassette.finish())
        .catch_unwind()
        .await;
    match (result, finalized) {
        (Ok(evidence), Ok(())) => Ok(evidence),
        (result, finalized) => Err(Error::Invariant(format!(
            "{}; cassette {} (finalization {})",
            result.err().map_or_else(
                || "cassette request/consumption check failed".into(),
                |error| error.to_string()
            ),
            path.display(),
            if finalized.is_ok() {
                "completed"
            } else {
                "failed; capture may be incomplete"
            }
        ))),
    }
}

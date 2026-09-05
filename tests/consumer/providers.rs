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

use super::{Case, Error, Evidence, Provider, execute_with_deadline};
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

impl Limits {
    pub(crate) fn for_case(case: &Case) -> Self {
        Self {
            output_tokens: case.output_tokens(),
            ..Self::default()
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
    case: &Case,
) -> Result<(&'static str, &'static str, &'static str, &'static str), Error> {
    Ok(match case.provider {
        Provider::Anthropic => (
            "anthropic",
            "https://api.anthropic.com",
            "ANTHROPIC_API_KEY",
            if case.repair {
                "claude-opus-5"
            } else {
                anthropic::completion::CLAUDE_HAIKU_4_5
            },
        ),
        Provider::Openai => (
            "openai",
            "https://api.openai.com/v1",
            "OPENAI_API_KEY",
            if case.repair {
                openai::GPT_5_6_SOL
            } else {
                openai::GPT_4_1_MINI
            },
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
    let (provider, upstream, variable, model) = identity(case)?;
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
    let cassette = tokio::time::timeout_at(
        execution_deadline(mode, budget).into(),
        ProviderCassette::start_at(
            provider,
            CassetteSpec::new(case.id),
            upstream,
            mode,
            path.to_owned(),
        ),
    )
    .await
    .map_err(|_| Error::Invariant("provider cassette setup deadline exhausted".into()))?;
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
    let execution_deadline = execution_deadline(mode, budget);
    let api_key = cassette.api_key(variable);
    let execution = async {
        match case.provider {
            Provider::Anthropic => {
                let client = anthropic::Client::builder()
                    .api_key(api_key)
                    .base_url(&base)
                    .http_client(http)
                    .build()?;
                execute_with_deadline(
                    case,
                    CompletionAdapter::new(model, client.completion_model(model)),
                    Some(execution_deadline),
                )
                .await
            }
            Provider::Openai => {
                let client = openai::Client::builder()
                    .api_key(api_key)
                    .base_url(&base)
                    .http_client(http)
                    .build()?;
                execute_with_deadline(
                    case,
                    CompletionAdapter::new(model, client.completion_model(model)),
                    Some(execution_deadline),
                )
                .await
            }
            Provider::Gemini => {
                let client = gemini::Client::builder()
                    .api_key(api_key)
                    .base_url(&base)
                    .http_client(http)
                    .build()?;
                execute_with_deadline(
                    case,
                    CompletionAdapter::new(model, client.completion_model(model)),
                    Some(execution_deadline),
                )
                .await
            }
            Provider::Synthetic => Err(Error::Invocation("synthetic cassette mode".into())),
        }
    };
    complete_capture(cassette, path, budget, mode, execution).await
}

fn execution_deadline(mode: CassetteMode, budget: &Budget) -> Instant {
    if mode == CassetteMode::Record {
        budget
            .deadline
            .checked_sub(Duration::from_secs(10))
            .unwrap_or(budget.deadline)
    } else {
        budget.deadline
    }
}

async fn complete_capture(
    cassette: ProviderCassette,
    path: &Path,
    budget: &Budget,
    mode: CassetteMode,
    execution: impl std::future::Future<Output = Result<Evidence, Error>>,
) -> Result<Evidence, Error> {
    let result = {
        let execution = std::panic::AssertUnwindSafe(execution).catch_unwind();
        tokio::pin!(execution);
        if mode == CassetteMode::Record {
            let deadline = execution_deadline(mode, budget);
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                tokio::select! {
                    result=&mut execution => break result.unwrap_or_else(|payload|Err(Error::Invariant(format!("consumer panicked: {}",panic_text(payload))))),
                    _=tokio::time::sleep_until(deadline.into())=>break Err(Error::Invariant("provider execution deadline exhausted; remaining time reserved for recording finalization".into())),
                    _=interval.tick()=> {
                        let snapshot_deadline=deadline.min(Instant::now()+Duration::from_secs(1));
                        let _=tokio::time::timeout_at(snapshot_deadline.into(),cassette.checkpoint_recording(&path.with_file_name("provider.partial.yaml"))).await;
                    }
                }
            }
        } else {
            execution.await.unwrap_or_else(|payload| {
                Err(Error::Invariant(format!(
                    "consumer panicked: {}",
                    panic_text(payload)
                )))
            })
        }
    };
    if mode == CassetteMode::Record {
        let snapshot_deadline = budget.deadline.min(Instant::now() + Duration::from_secs(2));
        let _ = tokio::time::timeout_at(
            snapshot_deadline.into(),
            cassette.checkpoint_recording(&path.with_file_name("provider.partial.yaml")),
        )
        .await;
    }
    let finalized = tokio::time::timeout_at(
        budget.deadline.into(),
        std::panic::AssertUnwindSafe(cassette.finish()).catch_unwind(),
    )
    .await;
    let finalized = match finalized {
        Ok(Ok(())) => Ok(()),
        Ok(Err(payload)) => Err(panic_text(payload)),
        Err(_) => Err("recording finalization deadline exhausted".into()),
    };
    match (result, finalized) {
        (Ok(evidence), Ok(())) => Ok(evidence),
        (result, finalized) => Err(Error::Invariant(format!(
            "{}; cassette {} (finalization {})",
            result.err().map_or_else(
                || "cassette request/consumption check failed".into(),
                |error| error.to_string()
            ),
            path.display(),
            finalized.err().map_or_else(
                || "completed".into(),
                |error| {
                    let partial = path.with_file_name("provider.partial.yaml");
                    let retained = if partial.is_file() {
                        format!("retained partial {}", partial.display())
                    } else {
                        "no partial recording available".into()
                    };
                    format!("failed: {error}; capture may be incomplete; {retained}")
                }
            )
        ))),
    }
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).into()
    } else {
        "non-text panic payload".into()
    }
}

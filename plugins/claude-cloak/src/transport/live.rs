//! Opt-in production acceptance probes. Duplicate SSE indexes and cancelled
//! credential refreshes remain deterministic offline tests, not live probes.

use super::*;
use rig_harness::{
    prelude::*,
    rig_core::{
        completion::{
            CachePoint, CacheRetention, CompletionRequest, CompletionResponse, Message,
            ToolDefinition,
        },
        effect::{Answered, EffectId, EffectKind},
        message::{AssistantContent, ToolName, ToolResultContent},
        serve::{Dispatch, ErasedHandler},
    },
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

mod tests;

pub(super) type ProbeResult<T> = Result<T, Box<dyn std::error::Error>>;
pub(super) const MODEL: &str = "claude-cloak/claude-opus-5-5";

#[derive(Clone)]
struct Audit {
    inner: DynHttpClient,
    records: Arc<Mutex<Vec<(usize, u16)>>>,
    root_ttl_attempts: Option<Arc<AtomicUsize>>,
}

fn markers(value: &Value) -> usize {
    match value {
        Value::Object(fields) => {
            usize::from(fields.contains_key("cache_control"))
                + fields.values().map(markers).sum::<usize>()
        }
        Value::Array(values) => values.iter().map(markers).sum(),
        _ => 0,
    }
}

impl Audit {
    #[allow(clippy::result_large_err)]
    fn inspect(&self, request: &Request<Bytes>) -> http_client::Result<usize> {
        validate(request.uri()).map_err(http_client::Error::instance)?;
        let mut decoded = Vec::new();
        if request.headers().get("content-encoding") == Some(&HeaderValue::from_static("gzip")) {
            use std::io::Read;
            flate2::read::GzDecoder::new(request.body().as_ref())
                .read_to_end(&mut decoded)
                .map_err(http_client::Error::instance)?;
        } else {
            decoded.extend_from_slice(request.body());
        }
        let value: Value =
            serde_json::from_slice(&decoded).map_err(http_client::Error::instance)?;
        if let Some(attempts) = &self.root_ttl_attempts {
            inspect_root_ttl(&value, attempts).map_err(http_client::Error::instance)?;
        }
        let count = markers(&value);
        if count > 4 {
            return Err(http_client::Error::instance(Error::CacheBudget));
        }
        if let Some(messages) = value.get("messages").and_then(Value::as_array) {
            for block in messages
                .iter()
                .filter_map(|message| message.get("content").and_then(Value::as_array))
                .flatten()
            {
                if block.get("type").and_then(Value::as_str) == Some("tool_use")
                    && block.get("name").and_then(Value::as_str) != Some("_echo")
                {
                    return Err(http_client::Error::instance(Error::Body));
                }
            }
        }
        Ok(count)
    }

    #[allow(clippy::result_large_err)]
    fn record(&self, count: usize, status: u16) -> http_client::Result<()> {
        let mut records = self
            .records
            .lock()
            .map_err(|_| http_client::Error::instance(Error::Body))?;
        records.push((count, status));
        Ok(())
    }
}

fn inspect_root_ttl(value: &Value, attempts: &AtomicUsize) -> Result<(), Error> {
    if attempts.fetch_add(1, Ordering::SeqCst) >= 2 {
        return Err(Error::Body);
    }
    let object = value.as_object().ok_or(Error::Body)?;
    if value.get("model").and_then(Value::as_str) != Some("claude-opus-5-5")
        || value.get("max_tokens").and_then(Value::as_u64) != Some(2048)
        || value.get("cache_control") != Some(&json!({"type":"ephemeral","ttl":"5m"}))
        || !value
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
        || cache_blocks(object)
            .any(|block| block.get("cache_control").is_some_and(|v| !v.is_null()))
    {
        return Err(Error::Body);
    }
    validate_caching(object)?;
    Ok(())
}

impl HttpClientExt for Audit {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let this = self.clone();
        let req = req.map(Into::into);
        async move {
            let count = this.inspect(&req)?;
            let response = this.inner.send(req).await?;
            this.record(count, response.status().as_u16())?;
            Ok(response)
        }
    }

    #[allow(clippy::manual_async_fn)]
    fn send_multipart<U>(
        &self,
        _: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        async { Err(http_client::Error::instance(Error::Multipart)) }
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let req = req.map(Into::into);
        async move {
            let count = self.inspect(&req)?;
            let response = self.inner.send_streaming(req).await?;
            self.record(count, response.status().as_u16())?;
            Ok(response)
        }
    }
}

pub(super) fn request(prompt: &str) -> CompletionRequest {
    CompletionRequest::new(prompt)
        .max_tokens(1024)
        .additional_params(json!({"output_config":{"effort":"low"}}))
}

fn echo() -> ProbeResult<ToolDefinition> {
    Ok(ToolDefinition::new(
        ToolName::new("echo")?,
        "Return the supplied synthetic text unchanged",
        json!({"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}),
    ))
}

pub(super) async fn complete(
    handler: &ErasedHandler,
    request: CompletionRequest,
    stream: bool,
    id: u64,
) -> ProbeResult<Arc<CompletionResponse>> {
    let outcome = handler
        .handle(
            EffectKind::Completion {
                request: Arc::new(request),
                stream,
            },
            Dispatch::new(EffectId::from_raw(id), stream),
        )
        .await
        .into_outcome()
        .await
        .map_err(|error| {
            format!(
                "live completion failed: status={:?}, code={:?}",
                error.http_status, error.code
            )
        })?;
    let Answered::Completion(response) = outcome else {
        return Err("expected completion".into());
    };
    if response.error.is_some() || response.aborted.is_some() || response.choice.is_empty() {
        return Err("live completion did not finish successfully".into());
    }
    Ok(response)
}

fn report(name: &str, response: &CompletionResponse, requests: usize) {
    eprintln!(
        "PASS {name} model={MODEL} requests={requests} input_tokens={:?} output_tokens={:?}",
        response.usage.input_tokens, response.usage.output_tokens
    );
}

#[test]
#[ignore = "requires explicit live opt-in and stored OAuth credentials"]
fn real_api_three_probes_without_tokio_runtime() -> ProbeResult<()> {
    if std::env::var("CLAUDE_CLOAK_LIVE").as_deref() != Ok("1")
        || std::env::var("CLAUDE_CLOAK_LIVE_MODEL").as_deref() != Ok(MODEL)
    {
        return Err(
            "set CLAUDE_CLOAK_LIVE=1 and CLAUDE_CLOAK_LIVE_MODEL=claude-cloak/claude-opus-5-5"
                .into(),
        );
    }
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err("live probes must run outside a Tokio runtime".into());
    }
    bevy_tasks::block_on(async {
        rig_http::wasm_compat::timeout(Duration::from_secs(240), probes())
            .await
            .map_err(|_| "live probes timed out")?
    })
}

pub(super) fn handler() -> ProbeResult<ErasedHandler> {
    let records = Arc::new(Mutex::new(Vec::new()));
    audited_handler(records)
}

fn audited_handler(records: Arc<Mutex<Vec<(usize, u16)>>>) -> ProbeResult<ErasedHandler> {
    audited_handler_with_root_ttl(records, None)
}

fn audited_handler_with_root_ttl(
    records: Arc<Mutex<Vec<(usize, u16)>>>,
    root_ttl_attempts: Option<Arc<AtomicUsize>>,
) -> ProbeResult<ErasedHandler> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| "HTTP client initialization failed")?;
    let client = DynHttpClient::new(OAuthTransport {
        inner: DynHttpClient::new(Audit {
            inner: DynHttpClient::new(rig_reqwest::ReqwestClient::from(client)),
            records: records.clone(),
            root_ttl_attempts,
        }),
        device: device_id().map_err(|_| "stored device identity unavailable")?,
        session: uuid::Uuid::new_v4().to_string(),
        credentials: None,
        platform: None,
    });
    let mut app = App::new();
    app.init_resource::<Models>();
    app.add_plugins(crate::ClaudeCloakPlugin);
    let catalog = app.world().resource::<Models>().0.catalog();
    let spec = catalog.resolve(MODEL)?;
    Ok(crate::connect_with_client(
        spec.spec,
        catalog,
        &crate::Transport::default(),
        client,
    )?)
}

async fn probes() -> ProbeResult<()> {
    let records = Arc::new(Mutex::new(Vec::new()));
    let handler = audited_handler(records.clone())?;
    let simple = complete(
        &handler,
        request("Reply with exactly: synthetic-ok"),
        false,
        0,
    )
    .await?;
    if simple.text().trim() != "synthetic-ok" {
        return Err("unary reply did not contain the requested synthetic text".into());
    }
    report("unary", &simple, 1);

    let mut tool_request = request(
        "Call echo exactly once with text synthetic-echo. Do not answer without calling echo.",
    )
    .tools([echo()?]);
    let first = complete(&handler, tool_request.clone(), true, 1).await?;
    let calls: Vec<_> = first
        .choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    let call = calls
        .first()
        .ok_or("model did not call echo; no retries permitted")?;
    if calls.len() != 1
        || call.function.name.as_str() != "echo"
        || call.function.arguments.len() != 1
        || call.function.arguments.get("text") != Some(&json!("synthetic-echo"))
    {
        return Err("unexpected synthetic echo call".into());
    }
    let saved = serde_json::to_vec(&first.message().ok_or("missing tool assistant message")?)?;
    let restored: Message = serde_json::from_slice(&saved)?;
    tool_request.chat_history.push(Arc::new(restored));
    tool_request
        .chat_history
        .push(Arc::new(Message::tool_results(vec![
            call.result(vec![ToolResultContent::text("synthetic-echo")]),
        ])));
    tool_request.chat_history.push(Arc::new(Message::user(
        "Now reply with the echoed text, without more tool calls.",
    )));
    let second = complete(&handler, tool_request, true, 2).await?;
    if second
        .choice
        .iter()
        .any(|block| matches!(block, AssistantContent::ToolCall(_)))
    {
        return Err("model unexpectedly called another tool".into());
    }
    if !second.text().contains("synthetic-echo") {
        return Err("tool continuation did not return the synthetic result".into());
    }
    report("streamed-tool-history", &second, 2);

    let mut cached = request("Synthetic turn one.")
        .cache(CacheRetention::Long)
        .tools([echo()?]);
    for message in [
        Message::assistant("Synthetic turn two."),
        Message::user("Synthetic turn three."),
        Message::assistant("Synthetic turn four."),
        Message::user("Synthetic turn five. Reply only cache-ok without calling tools."),
    ] {
        cached.chat_history.push(Arc::new(message));
    }
    cached.cache_points = vec![
        CachePoint::Messages(1),
        CachePoint::Messages(2),
        CachePoint::Messages(3),
    ];
    let response = complete(&handler, cached, false, 3).await?;
    if response.text().trim() != "cache-ok" {
        return Err("cache probe did not contain the requested synthetic text".into());
    }
    let records = records.lock().map_err(|_| "audit unavailable")?;
    if records.len() != 4
        || records
            .iter()
            .any(|(count, status)| *count > 4 || !(200..300).contains(status))
    {
        return Err("live request audit failed".into());
    }
    report("cache-budget", &response, 1);
    eprintln!(
        "PASS live audit requests=4 cache_markers={:?} statuses={:?}",
        records.iter().map(|(count, _)| count).collect::<Vec<_>>(),
        records.iter().map(|(_, status)| status).collect::<Vec<_>>()
    );
    Ok(())
}

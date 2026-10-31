use super::*;
use rig_harness::{
    prelude::*,
    rig_core::{
        completion::{CompletionRequest, Message, ToolDefinition},
        effect::EffectKind,
        serve::Dispatch,
    },
};
use std::sync::{Arc, Mutex};

use rig_harness::rig_core::json_utils::remove_preserving_order;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Clone)]
struct LocalSender {
    inner: DynHttpClient,
    uri: Uri,
    capture: Arc<Mutex<Option<Value>>>,
}

impl LocalSender {
    #[allow(clippy::result_large_err)]
    fn reroute(&self, mut request: Request<Bytes>) -> http_client::Result<Request<Bytes>> {
        let headers: serde_json::Map<String, Value> = request
            .headers()
            .iter()
            .map(|(k, v)| {
                (
                    k.to_string(),
                    Value::String(v.to_str().unwrap_or_default().into()),
                )
            })
            .collect();
        let mut decoded = Vec::new();
        if request
            .headers()
            .get("content-encoding")
            .and_then(|v| v.to_str().ok())
            == Some("gzip")
        {
            use std::io::Read;
            flate2::read::GzDecoder::new(request.body().as_ref())
                .read_to_end(&mut decoded)
                .map_err(http_client::Error::permanent)?;
        } else {
            decoded.extend_from_slice(request.body());
        }
        let capture = json!({"url":request.uri().to_string(),"headers":headers,"wireBytes":request.body().as_ref(),"bodyText":String::from_utf8_lossy(&decoded),"body":serde_json::from_slice::<Value>(&decoded).map_err(http_client::Error::permanent)?});
        *self
            .capture
            .lock()
            .map_err(|_| http_client::Error::permanent(Error::Body))? = Some(capture);
        *request.uri_mut() = self.uri.clone();
        Ok(request)
    }
}
impl HttpClientExt for LocalSender {
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
        async move { this.inner.send(this.reroute(req)?).await }
    }
    #[allow(clippy::manual_async_fn)]
    fn send_multipart<U>(
        &self,
        _: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        async { Err(http_client::Error::permanent(Error::Multipart)) }
    }
    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let req = req.map(Into::into);
        async move { self.inner.send_streaming(self.reroute(req)?).await }
    }
}

#[tokio::test]
#[ignore = "requires the shared OMP fixture file; performs localhost HTTP only"]
async fn capture_catalog_requests() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::var("OMP_PARITY_DIR")?;
    let captures = capture_requests(&root).await?;
    std::fs::write(
        format!("{root}/rig-captures.json"),
        serde_json::to_vec_pretty(&captures)?,
    )?;
    Ok(())
}

#[tokio::test]
async fn preserves_historical_omp_request_contract() -> Result<(), Box<dyn std::error::Error>> {
    let root = format!("{}/tests/fixtures/omp", env!("CARGO_MANIFEST_DIR"));
    for capture in capture_requests(&root).await? {
        let name = capture
            .get("name")
            .ok_or("missing capture name")?
            .as_str()
            .ok_or("fixture name")?;
        let expected: Value =
            serde_json::from_slice(&std::fs::read(format!("{root}/omp-{name}.json"))?)?;
        let expected_bytes = std::fs::read(format!("{root}/omp-{name}.decoded"))?;
        if let Some(length) = capture
            .get("expectedUtf16Length")
            .unwrap_or(&Value::Null)
            .as_u64()
        {
            let text = std::str::from_utf8(&expected_bytes)?;
            if text.encode_utf16().count() as u64 != length {
                return Err(format!("{name}: reference compression boundary drifted").into());
            }
        }
        if capture.get("url").ok_or("missing capture URL")?
            != expected.get("originalUrl").ok_or("missing reference URL")?
        {
            return Err(format!("{name}: request URL mismatch").into());
        }
        let headers =
            |value: &Value| -> Result<serde_json::Map<String, Value>, Box<dyn std::error::Error>> {
                Ok(value
                    .as_object()
                    .ok_or("headers object")?
                    .iter()
                    .filter(|(name, _)| {
                        !matches!(
                            name.as_str(),
                            "host"
                                | "content-length"
                                | "transfer-encoding"
                                | "user-agent"
                                | "x-stainless-package-version"
                                | "anthropic-beta"
                                | "content-encoding"
                        )
                    })
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect())
            };
        if headers(capture.get("headers").ok_or("missing capture headers")?)?
            != headers(expected.get("headers").ok_or("missing reference headers")?)?
        {
            return Err(format!("{name}: application headers mismatch").into());
        }
        assert_current_headers(&capture, &expected)?;
        let actual = capture
            .get("bodyText")
            .ok_or("missing capture body")?
            .as_str()
            .ok_or("body text")?
            .as_bytes();
        let actual = current_profile_projection(actual, true, false)?;
        let expected_bytes =
            current_profile_projection(&expected_bytes, false, name == "cache_none")?;
        if actual != expected_bytes {
            let offset = actual
                .iter()
                .zip(&expected_bytes)
                .position(|(a, b)| a != b)
                .unwrap_or(actual.len().min(expected_bytes.len()));
            return Err(format!("{name}: uncompressed JSON differs at byte {offset}").into());
        }
    }
    Ok(())
}

fn assert_current_headers(
    capture: &Value,
    historical: &Value,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::BTreeSet;
    let reference: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/claude-code/profile-2.1.307.json"
    )))?;
    let headers = capture.get("headers").ok_or("current headers")?;
    for (key, expected) in reference
        .get("headers")
        .and_then(Value::as_object)
        .ok_or("reference headers")?
    {
        if headers.get(key) != Some(expected) {
            return Err(format!("current profile header differs: {key}").into());
        }
    }
    let old_betas: BTreeSet<_> = historical
        .pointer("/headers/anthropic-beta")
        .and_then(Value::as_str)
        .ok_or("historical betas")?
        .split(',')
        .collect();
    let agent = old_betas.contains("claude-code-20250219");
    let mut betas: BTreeSet<_> = reference
        .get(if agent { "betas" } else { "utility_betas" })
        .and_then(Value::as_array)
        .ok_or("reference betas")?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    for dynamic in [
        "effort-2025-11-24",
        "thinking-binding-controls-2026-08-01",
        "thinking-display-updates-2026-08-18",
    ] {
        betas.remove(dynamic);
    }
    if old_betas.contains("effort-2025-11-24") {
        betas.insert("effort-2025-11-24");
    }
    if matches!(
        historical
            .pointer("/body/thinking/type")
            .and_then(Value::as_str),
        Some("adaptive" | "enabled")
    ) {
        betas.insert("thinking-binding-controls-2026-08-01");
        if historical
            .pointer("/body/thinking/display")
            .and_then(Value::as_str)
            .is_none_or(|display| display == "updates")
        {
            betas.insert("thinking-display-updates-2026-08-18");
        }
    }
    let actual: BTreeSet<_> = headers
        .get("anthropic-beta")
        .and_then(Value::as_str)
        .ok_or("current betas")?
        .split(',')
        .collect();
    if actual != betas {
        return Err(format!(
            "current profile beta set differs for {}: actual {actual:?}, expected {betas:?}",
            capture
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("unnamed")
        )
        .into());
    }
    let body = capture
        .get("bodyText")
        .and_then(Value::as_str)
        .ok_or("current body")?;
    let custom_network = [
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
    ]
    .into_iter()
    .any(|key| std::env::var_os(key).is_some_and(|value| !value.is_empty()));
    let gzip = !custom_network && body.encode_utf16().count() >= 4096;
    if headers.get("content-encoding").and_then(Value::as_str) != gzip.then_some("gzip") {
        return Err("current profile compression mode differs".into());
    }
    Ok(())
}

fn current_profile_projection(
    bytes: &[u8],
    current: bool,
    cache_none: bool,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let anchor = bytes
        .windows(MARKER.len())
        .position(|part| part == MARKER.as_bytes())
        .ok_or("billing anchor")?;
    let offset = bytes
        .get(anchor..)
        .ok_or("billing range")?
        .windows(4)
        .position(|part| part == b"cch=")
        .ok_or("checksum")?
        + anchor
        + 4;
    let actual = std::str::from_utf8(bytes.get(offset..offset + 5).ok_or("checksum digits")?)?;
    let mut placeholder = bytes.to_vec();
    placeholder
        .get_mut(offset..offset + 5)
        .ok_or("checksum digits")?
        .copy_from_slice(b"00000");
    let expected = if current {
        profile::checksum::compute(&placeholder)
    } else {
        format!(
            "{:05x}",
            xxhash_rust::xxh64::xxh64(&placeholder, 0x4d659218e32a3268) & 0xfffff
        )
    };
    if actual != expected {
        return Err("reference or current attestation mismatch".into());
    }
    let mut value: Value = serde_json::from_slice(bytes)?;
    if cache_none {
        let identity = value
            .pointer_mut("/system/1")
            .and_then(Value::as_object_mut)
            .ok_or("identity block")?;
        if remove_preserving_order(identity, "cache_control") != Some(json!({"type":"ephemeral"})) {
            return Err("cache_none reference must retain the historical identity marker".into());
        }
    }
    let human = first_user_text(&value);
    let units: Vec<_> = human.encode_utf16().collect();
    let selected: Vec<_> = [4, 7, 20]
        .into_iter()
        .map(|index| units.get(index).copied().unwrap_or(u16::from(b'0')))
        .collect();
    let version = if current { "2.1.307" } else { "2.1.280" };
    let fingerprint = format!(
        "{:x}",
        Sha256::digest(format!(
            "59cf53e54c78{}{version}",
            String::from_utf16_lossy(&selected)
        ))
    );
    let billing = value.pointer_mut("/system/0/text").ok_or("billing block")?;
    let text = billing.as_str().ok_or("billing text")?;
    if !text.starts_with(&format!(
        "x-anthropic-billing-header: cc_version={version}.{};",
        fingerprint.get(..3).ok_or("fingerprint digits")?
    )) {
        return Err("unexpected compatibility profile".into());
    }
    *billing = "profile identity and checksum verified separately".into();
    if !current
        && let Some(thinking) = value.get_mut("thinking").and_then(Value::as_object_mut)
        && matches!(
            thinking.get("type").and_then(Value::as_str),
            Some("enabled" | "adaptive")
        )
    {
        thinking
            .entry("display")
            .or_insert_with(|| "updates".into());
    }
    js_json(&value).map_err(Into::into)
}

async fn capture_requests(root: &str) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let fixtures: Value = serde_json::from_slice(&std::fs::read(format!("{root}/fixtures.json"))?)?;
    let mut captures = Vec::new();
    for fixture in fixtures
        .get("cases")
        .ok_or("missing fixture cases")?
        .as_array()
        .ok_or("fixtures cases must be array")?
    {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let uri = format!("http://{}/v1/messages?beta=true", listener.local_addr()?).parse()?;
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await?;
            let mut bytes = Vec::new();
            let mut buf = [0; 8192];
            loop {
                let n = stream.read(&mut buf).await?;
                if n == 0 {
                    break;
                }
                bytes.extend_from_slice(
                    buf.get(..n)
                        .ok_or_else(|| std::io::Error::other("read size"))?,
                );
                if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(bytes.get(..end).unwrap_or_default());
                    let length = headers
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|s| s.trim().parse::<usize>().ok())
                        })
                        .unwrap_or(0);
                    if bytes.len() >= end + 4 + length {
                        break;
                    }
                }
            }
            let body = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fake\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-6\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
            stream.write_all(format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",body.len()).as_bytes()).await?;
            Ok::<_, std::io::Error>(bytes)
        });
        let capture = Arc::new(Mutex::new(None));
        let client = DynHttpClient::new(OAuthTransport {
            inner: DynHttpClient::new(LocalSender {
                inner: DynHttpClient::new(rig_reqwest::ReqwestClient::from(reqwest::Client::new())),
                uri,
                capture: capture.clone(),
            }),
            home: Default::default(),
            device: std::fs::read_to_string(format!("{root}/home/.omp/install-id"))?
                .trim()
                .into(),
            platform: Some(("Linux", "arm64")),
            session: fixtures
                .get("sessionId")
                .ok_or("missing fixture session")?
                .as_str()
                .ok_or("session")?
                .into(),
            credentials: Some((
                fixtures
                    .get("apiKey")
                    .ok_or("missing fixture token")?
                    .as_str()
                    .ok_or("fake token")?
                    .into(),
                Some(
                    fixtures
                        .get("accountUUID")
                        .ok_or("missing fixture account")?
                        .as_str()
                        .ok_or("account")?
                        .into(),
                ),
            )),
        });
        let mut app = App::new();
        app.init_resource::<Models>();
        app.add_plugins(crate::ClaudeCloakPlugin);
        let models = app.world().resource::<Models>();
        let catalog = models.0.catalog();
        let model = fixture
            .get("model")
            .unwrap_or(&Value::Null)
            .as_str()
            .or_else(|| fixtures.get("model").unwrap_or(&Value::Null).as_str())
            .ok_or("model")?;
        let spec = catalog.resolve(&format!("claude-cloak/{model}"))?;
        let handler =
            crate::connect_with_client(spec.spec, catalog, &crate::Transport::default(), client)?;
        let request = request(fixture)?.max_tokens(
            fixture
                .get("maxTokens")
                .unwrap_or(&Value::Null)
                .as_u64()
                .or_else(|| fixtures.get("maxTokens").unwrap_or(&Value::Null).as_u64()),
        );
        let reply = handler
            .handle(
                EffectKind::Completion {
                    request: Arc::new(request),
                    stream: true,
                },
                Dispatch::new(rig_harness::rig_core::effect::EffectId::from_raw(0), true),
            )
            .await
            .into_outcome()
            .await;
        if let Err(e) = reply {
            return Err(format!("fixture response failed: {e}").into());
        }
        let raw = tokio::time::timeout(std::time::Duration::from_secs(10), server).await???;
        let mut captured = capture
            .lock()
            .map_err(|_| "capture poisoned")?
            .take()
            .ok_or("missing capture")?;
        let object = captured
            .as_object_mut()
            .ok_or("capture must be an object")?;
        object.insert(
            "name".into(),
            fixture.get("name").ok_or("missing fixture name")?.clone(),
        );
        object.insert(
            "expectedUtf16Length".into(),
            fixture
                .get("expectedUtf16Length")
                .cloned()
                .unwrap_or(Value::Null),
        );
        object.insert("localHttpBytes".into(), json!(raw));
        captures.push(captured);
    }
    Ok(captures)
}

fn request(fixture: &Value) -> Result<CompletionRequest, Box<dyn std::error::Error>> {
    let mut request = CompletionRequest::new("").max_tokens(
        fixture
            .get("max_tokens")
            .unwrap_or(&Value::Null)
            .as_u64()
            .unwrap_or(4096),
    );
    request.chat_history.clear();
    if let Some(system) = fixture.get("systemPrompt").unwrap_or(&Value::Null).as_str() {
        request.chat_history.push(Arc::new(Message::system(system)));
    } else if let Some(blocks) = fixture
        .get("systemPrompt")
        .unwrap_or(&Value::Null)
        .as_array()
    {
        for block in blocks {
            request.chat_history.push(Arc::new(Message::system(
                block.as_str().ok_or("system text")?,
            )));
        }
    }
    for message in fixture
        .get("messages")
        .ok_or("missing fixture messages")?
        .as_array()
        .ok_or("missing messages")?
    {
        use rig_harness::rig_core::completion::message::{CallId, ToolName};
        if message.get("role").and_then(Value::as_str) == Some("toolResult") {
            request.chat_history.push(Arc::new(Message::tool_result(
                CallId::from_wire(
                    message
                        .get("toolCallId")
                        .ok_or("missing tool call id")?
                        .as_str()
                        .ok_or("tool call id")?,
                ),
                ToolName::new(
                    message
                        .get("toolName")
                        .ok_or("missing tool name")?
                        .as_str()
                        .ok_or("tool name")?,
                )?,
                message
                    .get("content")
                    .ok_or("missing message content")?
                    .as_array()
                    .ok_or("tool content")?
                    .iter()
                    .filter_map(|block| block.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )));
            continue;
        }
        if message.get("role").and_then(Value::as_str) == Some("assistant")
            && message
                .get("content")
                .ok_or("missing message content")?
                .as_array()
                .is_some_and(|blocks| {
                    blocks
                        .iter()
                        .any(|b| b.get("type").and_then(Value::as_str) == Some("toolCall"))
                })
        {
            use rig_harness::rig_core::completion::message::AssistantContent;
            let content = message
                .get("content")
                .ok_or("missing message content")?
                .as_array()
                .ok_or("assistant content")?
                .iter()
                .map(|block| {
                    if block.get("type").and_then(Value::as_str) == Some("toolCall") {
                        Ok(AssistantContent::tool_call(
                            block
                                .get("id")
                                .ok_or("missing tool id")?
                                .as_str()
                                .ok_or("tool id")?,
                            ToolName::new(
                                block
                                    .get("name")
                                    .ok_or("missing tool name")?
                                    .as_str()
                                    .ok_or("tool name")?,
                            )?,
                            block
                                .get("arguments")
                                .ok_or("missing tool arguments")?
                                .clone(),
                        ))
                    } else {
                        Ok(AssistantContent::text(
                            block
                                .get("text")
                                .and_then(Value::as_str)
                                .ok_or("assistant text")?,
                        ))
                    }
                })
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            request.chat_history.push(Arc::new(Message::from(content)));
            continue;
        }
        let texts: Vec<String> = if let Some(text) = message
            .get("content")
            .ok_or("missing message content")?
            .as_str()
        {
            vec![text.into()]
        } else {
            message
                .get("content")
                .ok_or("missing message content")?
                .as_array()
                .ok_or("text content")?
                .iter()
                .map(|b| {
                    b.get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned()
                })
                .collect()
        };
        use rig_harness::rig_core::completion::message::{AssistantContent, UserContent};
        request.chat_history.push(Arc::new(
            match message.get("role").and_then(Value::as_str) {
                Some("assistant") => Message::from(
                    texts
                        .into_iter()
                        .map(AssistantContent::text)
                        .collect::<Vec<_>>(),
                ),
                Some("user") => {
                    Message::from(texts.into_iter().map(UserContent::text).collect::<Vec<_>>())
                }
                _ => return Err("unsupported role".into()),
            },
        ));
    }
    if let Some(tools) = fixture.get("tools").and_then(Value::as_array) {
        for tool in tools {
            request = request.tool(ToolDefinition::new(
                rig_harness::rig_core::completion::message::ToolName::new(
                    tool.get("name")
                        .ok_or("missing tool name")?
                        .as_str()
                        .ok_or("tool name")?,
                )?,
                tool.get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                tool.get("parameters")
                    .ok_or("missing tool parameters")?
                    .clone(),
            ));
        }
    }
    use rig_harness::rig_core::completion::{
        CacheRetention,
        message::{ToolChoice, ToolName},
    };
    if let Some(choice) = fixture.get("toolChoice") {
        request = request.tool_choice(match choice.as_str() {
            Some("auto") => ToolChoice::Auto,
            Some("none") => ToolChoice::None,
            Some("any") => ToolChoice::Required,
            _ => ToolChoice::Specific {
                function_names: vec![ToolName::new(
                    choice
                        .get("name")
                        .ok_or("missing forced tool name")?
                        .as_str()
                        .ok_or("forced tool")?,
                )?],
            },
        });
    }
    if let Some(cache) = fixture.get("cacheRetention").and_then(Value::as_str) {
        request.options.cache = Some(match cache {
            "none" => CacheRetention::None,
            "short" => CacheRetention::Short,
            "long" => CacheRetention::Long,
            _ => return Err("unsupported cache retention".into()),
        });
    }
    let mut params = serde_json::Map::new();
    if fixture.get("thinkingEnabled").and_then(Value::as_bool) == Some(true) {
        if let Some(budget) = fixture.get("thinkingBudgetTokens").and_then(Value::as_u64) {
            params.insert(
                "thinking".into(),
                json!({"type":"enabled", "budget_tokens":budget, "display":"summarized"}),
            );
        } else {
            params.insert("thinking".into(), json!({"type":"adaptive"}));
            params.insert(
                "output_config".into(),
                json!({"effort":fixture.get("effort").unwrap_or(&Value::Null)}),
            );
        }
    }
    request = request.temperature(fixture.get("temperature").and_then(Value::as_f64));
    for (option, wire) in [
        ("topP", "top_p"),
        ("topK", "top_k"),
        ("stopSequences", "stop_sequences"),
    ] {
        if let Some(value) = fixture.get(option) {
            params.insert(wire.into(), value.clone());
        }
    }
    if !params.is_empty() {
        request = request.additional_params(Value::Object(params));
    }
    Ok(request)
}

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
                .map_err(http_client::Error::instance)?;
        } else {
            decoded.extend_from_slice(request.body());
        }
        let capture = json!({"url":request.uri().to_string(),"headers":headers,"wireBytes":request.body().as_ref(),"bodyText":String::from_utf8_lossy(&decoded),"body":serde_json::from_slice::<Value>(&decoded).map_err(http_client::Error::instance)?});
        *self
            .capture
            .lock()
            .map_err(|_| http_client::Error::instance(Error::Body))? = Some(capture);
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
async fn matches_recorded_omp_requests() -> Result<(), Box<dyn std::error::Error>> {
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
                            "host" | "content-length" | "transfer-encoding"
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
        let actual = capture
            .get("bodyText")
            .ok_or("missing capture body")?
            .as_str()
            .ok_or("body text")?
            .as_bytes();
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

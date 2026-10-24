use super::*;
use rig_harness::{
    prelude::*,
    rig_core::{
        completion::{CompletionRequest, Message, ToolDefinition},
        effect::{Answered, EffectId, EffectKind},
        message::{AssistantContent, ToolName, ToolResultContent},
        serve::Dispatch,
    },
};
use std::{sync::Arc, time::Duration};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

// Rerouting happens only after the production destination guard and preparation.
#[derive(Clone)]
struct Loopback {
    inner: DynHttpClient,
    uri: Uri,
}
impl Loopback {
    #[allow(clippy::result_large_err)]
    fn reroute(&self, mut request: Request<Bytes>) -> http_client::Result<Request<Bytes>> {
        if request.uri() != "https://api.anthropic.com/v1/messages?beta=true"
            || request.headers().get("authorization")
                != Some(&HeaderValue::from_static("Bearer fake-roundtrip-token"))
        {
            return Err(http_client::Error::instance(Error::Destination));
        }
        *request.uri_mut() = self.uri.clone();
        Ok(request)
    }
}
impl HttpClientExt for Loopback {
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

fn events(tool_calls: bool) -> String {
    let mut events = vec![
        json!({"type":"message_start","message":{"id":"msg_local","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"usage":{"input_tokens":12,"output_tokens":0}}}),
    ];
    if tool_calls {
        for (index, id, name) in [
            (0, "call_read", "_read_file"),
            (1, "call_private", "__private"),
        ] {
            events.push(json!({"type":"content_block_start","index":index,"content_block":{"type":"tool_use","id":id,"name":name,"input":{},"caller":{"type":"code_execution_20250825","tool_id":"server_1"},"future_field":{"keep":true}}}));
        }
        for (index, fragment) in [
            (0, "{\"path\":"),
            (1, "{\"secret\":"),
            (0, "\"README.md\"}"),
            (1, "true}"),
        ] {
            events.push(json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":fragment}}));
        }
        for index in 0..2 {
            events.push(json!({"type":"content_block_stop","index":index}));
        }
    } else {
        events.extend([
            json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Finished both tools."}}),
            json!({"type":"content_block_stop","index":0}),
        ]);
    }
    events.push(json!({"type":"message_delta","delta":{"stop_reason":if tool_calls {"tool_use"} else {"end_turn"}},"usage":{"output_tokens":8}}));
    events.push(json!({"type":"message_stop"}));
    events
        .into_iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
            )
        })
        .collect()
}

async fn exchange(listener: &tokio::net::TcpListener, response: String) -> std::io::Result<Value> {
    let (mut stream, _) = listener.accept().await?;
    let mut bytes = Vec::new();
    let (end, length) = loop {
        let mut buffer = [0; 8192];
        let n = stream.read(&mut buffer).await?;
        if n == 0 {
            return Err(std::io::Error::other("incomplete HTTP request"));
        }
        bytes.extend_from_slice(
            buffer
                .get(..n)
                .ok_or_else(|| std::io::Error::other("read size"))?,
        );
        if bytes.len() > 1_048_576 {
            return Err(std::io::Error::other("oversized request"));
        }
        if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(bytes.get(..end).unwrap_or_default());
            if let Some(length) = headers.lines().find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .and_then(|s| s.trim().parse::<usize>().ok())
            }) && bytes.len() >= end + 4 + length
            {
                break (end, length);
            }
        }
    };
    let headers = String::from_utf8_lossy(bytes.get(..end).unwrap_or_default());
    let body = bytes
        .get(end + 4..end + 4 + length)
        .ok_or_else(|| std::io::Error::other("missing body"))?;
    let mut decoded = Vec::new();
    if headers
        .to_ascii_lowercase()
        .contains("content-encoding: gzip")
    {
        use std::io::Read;
        flate2::read::GzDecoder::new(body).read_to_end(&mut decoded)?;
    } else {
        decoded.extend_from_slice(body);
    }
    let request = serde_json::from_slice(&decoded)?;
    stream.write_all(format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n", response.len()).as_bytes()).await?;
    // Split the SSE byte stream independently from its JSON argument fragments.
    for chunk in response.as_bytes().chunks(37) {
        stream.write_all(chunk).await?;
    }
    stream.shutdown().await?;
    Ok(request)
}

#[tokio::test]
async fn streamed_tool_results_roundtrip_through_catalog() -> TestResult<()> {
    tokio::time::timeout(Duration::from_secs(15), roundtrip()).await?
}

async fn roundtrip() -> TestResult<()> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let uri = format!("http://{}/v1/messages?beta=true", listener.local_addr()?).parse()?;
    let server = tokio::spawn(async move {
        let first = exchange(&listener, events(true)).await?;
        let second = exchange(&listener, events(false)).await?;
        Ok::<_, std::io::Error>((first, second))
    });
    let client = DynHttpClient::new(OAuthTransport {
        inner: DynHttpClient::new(Loopback {
            inner: DynHttpClient::new(rig_reqwest::ReqwestClient::from(
                reqwest::Client::builder()
                    .redirect(reqwest::redirect::Policy::none())
                    .build()?,
            )),
            uri,
        }),
        device: "fake-roundtrip-device".into(),
        session: "fake-roundtrip-session".into(),
        credentials: Some(("fake-roundtrip-token".into(), Some("fake-account".into()))),
        platform: None,
    });
    let mut app = App::new();
    app.init_resource::<Models>();
    app.add_plugins(crate::ClaudeCloakPlugin);
    let catalog = app.world().resource::<Models>().0.catalog();
    let spec = catalog.resolve("claude-cloak/claude-sonnet-4-6")?;
    let handler =
        crate::connect_with_client(spec.spec, catalog, &crate::Transport::default(), client)?;
    let mut request = CompletionRequest::new("Read the file and private data.").max_tokens(1024).tools([
        ToolDefinition::new(ToolName::new("read_file")?, "Read a file", json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})),
        ToolDefinition::new(ToolName::new("_private")?, "Private lookup", json!({"type":"object","properties":{"secret":{"type":"boolean"}},"required":["secret"]})),
    ]);
    let first = handler
        .handle(
            EffectKind::Completion {
                request: Arc::new(request.clone()),
                stream: true,
            },
            Dispatch::new(EffectId::from_raw(0), true),
        )
        .await
        .into_outcome()
        .await?;
    let Answered::Completion(first) = first else {
        return Err("expected completion".into());
    };
    let calls: Vec<_> = first
        .choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    let decoded = calls.iter().map(|call| json!({"id":call.id.wire(),"name":call.function.name,"arguments":call.function.arguments})).collect::<Vec<_>>();
    if decoded
        != vec![
            json!({"id":"call_read","name":"read_file","arguments":{"path":"README.md"}}),
            json!({"id":"call_private","name":"_private","arguments":{"secret":true}}),
        ]
    {
        return Err(format!("decoded calls differ: {decoded:?}").into());
    }
    let saved = serde_json::to_vec(&first.message().ok_or("missing assistant message")?)?;
    let restored: Message = serde_json::from_slice(&saved)?;
    request.chat_history.push(Arc::new(restored));
    let results = calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            let mut result = call.result(vec![ToolResultContent::text(if index == 0 {
                "file contents"
            } else {
                "permission denied"
            })]);
            result.is_error = index == 1;
            result
        })
        .collect();
    request
        .chat_history
        .push(Arc::new(Message::tool_results(results)));
    request
        .chat_history
        .push(Arc::new(Message::user("Summarize the results.")));
    let final_reply = handler
        .handle(
            EffectKind::Completion {
                request: Arc::new(request),
                stream: true,
            },
            Dispatch::new(EffectId::from_raw(1), true),
        )
        .await
        .into_outcome()
        .await?;
    let Answered::Completion(final_reply) = final_reply else {
        return Err("expected final completion".into());
    };
    if !matches!(final_reply.choice.as_slice(), [AssistantContent::Text(text)] if text.text == "Finished both tools.")
    {
        return Err("final streamed text differs".into());
    }
    let (first_wire, next_wire) = server.await??;
    for wire in [&first_wire, &next_wire] {
        if wire.pointer("/tools/0/name") != Some(&json!("_read_file"))
            || wire.pointer("/tools/1/name") != Some(&json!("__private"))
            || wire.get("stream") != Some(&json!(true))
        {
            return Err("tool declarations or stream flag differ".into());
        }
    }
    let messages = next_wire
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("missing history")?;
    let uses: Vec<_> = messages
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|b| b.get("type") == Some(&json!("tool_use")))
        .map(|b| json!({"id":b.get("id"),"name":b.get("name"),"input":b.get("input"),"caller":b.get("caller"),"future_field":b.get("future_field")}))
        .collect();
    if uses
        != vec![
            json!({"id":"call_read","name":"_read_file","input":{"path":"README.md"},"caller":{"type":"code_execution_20250825","tool_id":"server_1"},"future_field":{"keep":true}}),
            json!({"id":"call_private","name":"__private","input":{"secret":true},"caller":{"type":"code_execution_20250825","tool_id":"server_1"},"future_field":{"keep":true}}),
        ]
    {
        return Err(format!("replayed calls differ: {uses:?}").into());
    }
    let results: Vec<_> = messages
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|b| b.get("type") == Some(&json!("tool_result")))
        .collect();
    if results.len() != 2 {
        return Err("expected both tool results".into());
    }
    for (index, (id, error, text)) in [
        ("call_read", false, "file contents"),
        ("call_private", true, "permission denied"),
    ]
    .into_iter()
    .enumerate()
    {
        let result = results.get(index).ok_or("missing result")?;
        if result.get("tool_use_id") != Some(&json!(id))
            || result.get("is_error") != Some(&json!(error))
            || !result
                .get("content")
                .is_some_and(|v| v == &json!(text) || v == &json!([{"type":"text","text":text}]))
        {
            return Err(format!("result {index} differs: {result}").into());
        }
    }
    let last = messages.last().ok_or("empty history")?;
    if last.get("role") != Some(&json!("user"))
        || !last.get("content").is_some_and(|v| {
            v == &json!("Summarize the results.")
                || v.as_array().is_some_and(|a| {
                    a.first().and_then(|b| b.get("text")) == Some(&json!("Summarize the results."))
                })
        })
    {
        return Err("follow-up user message must follow tool results".into());
    }
    Ok(())
}

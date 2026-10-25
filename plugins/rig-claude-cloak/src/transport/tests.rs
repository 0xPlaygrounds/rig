use super::*;

mod cache_override;
mod current_profile;

#[test]
fn platform_headers_match_host_architecture() {
    let (os, arch) = platform();
    #[cfg(target_os = "linux")]
    assert_eq!(os, "Linux");
    #[cfg(target_os = "macos")]
    assert_eq!(os, "MacOS");
    match std::env::consts::ARCH {
        "aarch64" => assert_eq!(arch, "arm64"),
        "x86" => assert_eq!(arch, "x86"),
        _ => assert_eq!(arch, "x64"),
    }
}

#[test]
fn human_provenance_overrides_injected_wire_text() -> anyhow::Result<()> {
    let human = "The original human prompt for fingerprinting";
    let baseline = json!({"model":"claude-sonnet-4-6","max_tokens":100,
        "messages":[{"role":"user","content":human}]});
    let injected = json!({"model":"claude-sonnet-4-6","max_tokens":100,
        "_omp":{"human_suffix":profile::suffix(human)},
        "messages":[{"role":"user","content":"Injected environment reminder"},
                    {"role":"user","content":human}]});
    let plain: Value = serde_json::from_slice(&body(
        &serde_json::to_vec(&baseline)?,
        "device",
        "session",
        None,
    )?)?;
    let shaped: Value = serde_json::from_slice(&body(
        &serde_json::to_vec(&injected)?,
        "device",
        "session",
        None,
    )?)?;
    let version = |value: &Value| {
        value
            .pointer("/system/0/text")
            .and_then(Value::as_str)
            .and_then(|text| text.split(';').next())
            .map(str::to_owned)
    };
    anyhow::ensure!(version(&plain).is_some());
    anyhow::ensure!(version(&plain) == version(&shaped));
    anyhow::ensure!(shaped.get("_omp").is_none());
    Ok(())
}

#[test]
fn destination_guard_fails_closed() -> anyhow::Result<()> {
    for url in [
        "http://api.anthropic.com/v1/messages",
        "https://api.anthropic.com.evil.test/v1/messages",
        "https://api.anthropic.com:444/v1/messages",
        "https://api.anthropic.com/v1/other",
        "https://evil.test/v1/messages",
        "https://user@api.anthropic.com/v1/messages",
    ] {
        let uri = url.parse::<Uri>()?;
        anyhow::ensure!(validate(&uri).is_err(), "{url}");
    }
    anyhow::ensure!(validate(&Uri::from_static("https://api.anthropic.com/v1/messages")).is_ok());
    Ok(())
}

#[test]
fn attestation_hashes_final_order_and_preserves_user_placeholder() -> anyhow::Result<()> {
    let source = br#"{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"user cch=00000 remains"}]}],"system":[{"type":"text","text":"original"}],"stream":true}"#;
    let wire = body(source, "device", "session", Some("account"))?;
    let text = String::from_utf8(wire.to_vec())?;
    anyhow::ensure!(text.contains("user cch=00000 remains"));
    let marker = text
        .find(MARKER)
        .ok_or_else(|| anyhow::anyhow!("missing marker"))?;
    let offset = text
        .get(marker..)
        .and_then(|s| s.find("cch="))
        .map(|i| i + marker + 4)
        .ok_or_else(|| anyhow::anyhow!("missing hash"))?;
    let actual = text
        .get(offset..offset + 5)
        .ok_or_else(|| anyhow::anyhow!("missing digits"))?;
    let mut placeholder = wire.to_vec();
    placeholder
        .get_mut(offset..offset + 5)
        .ok_or_else(|| anyhow::anyhow!("missing digits"))?
        .copy_from_slice(b"00000");
    anyhow::ensure!(actual == profile::checksum::compute(&placeholder));
    let value: Value = serde_json::from_slice(&wire)?;
    anyhow::ensure!(
        (value.pointer("/system/2/text").and_then(Value::as_str)) == (Some("original"))
    );
    let identity: Value = serde_json::from_str(
        value
            .pointer("/metadata/user_id")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    )?;
    anyhow::ensure!((identity.get("session_id").and_then(Value::as_str)) == (Some("session")));
    anyhow::ensure!((identity.get("account_uuid").and_then(Value::as_str)) == (Some("account")));
    anyhow::ensure!((wire) == (body(source, "device", "session", Some("account"))?));
    Ok(())
}

#[test]
fn tool_names_are_prefixed_only_in_wire_name_fields() -> anyhow::Result<()> {
    let source = json!({"model":"test","tools":[{"name":"read","input_schema":{"type":"object"}},{"name":"web_search"}],"tool_choice":{"type":"tool","name":"read"},"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call","name":"read","input":{"name":"unchanged"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call","content":"read"}]}]});
    let wire = body(&serde_json::to_vec(&source)?, "device", "session", None)?;
    let value: Value = serde_json::from_slice(&wire)?;
    for pointer in [
        "/tools/0/name",
        "/tool_choice/name",
        "/messages/0/content/0/name",
    ] {
        anyhow::ensure!((value.pointer(pointer).and_then(Value::as_str)) == (Some("_read")));
    }
    anyhow::ensure!(
        (value.pointer("/tools/1/name").and_then(Value::as_str)) == (Some("web_search"))
    );
    anyhow::ensure!(
        (value
            .pointer("/messages/0/content/0/input/name")
            .and_then(Value::as_str))
            == (Some("unchanged"))
    );
    anyhow::ensure!(
        (value
            .pointer("/messages/1/content/0/tool_use_id")
            .and_then(Value::as_str))
            == (Some("call"))
    );
    Ok(())
}

#[test]
fn billing_uses_utf16_and_first_nonempty_text_block() -> anyhow::Result<()> {
    let source = json!({"messages":[{"role":"user","content":"  "},{"role":"user","content":[{"type":"text","text":" "},{"type":"text","text":"abcd😀XYabcdefghijklmnop"},{"type":"text","text":"not included"}]}]});
    anyhow::ensure!((first_user_text(&source)) == ("abcd😀XYabcdefghijklmnop"));
    let wire = body(&serde_json::to_vec(&source)?, "device", "session", None)?;
    let value: Value = serde_json::from_slice(&wire)?;
    anyhow::ensure!(
        value
            .pointer("/system/0/text")
            .and_then(Value::as_str)
            .is_some_and(|s| s.starts_with("x-anthropic-billing-header: cc_version=2.1.307.05c;"))
    );
    let image = json!({"messages":[{"role":"user","content":[{"type":"image"}]},{"role":"user","content":"later"}]});
    anyhow::ensure!((first_user_text(&image)) == (""));
    Ok(())
}

#[test]
fn existing_billing_header_is_rejected() -> anyhow::Result<()> {
    anyhow::ensure!(
        body(
            br#"{"system":[{"type":"text","text":"x-anthropic-billing-header: existing"}]}"#,
            "device",
            "session",
            None
        )
        .is_err()
    );
    Ok(())
}

#[tokio::test]
async fn prepare_preserves_required_betas_and_removes_oauth_incompatible_headers()
-> anyhow::Result<()> {
    let transport = OAuthTransport {
        home: Default::default(),
        inner: DynHttpClient::new(rig_reqwest::ReqwestClient::from(reqwest::Client::new())),
        device: "fake-device".into(),
        session: "fake-session".into(),
        credentials: Some(("fake-token".into(), None)),
        platform: None,
    };
    let request = Request::builder()
        .uri("https://api.anthropic.com/v1/messages")
        .header("anthropic-beta", "thinking-binding-controls-2026-08-01,fine-grained-tool-streaming-2025-05-14,context-1m-2025-08-07,oauth-2025-04-20")
        .header("x-client-request-id", "discard")
        .header("x-api-key", "discard")
        .body(Bytes::from_static(br#"{"thinking":{"type":"adaptive"},"output_config":{"effort":"high"}}"#))
        ?;
    let prepared = transport.prepare(request).await?;
    let betas = prepared
        .headers()
        .get("anthropic-beta")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    for beta in [
        "thinking-binding-controls-2026-08-01",
        "fine-grained-tool-streaming-2025-05-14",
        "effort-2025-11-24",
        "thinking-display-updates-2026-08-18",
        "extended-cache-ttl-2025-04-11",
    ] {
        anyhow::ensure!(betas.split(',').any(|b| b == beta), "missing {beta}");
    }
    anyhow::ensure!(!betas.contains("context-1m"));
    anyhow::ensure!(!betas.contains("fallback-credit"));
    anyhow::ensure!(
        (betas
            .split(',')
            .filter(|b| *b == "oauth-2025-04-20")
            .count())
            == (1)
    );
    anyhow::ensure!(!prepared.headers().contains_key("x-client-request-id"));
    anyhow::ensure!(!prepared.headers().contains_key("x-api-key"));
    Ok(())
}

#[test]
fn explicit_cache_points_take_priority_over_automatic_points() -> anyhow::Result<()> {
    let control = json!({"type":"ephemeral","ttl":"1h"});
    for count in 0..=5 {
        let messages: Vec<_> = (0..count).map(|_| json!({"role":"user","content":[{"type":"text","text":"explicit","cache_control":control}]})).chain(std::iter::once(json!({"role":"user","content":"tail"}))).collect();
        let source = json!({"system":"stable","tools":[{"name":"read"}],"messages":messages});
        let result = body(&serde_json::to_vec(&source)?, "device", "session", None);
        if count > 4 {
            anyhow::ensure!(matches!(result, Err(Error::CacheBudget)));
            continue;
        }
        let value: Value = serde_json::from_slice(&result?)?;
        let object = value.as_object().ok_or_else(|| anyhow::anyhow!("object"))?;
        anyhow::ensure!(
            cache_blocks(object)
                .filter(|b| b.get("cache_control").is_some())
                .count()
                <= 4
        );
        for i in 0..count {
            anyhow::ensure!(
                value.pointer(&format!("/messages/{i}/content/0/cache_control")) == Some(&control)
            );
        }
        if count == 4 {
            anyhow::ensure!(value.pointer("/tools/0/cache_control").is_none());
            anyhow::ensure!(value.pointer("/system/2/cache_control").is_none());
        }
    }
    Ok(())
}

#[test]
fn cache_controls_preserve_explicit_heads_and_valid_ttl_order() -> anyhow::Result<()> {
    for retention in ["long", "short", "none"] {
        let value = shaped(
            json!({"_omp":{"cache_retention":retention},"tools":[{"name":"read","cache_control":{"type":"ephemeral","ttl":"1h"}}],"system":[{"type":"text","text":"explicit","cache_control":{"type":"ephemeral"}},{"type":"text","text":"stable"}],"messages":[{"role":"user","content":"tail"}]}),
        )?;
        anyhow::ensure!(
            value.pointer("/system/2/cache_control") == Some(&json!({"type":"ephemeral"}))
        );
        let mut object = value
            .as_object()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("object"))?;
        prompt_caching(&mut object, &json!({"cache_retention":"long"}))?;
    }
    let mut invalid = json!({"tools":[{"cache_control":{"type":"ephemeral"}}],"system":[{"cache_control":{"type":"ephemeral","ttl":"1h"}}]});
    anyhow::ensure!(matches!(
        prompt_caching(
            invalid
                .as_object_mut()
                .ok_or_else(|| anyhow::anyhow!("object"))?,
            &Value::Null
        ),
        Err(Error::CacheOrder)
    ));
    Ok(())
}

#[derive(Clone)]
struct CacheCapture(std::sync::Arc<std::sync::Mutex<Option<Value>>>);

impl CacheCapture {
    #[allow(clippy::result_large_err)]
    fn capture(&self, request: Request<Bytes>) -> http_client::Result<()> {
        use std::io::Read;
        let bytes = if request.headers().contains_key("content-encoding") {
            let mut bytes = Vec::new();
            flate2::read::GzDecoder::new(request.body().as_ref())
                .read_to_end(&mut bytes)
                .map_err(http_client::Error::instance)?;
            bytes
        } else {
            request.body().to_vec()
        };
        *self
            .0
            .lock()
            .map_err(|_| http_client::Error::instance(Error::Body))? =
            Some(serde_json::from_slice(&bytes).map_err(http_client::Error::instance)?);
        Ok(())
    }
}

impl HttpClientExt for CacheCapture {
    fn send<T, U>(
        &self,
        request: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let this = self.clone();
        let request = request.map(Into::into);
        async move {
            this.capture(request)?;
            Err(http_client::Error::instance(Error::Body))
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
        request: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let request = request.map(Into::into);
        async move {
            self.capture(request)?;
            Err(http_client::Error::instance(Error::Body))
        }
    }
}

#[tokio::test]
async fn generic_provider_explicit_cache_points_fit_final_wire_budget() -> anyhow::Result<()> {
    use rig_harness::{
        prelude::*,
        rig_core::{
            completion::{CachePoint, CacheRetention, CompletionRequest, Message, ToolDefinition},
            effect::{EffectId, EffectKind},
            message::ToolName,
            serve::Dispatch,
        },
    };
    use std::sync::{Arc, Mutex};
    for count in 1..=3 {
        let captured = Arc::new(Mutex::new(None));
        let client = DynHttpClient::new(OAuthTransport {
            home: Default::default(),
            inner: DynHttpClient::new(CacheCapture(captured.clone())),
            device: "offline-device".into(),
            session: "offline-session".into(),
            credentials: Some(("offline-token".into(), None)),
            platform: None,
        });
        let mut app = App::new();
        app.init_resource::<Models>();
        app.add_plugins(crate::ClaudeCloakPlugin);
        let catalog = app.world().resource::<Models>().0.catalog();
        let spec = catalog.resolve("claude-cloak/claude-sonnet-4-6")?;
        let handler =
            crate::connect_with_client(spec.spec, catalog, &crate::Transport::default(), client)?;
        let mut request = CompletionRequest::new("first")
            .max_tokens(32)
            .cache(CacheRetention::Long)
            .tools([ToolDefinition::new(
                ToolName::new("read")?,
                "Read",
                json!({"type":"object","properties":{}}),
            )]);
        request
            .chat_history
            .push(Arc::new(Message::assistant("second")));
        request.chat_history.push(Arc::new(Message::user("third")));
        request
            .chat_history
            .push(Arc::new(Message::assistant("tail")));
        request.cache_points = (1..=count).map(CachePoint::Messages).collect();
        let outcome = handler
            .handle(
                EffectKind::Completion {
                    request: Arc::new(request),
                    stream: false,
                },
                Dispatch::new(EffectId::from_raw(0), false),
            )
            .await
            .into_outcome()
            .await;
        anyhow::ensure!(
            outcome.is_err(),
            "capture transport must stop before network"
        );
        let value = captured
            .lock()
            .map_err(|_| anyhow::anyhow!("capture poisoned"))?
            .take()
            .ok_or_else(|| anyhow::anyhow!("request did not reach transport"))?;
        let object = value.as_object().ok_or_else(|| anyhow::anyhow!("object"))?;
        anyhow::ensure!(!object.contains_key("cache_control"));
        prompt_caching(&mut object.clone(), &json!({"cache_retention":"none"}))?;
        anyhow::ensure!(
            cache_blocks(object)
                .filter(|b| b.get("cache_control").is_some())
                .count()
                <= 4
        );
        for i in 0..count {
            anyhow::ensure!(
                value
                    .pointer(&format!("/messages/{i}/content/0/cache_control"))
                    .is_some(),
                "explicit point {i} missing"
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn generic_provider_preserves_explicit_raw_cache_override() -> anyhow::Result<()> {
    use rig_harness::{
        prelude::*,
        rig_core::{
            completion::{CacheRetention, CompletionRequest},
            effect::{EffectId, EffectKind},
            serve::Dispatch,
        },
    };
    use std::sync::{Arc, Mutex};
    for (retention, ttl) in [
        None,
        Some(CacheRetention::None),
        Some(CacheRetention::Short),
        Some(CacheRetention::Long),
    ]
    .into_iter()
    .flat_map(|retention| ["5m", "1h"].map(|ttl| (retention, ttl)))
    {
        let captured = Arc::new(Mutex::new(None));
        let client = DynHttpClient::new(OAuthTransport {
            home: Default::default(),
            inner: DynHttpClient::new(CacheCapture(captured.clone())),
            device: "offline-device".into(),
            session: "offline-session".into(),
            credentials: Some(("offline-token".into(), None)),
            platform: None,
        });
        let mut app = App::new();
        app.init_resource::<Models>();
        app.add_plugins(crate::ClaudeCloakPlugin);
        let catalog = app.world().resource::<Models>().0.catalog();
        let spec = catalog.resolve("claude-cloak/claude-sonnet-4-6")?;
        let handler =
            crate::connect_with_client(spec.spec, catalog, &crate::Transport::default(), client)?;
        let control = json!({"type":"ephemeral","ttl":ttl});
        let mut request = CompletionRequest::new("synthetic")
            .max_tokens(16)
            .additional_params(json!({"cache_control":control}));
        request.options.cache = retention;
        let outcome = handler
            .handle(
                EffectKind::Completion {
                    request: Arc::new(request),
                    stream: false,
                },
                Dispatch::new(EffectId::from_raw(0), false),
            )
            .await
            .into_outcome()
            .await;
        anyhow::ensure!(outcome.is_err(), "capture must stop before network");
        let value = captured
            .lock()
            .map_err(|_| anyhow::anyhow!("capture poisoned"))?
            .take()
            .ok_or_else(|| anyhow::anyhow!("request never reached transport"))?;
        anyhow::ensure!(value.get("cache_control") == Some(&control));
        let object = value.as_object().ok_or_else(|| anyhow::anyhow!("object"))?;
        anyhow::ensure!(cache_blocks(object).all(|b| b.get("cache_control").is_none()));
    }
    Ok(())
}

fn shaped(source: Value) -> anyhow::Result<Value> {
    let wire = body(&serde_json::to_vec(&source)?, "device", "session", None)?;
    Ok(serde_json::from_slice(&wire)?)
}

#[test]
fn cache_head_and_tail_match_omp_boundaries() -> anyhow::Result<()> {
    let value = shaped(
        json!({"system":[{"type":"text","text":"stable"},{"type":"text","text":"<memories>volatile"}],"tools":[{"name":"read"},{"name":"later","defer_loading":true}],"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":[{"type":"text","text":"answer"},{"type":"thinking","thinking":"reason"}]}]}),
    )?;
    let control = json!({"type":"ephemeral","ttl":"1h"});
    for path in [
        "/tools/0/cache_control",
        "/system/2/cache_control",
        "/messages/0/content/0/cache_control",
        "/messages/1/content/0/cache_control",
    ] {
        anyhow::ensure!((value.pointer(path)) == (Some(&control)), "{path}");
    }
    for path in [
        "/tools/1/cache_control",
        "/system/1/cache_control",
        "/system/3/cache_control",
        "/messages/1/content/1/cache_control",
    ] {
        anyhow::ensure!(value.pointer(path).is_none(), "{path}");
    }
    Ok(())
}

#[test]
fn decimation_has_priority_over_second_trailing_message() -> anyhow::Result<()> {
    let messages: Vec<_> = (0..31)
        .map(|i| json!({"role":"user","content":format!("turn {i}")}))
        .collect();
    let value = shaped(json!({"system":"stable","tools":[{"name":"read"}],"messages":messages}))?;
    for i in [29, 30] {
        anyhow::ensure!(
            value
                .pointer(&format!("/messages/{i}/content/0/cache_control"))
                .is_some()
        );
    }
    anyhow::ensure!(
        value
            .pointer("/messages/28/content/0/cache_control")
            .is_none()
    );
    anyhow::ensure!(
        value
            .pointer("/messages/14/content/0/cache_control")
            .is_none()
    );
    Ok(())
}

#[test]
fn explicit_raw_cache_control_survives_and_orders_last() -> anyhow::Result<()> {
    for retention in ["none", "short", "long"] {
        let control = json!({"type":"ephemeral","ttl":"1h"});
        let value = shaped(
            json!({"_omp":{"cache_retention":retention,"explicit_cache_control":true},"cache_control":control,"messages":[{"role":"user","content":"tail"}]}),
        )?;
        anyhow::ensure!(value.get("cache_control") == Some(&control));
        let object = value.as_object().ok_or_else(|| anyhow::anyhow!("object"))?;
        let markers: Vec<_> = cache_blocks(object)
            .filter_map(|b| b.get("cache_control"))
            .collect();
        anyhow::ensure!(markers.len() < 4);
        anyhow::ensure!(markers.iter().all(|marker| **marker == control));
        if retention == "none" {
            anyhow::ensure!(markers.is_empty());
        }
    }
    let invalid = json!({"_omp":{"cache_retention":"none","explicit_cache_control":true},"cache_control":{"type":"ephemeral","ttl":"1h"},"messages":[{"role":"user","content":[{"type":"text","text":"tail","cache_control":{"type":"ephemeral"}}]}]});
    anyhow::ensure!(matches!(
        body(&serde_json::to_vec(&invalid)?, "device", "session", None),
        Err(Error::CacheOrder)
    ));
    Ok(())
}

#[test]
fn retention_controls_only_automatic_markers() -> anyhow::Result<()> {
    for (retention, expected) in [
        ("none", None),
        ("short", Some(json!({"type":"ephemeral"}))),
        ("long", Some(json!({"type":"ephemeral","ttl":"1h"}))),
    ] {
        let value = shaped(
            json!({"_omp":{"cache_retention":retention},"cache_control":{"type":"ephemeral","ttl":"1h"},"tools":[{"name":"read"}],"messages":[{"role":"user","content":"tail"}]}),
        )?;
        let object = value.as_object().ok_or_else(|| anyhow::anyhow!("object"))?;
        anyhow::ensure!(!object.contains_key("cache_control"));
        let controls: Vec<_> = cache_blocks(object)
            .filter_map(|b| b.get("cache_control"))
            .collect();
        if let Some(expected) = expected {
            anyhow::ensure!(!controls.is_empty() && controls.len() <= 4);
            anyhow::ensure!(controls.iter().all(|control| **control == expected));
        } else {
            anyhow::ensure!(controls.is_empty());
        }
        let explicit = json!({"type":"ephemeral","ttl":"1h"});
        let value = shaped(
            json!({"_omp":{"cache_retention":retention},"messages":[{"role":"user","content":[{"type":"text","text":"explicit","cache_control":explicit}]}]}),
        )?;
        anyhow::ensure!(value.pointer("/messages/0/content/0/cache_control") == Some(&explicit));
    }
    Ok(())
}

#[test]
fn explicit_controls_survive_and_private_metadata_does_not() -> anyhow::Result<()> {
    let source = json!({"_omp":{"cache_retention":"none","tool_choice_explicit":true,"session_id":"own-session"},"tool_choice":{"type":"auto"},"thinking":{"type":"adaptive"},"context_management":{"edits":[]},"messages":[{"role":"user","content":"hello"}]});
    let value = shaped(source.clone())?;
    anyhow::ensure!(value.get("_omp").is_none());
    anyhow::ensure!((value.get("tool_choice")) == (Some(&json!({"type":"auto"}))));
    anyhow::ensure!((value.get("context_management")) == (Some(&json!({"edits":[]}))));
    anyhow::ensure!(value.pointer("/system/1/cache_control").is_none());
    anyhow::ensure!(
        value
            .pointer("/messages/0/content")
            .is_some_and(Value::is_string)
    );
    let identity: Value = serde_json::from_str(
        value
            .pointer("/metadata/user_id")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    )?;
    anyhow::ensure!((identity.get("session_id")) == (Some(&json!("own-session"))));
    let mut source = source;
    *source
        .pointer_mut("/_omp/tool_choice_explicit")
        .ok_or_else(|| anyhow::anyhow!("missing explicit choice"))? = false.into();
    *source
        .pointer_mut("/_omp/cache_retention")
        .ok_or_else(|| anyhow::anyhow!("missing cache retention"))? = "short".into();
    let value = shaped(source)?;
    anyhow::ensure!(value.get("tool_choice").is_none());
    anyhow::ensure!(
        (value.pointer("/messages/0/content/0/cache_control"))
            == (Some(&json!({"type":"ephemeral"})))
    );
    Ok(())
}

#[test]
fn thinking_context_default_and_tool_result_false_are_present() -> anyhow::Result<()> {
    let value = shaped(
        json!({"thinking":{"type":"enabled","budget_tokens":100},"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"id","content":[]}]}]}),
    )?;
    anyhow::ensure!(
        (value.pointer("/context_management/edits/0"))
            == (Some(&json!({"type":"clear_thinking_20251015","keep":"all"})))
    );
    anyhow::ensure!((value.pointer("/messages/0/content/0/is_error")) == (Some(&json!(false))));
    Ok(())
}

#[test]
fn forced_tools_sampling_and_buffered_yield_follow_omp_policy() -> anyhow::Result<()> {
    let value = shaped(
        json!({"_omp":{"adaptive_only":true},"thinking":{"type":"adaptive"},"tool_choice":{"type":"any"},"temperature":0.2,"top_p":0.5,"top_k":3,"stop_sequences":["a","b","c","d","e"],"tools":[{"name":"yield","eager_input_streaming":true}]}),
    )?;
    for field in [
        "thinking",
        "context_management",
        "temperature",
        "top_p",
        "top_k",
    ] {
        anyhow::ensure!(value.get(field).is_none(), "{field}");
    }
    anyhow::ensure!((value.pointer("/output_config/effort")) == (Some(&json!("low"))));
    let wire = value.to_string();
    anyhow::ensure!(wire.find("\"tool_choice\"") < wire.find("\"output_config\""));
    anyhow::ensure!((value.pointer("/tools/0/eager_input_streaming")) == (Some(&json!(false))));
    anyhow::ensure!((value.get("stop_sequences")) == (Some(&json!(["a", "b", "c", "d"]))));
    let value = shaped(
        json!({"_omp":{"forced_tools_allowed":false,"sampling_allowed":false},"tool_choice":{"type":"tool","name":"read"},"temperature":0.2}),
    )?;
    anyhow::ensure!((value.get("tool_choice")) == (Some(&json!({"type":"auto"}))));
    anyhow::ensure!(value.get("temperature").is_none());
    Ok(())
}

#[test]
fn budget_thinking_reserves_visible_output_within_model_ceiling() -> anyhow::Result<()> {
    let value = shaped(
        json!({"_omp":{"max_output_tokens":8192},"max_tokens":4096,"thinking":{"type":"enabled","budget_tokens":1024}}),
    )?;
    anyhow::ensure!((value.get("max_tokens")) == (Some(&json!(5024))));
    let value = shaped(
        json!({"_omp":{"max_output_tokens":8192},"max_tokens":8192,"thinking":{"type":"enabled","budget_tokens":6000}}),
    )?;
    anyhow::ensure!((value.pointer("/thinking/budget_tokens")) == (Some(&json!(4192))));
    Ok(())
}

#[test]
fn javascript_numbers_and_property_order_match_stringify() -> anyhow::Result<()> {
    let value: Value = serde_json::from_str(
        r#"{"z":1.0,"10":-0.0,"2":1e-7,"a":1e20,"b":1e21,"01":9007199254740993}"#,
    )?;
    anyhow::ensure!(
        (String::from_utf8(js_json(&value)?)?)
            == (r#"{"2":1e-7,"10":0,"z":1,"a":100000000000000000000,"b":1e+21,"01":9007199254740992}"#)
    );
    Ok(())
}

#[tokio::test]
async fn compression_and_session_header_follow_shaped_request() -> anyhow::Result<()> {
    use std::io::Read;
    let transport = OAuthTransport {
        home: Default::default(),
        inner: DynHttpClient::new(rig_reqwest::ReqwestClient::from(reqwest::Client::new())),
        device: "device".into(),
        session: "fallback".into(),
        credentials: Some(("fake-token".into(), None)),
        platform: None,
    };
    let source = serde_json::to_vec(
        &json!({"_omp":{"session_id":"saved-session"},"messages":[{"role":"user","content":"x".repeat(8192)}]}),
    )?;
    let expected = body(&source, "device", "fallback", None)?;
    let request = Request::builder()
        .uri("https://api.anthropic.com/v1/messages")
        .body(Bytes::from(source))?;
    let prepared = transport.prepare(request).await?;
    anyhow::ensure!(
        (prepared
            .headers()
            .get("x-claude-code-session-id")
            .and_then(|v| v.to_str().ok()))
            == (Some("saved-session"))
    );
    let mut decoded = Vec::new();
    match prepared
        .headers()
        .get("content-encoding")
        .and_then(|v| v.to_str().ok())
    {
        Some("gzip") => {
            flate2::read::GzDecoder::new(prepared.body().as_ref()).read_to_end(&mut decoded)?;
        }
        None => decoded.extend_from_slice(prepared.body()),
        Some(_) => anyhow::bail!("unexpected compression"),
    }
    anyhow::ensure!((decoded) == (expected));
    Ok(())
}

/// Native compression boundaries are checked without inherited proxy or CA settings.
#[test]
fn compression_uses_utf16_threshold_and_respects_custom_networks() {
    assert!(!compress_request("x".repeat(4095).as_bytes(), false));
    assert!(compress_request("x".repeat(4096).as_bytes(), false));
    assert!(!compress_request("😀".repeat(2047).as_bytes(), false));
    assert!(compress_request("😀".repeat(2048).as_bytes(), false));
    assert!(!compress_request("x".repeat(8192).as_bytes(), true));
}

#[test]
fn invalid_request_is_rejected() -> anyhow::Result<()> {
    for source in [b"not json".as_slice(), b"[]", br#"{"system":42}"#] {
        anyhow::ensure!(body(source, "device", "session", None).is_err());
    }
    Ok(())
}

use super::*;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
enum ProbeOrder {
    BaselineFirst,
    VariantFirst,
}

impl ProbeOrder {
    fn parse(value: &str) -> Result<Self, Error> {
        match value {
            "baseline-first" => Ok(Self::BaselineFirst),
            "variant-first" => Ok(Self::VariantFirst),
            _ => Err(Error::Body),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::BaselineFirst => "baseline-first",
            Self::VariantFirst => "variant-first",
        }
    }

    fn arrange(
        self,
        [baseline, variant]: [Request<Bytes>; 2],
    ) -> [(&'static str, Request<Bytes>); 2] {
        match self {
            Self::BaselineFirst => [("baseline", baseline), ("variant", variant)],
            Self::VariantFirst => [("variant", variant), ("baseline", baseline)],
        }
    }
}

#[derive(Clone, Copy)]
enum Variant {
    ToolResultString,
    ThinkingBindingAbsent,
    FinalErrorCacheAbsent,
    RequestClassMain,
    PreviousRequestId,
    NeutralDefaultIdentity,
}

impl Variant {
    fn parse(value: &str) -> Result<Self, Error> {
        match value {
            "tool_result_string" => Ok(Self::ToolResultString),
            "thinking_binding_absent" => Ok(Self::ThinkingBindingAbsent),
            "final_error_cache_absent" => Ok(Self::FinalErrorCacheAbsent),
            "request_class_main" => Ok(Self::RequestClassMain),
            "previous_request_id" => Ok(Self::PreviousRequestId),
            "neutral_default_identity" => Ok(Self::NeutralDefaultIdentity),
            _ => Err(Error::Body),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::ToolResultString => "tool_result_string",
            Self::ThinkingBindingAbsent => "thinking_binding_absent",
            Self::FinalErrorCacheAbsent => "final_error_cache_absent",
            Self::RequestClassMain => "request_class_main",
            Self::PreviousRequestId => "previous_request_id",
            Self::NeutralDefaultIdentity => "neutral_default_identity",
        }
    }

    fn apply(self, value: &mut Value) -> Result<(), Error> {
        match self {
            Self::ToolResultString => flatten_final_error(value),
            Self::FinalErrorCacheAbsent => remove_final_error_cache(value),
            Self::NeutralDefaultIdentity => neutral_default_identity(value),
            Self::RequestClassMain | Self::PreviousRequestId => Ok(()),
            Self::ThinkingBindingAbsent => value
                .get_mut("thinking")
                .and_then(Value::as_object_mut)
                .and_then(|thinking| thinking.shift_remove("block_binding"))
                .filter(|binding| !binding.is_null())
                .map(|_| ())
                .ok_or(Error::Body),
        }
    }
}

fn neutral_default_identity(value: &mut Value) -> Result<(), Error> {
    const ORIGINAL: &str = "You are rig, a coding agent";
    const FOLLOWING: &str = " working in the user's project from their terminal.";
    let blocks = value
        .get_mut("system")
        .and_then(Value::as_array_mut)
        .ok_or(Error::Body)?;
    let occurrences: usize = blocks
        .iter()
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .map(|text| text.matches(ORIGINAL).count())
        .sum();
    if occurrences != 1 {
        return Err(Error::Body);
    }
    let block = blocks
        .iter_mut()
        .find(|block| {
            block.get("type").and_then(Value::as_str) == Some("text")
                && block
                    .get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|text| text.starts_with(ORIGINAL))
        })
        .ok_or(Error::Body)?;
    let remainder = block
        .get("text")
        .and_then(Value::as_str)
        .and_then(|text| text.strip_prefix(ORIGINAL))
        .filter(|text| text.starts_with(FOLLOWING))
        .ok_or(Error::Body)?;
    block["text"] = format!("You are a coding agent{remainder}").into();
    Ok(())
}

fn previous_request_id(bytes: &[u8]) -> Result<String, Error> {
    if bytes.len() > 4096 {
        return Err(Error::Body);
    }
    let value: Value = serde_json::from_slice(bytes)?;
    let object = value
        .as_object()
        .filter(|value| value.len() == 1)
        .ok_or(Error::Body)?;
    let id = object
        .get("previous_request_id")
        .and_then(Value::as_str)
        .ok_or(Error::Body)?;
    let suffix = id.strip_prefix("req_").ok_or(Error::Body)?;
    if suffix.is_empty()
        || suffix.len() > 36
        || !suffix
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(Error::Body);
    }
    Ok(id.to_owned())
}

fn add_previous_request_id(value: &mut Value, id: &str) -> Result<(), Error> {
    previous_request_id(&serde_json::to_vec(&json!({"previous_request_id":id}))?)?;
    let text = value.pointer_mut("/system/0/text").ok_or(Error::Body)?;
    let billing = text.as_str().ok_or(Error::Body)?;
    if !billing.starts_with("x-anthropic-billing-header: cc_version=")
        || !billing.ends_with(';')
        || billing.contains("cc_prev_req=")
    {
        return Err(Error::Body);
    }
    *text = format!("{billing} cc_prev_req={id};").into();
    Ok(())
}

fn controlled_headers(baseline: &Request<Bytes>, variant: &Request<Bytes>, kind: Variant) -> bool {
    if baseline.uri() != variant.uri() || baseline.method() != variant.method() {
        return false;
    }
    let mut expected = baseline.headers().clone();
    if matches!(kind, Variant::RequestClassMain) {
        if expected.contains_key("x-claude-code-request-class") || baseline.body() != variant.body()
        {
            return false;
        }
        expected.insert(
            "x-claude-code-request-class",
            HeaderValue::from_static("main"),
        );
    }
    &expected == variant.headers()
}

fn remove_final_error_cache(value: &mut Value) -> Result<(), Error> {
    if validate_caching(value.as_object().ok_or(Error::Body)?)?.0 < 2 {
        return Err(Error::Body);
    }
    let message = value
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .and_then(|messages| messages.last_mut())
        .ok_or(Error::Body)?;
    if message.get("role").and_then(Value::as_str) != Some("user") {
        return Err(Error::Body);
    }
    let blocks = message
        .get_mut("content")
        .and_then(Value::as_array_mut)
        .ok_or(Error::Body)?;
    if blocks.len() != 1 {
        return Err(Error::Body);
    }
    let result = blocks.first_mut().ok_or(Error::Body)?;
    if result.get("type").and_then(Value::as_str) != Some("tool_result")
        || result.get("is_error") != Some(&Value::Bool(true))
    {
        return Err(Error::Body);
    }
    result
        .as_object_mut()
        .and_then(|result| result.shift_remove("cache_control"))
        .filter(|control| !control.is_null())
        .map(|_| ())
        .ok_or(Error::Body)
}

fn captured_request(path: &Path) -> ProbeResult<Vec<u8>> {
    if !path.is_absolute()
        || !path.starts_with("/dev/shm")
        || path
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
        || path.ancestors().any(|part| part.is_symlink())
    {
        return Err("capture must be a private regular file in /dev/shm".into());
    }
    let metadata = path
        .symlink_metadata()
        .map_err(|_| "capture metadata unavailable")?;
    let parent = path
        .parent()
        .ok_or("capture parent unavailable")?
        .metadata()
        .map_err(|_| "capture parent metadata unavailable")?;
    if !metadata.is_file()
        || metadata.permissions().mode() & 0o777 != 0o600
        || parent.permissions().mode() & 0o777 != 0o700
        || metadata.uid() != parent.uid()
        || metadata.uid()
            != std::fs::metadata("/proc/self")
                .map_err(|_| "process ownership unavailable")?
                .uid()
        || metadata.len() > 2 * 1024 * 1024
    {
        return Err("capture ownership, permissions or size invalid".into());
    }
    std::fs::read(path).map_err(|_| "capture read failed".into())
}

fn reconstructed(captured: &Value) -> Result<Value, Error> {
    if captured.get("model").and_then(Value::as_str) != Some("claude-opus-5-5")
        || captured.get("max_tokens").and_then(Value::as_u64) != Some(128_000)
        || captured.get("stream") != Some(&Value::Bool(true))
        || captured.get("_omp").is_some()
    {
        return Err(Error::Body);
    }
    let identity: Value = serde_json::from_str(
        captured
            .pointer("/metadata/user_id")
            .and_then(Value::as_str)
            .ok_or(Error::Body)?,
    )?;
    let session = identity
        .get("session_id")
        .and_then(Value::as_str)
        .ok_or(Error::Body)?;
    let billing = captured
        .pointer("/system/0/text")
        .and_then(Value::as_str)
        .ok_or(Error::Body)?;
    let suffix = billing
        .strip_prefix(&format!(
            "x-anthropic-billing-header: cc_version={VERSION}."
        ))
        .and_then(|rest| rest.split(';').next())
        .ok_or(Error::Body)?;
    if suffix.len() != 3
        || !suffix.bytes().all(|byte| byte.is_ascii_hexdigit())
        || captured.pointer("/system/1/text").and_then(Value::as_str) != Some(profile::IDENTITY)
        || captured.pointer("/system/0/cache_control").is_some()
        || captured.pointer("/system/1/cache_control").is_some()
    {
        return Err(Error::Body);
    }
    let mut source = captured.clone();
    let system = source
        .get_mut("system")
        .and_then(Value::as_array_mut)
        .ok_or(Error::Body)?;
    if system.len() < 2 {
        return Err(Error::Body);
    }
    system.drain(..2);
    let tools = source
        .get_mut("tools")
        .and_then(Value::as_array_mut)
        .ok_or(Error::Body)?;
    if tools.len() != 1 {
        return Err(Error::Body);
    }
    let tool = tools.first_mut().ok_or(Error::Body)?;
    if tool.get("name").and_then(Value::as_str) != Some("_read") {
        return Err(Error::Body);
    }
    tool["name"] = "read".into();
    for message in source
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .ok_or(Error::Body)?
    {
        if let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) {
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                    if block.get("name").and_then(Value::as_str) != Some("_read") {
                        return Err(Error::Body);
                    }
                    block["name"] = "read".into();
                }
            }
        }
    }
    if let Some(choice) = source.get_mut("tool_choice")
        && choice.get("name").is_some()
    {
        if choice.get("name").and_then(Value::as_str) != Some("_read") {
            return Err(Error::Body);
        }
        choice["name"] = "read".into();
    }
    source.as_object_mut().ok_or(Error::Body)?.insert(
        "_omp".into(),
        json!({"human_suffix":suffix,"session_id":session,
        "cache_retention":"none","explicit_cache_control":true,"agent_request":true}),
    );
    Ok(source)
}

fn flatten_final_error(value: &mut Value) -> Result<(), Error> {
    let message = value
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .and_then(|messages| messages.last_mut())
        .ok_or(Error::Body)?;
    if message.get("role").and_then(Value::as_str) != Some("user") {
        return Err(Error::Body);
    }
    let blocks = message
        .get_mut("content")
        .and_then(Value::as_array_mut)
        .ok_or(Error::Body)?;
    if blocks.len() != 1 {
        return Err(Error::Body);
    }
    let result = blocks.first_mut().ok_or(Error::Body)?;
    if result.get("type").and_then(Value::as_str) != Some("tool_result")
        || result.get("is_error") != Some(&Value::Bool(true))
    {
        return Err(Error::Body);
    }
    let content = result
        .get("content")
        .and_then(Value::as_array)
        .ok_or(Error::Body)?;
    if content.len() != 1 {
        return Err(Error::Body);
    }
    let text = content
        .first()
        .and_then(Value::as_object)
        .ok_or(Error::Body)?;
    if text.len() != 2 || text.get("type").and_then(Value::as_str) != Some("text") {
        return Err(Error::Body);
    }
    let text = text
        .get("text")
        .and_then(Value::as_str)
        .ok_or(Error::Body)?
        .to_owned();
    result["content"] = text.into();
    Ok(())
}

fn retag(mut bytes: Vec<u8>) -> Result<Vec<u8>, Error> {
    let start = bytes
        .windows(MARKER.len())
        .position(|window| window == MARKER.as_bytes())
        .ok_or(Error::Body)?
        + MARKER.len();
    let offset = bytes
        .get(start..)
        .ok_or(Error::Body)?
        .windows(4)
        .position(|window| window == b"cch=")
        .filter(|offset| *offset < 150)
        .ok_or(Error::Body)?
        + start
        + 4;
    bytes
        .get_mut(offset..offset + 5)
        .ok_or(Error::Body)?
        .copy_from_slice(b"00000");
    let checksum = profile::checksum::compute(&bytes);
    bytes
        .get_mut(offset..offset + 5)
        .ok_or(Error::Body)?
        .copy_from_slice(checksum.as_bytes());
    Ok(bytes)
}

fn probe_bodies(
    bytes: &[u8],
    variant: Variant,
    previous: Option<&str>,
) -> Result<[(Vec<u8>, Vec<u8>); 2], Error> {
    if matches!(variant, Variant::PreviousRequestId) != previous.is_some() {
        return Err(Error::Body);
    }
    let captured: Value = serde_json::from_slice(bytes)?;
    if retag(js_json(&captured)?)? != bytes {
        return Err(Error::Body);
    }
    validate_caching(captured.as_object().ok_or(Error::Body)?)?;
    let source = reconstructed(&captured)?;
    let baseline = serde_json::to_vec(&source)?;
    let mut variant_source = source;
    variant.apply(&mut variant_source)?;
    let mut variant_expected = captured;
    variant.apply(&mut variant_expected)?;
    if let Some(id) = previous {
        add_previous_request_id(&mut variant_expected, id)?;
    }
    Ok([
        (baseline, bytes.to_vec()),
        (
            serde_json::to_vec(&variant_source)?,
            retag(js_json(&variant_expected)?)?,
        ),
    ])
}

fn response_summary(bytes: &[u8], status: u16) -> Result<Value, Error> {
    let wire = std::str::from_utf8(bytes)
        .map_err(|_| Error::Body)?
        .replace("\r\n", "\n");
    let mut model_ok = false;
    let mut terminal = false;
    let mut reason = "unknown";
    let mut text = String::new();
    let mut output_tokens = None;
    for frame in wire.split("\n\n") {
        let payload = frame
            .lines()
            .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
            .collect::<Vec<_>>()
            .join("\n");
        if payload.is_empty() {
            continue;
        }
        let event: Value = serde_json::from_str(&payload)?;
        match event.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                model_ok = event.pointer("/message/model").and_then(Value::as_str)
                    == Some("claude-opus-5-5")
            }
            Some("message_delta") => {
                reason = match event.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    Some("refusal") => "refusal",
                    Some("end_turn") => "end_turn",
                    Some("tool_use") => "tool_use",
                    Some("max_tokens") => "max_tokens",
                    _ => "other",
                };
                terminal = event
                    .pointer("/delta/stop_reason")
                    .is_some_and(|value| !value.is_null());
                output_tokens = event
                    .pointer("/usage/output_tokens")
                    .and_then(Value::as_u64);
            }
            Some("content_block_delta")
                if event.pointer("/delta/type").and_then(Value::as_str) == Some("text_delta") =>
            {
                text.push_str(
                    event
                        .pointer("/delta/text")
                        .and_then(Value::as_str)
                        .ok_or(Error::Body)?,
                );
            }
            Some("error") => return Err(Error::Body),
            _ => (),
        }
    }
    let facts_all = [
        "amber-lantern-731",
        "violet-otter-482",
        "silver-pine-963",
        "copper-moon-254",
        "indigo-crane-615",
    ]
    .iter()
    .all(|fact| text.contains(fact));
    Ok(
        json!({"status":status,"served_model_ok":model_ok,"terminal":terminal,
        "stop_reason":reason,"nonempty_text":!text.trim().is_empty(),"output_tokens":output_tokens,
        "facts_all":facts_all}),
    )
}

#[test]
#[ignore = "requires an authorized private capture, stored credentials and two Messages attempts"]
fn real_tool_result_representation_probe() -> ProbeResult<()> {
    if std::env::var("CLAUDE_CLOAK_LIVE_TOOL_RESULT").as_deref() != Ok("1")
        || std::env::var("CLAUDE_CLOAK_LIVE_MODEL").as_deref() != Ok(MODEL)
    {
        return Err("tool result probe opt-in missing".into());
    }
    let path = PathBuf::from(
        std::env::var_os("CLAUDE_CLOAK_CAPTURED_REQUEST").ok_or("private capture path missing")?,
    );
    let report_path = PathBuf::from(
        std::env::var_os("CLAUDE_CLOAK_TOOL_RESULT_REPORT").ok_or("safe report path missing")?,
    );
    let bytes = captured_request(&path)?;
    let variant = Variant::parse(
        &std::env::var("CLAUDE_CLOAK_PROBE_VARIANT")
            .map_err(|_| "controlled probe variant missing")?,
    )
    .map_err(|_| "unknown controlled probe variant")?;
    let metadata_path = std::env::var_os("CLAUDE_CLOAK_PROBE_METADATA");
    if matches!(variant, Variant::PreviousRequestId) != metadata_path.is_some() {
        return Err("controlled probe metadata requirement mismatch".into());
    }
    let previous = metadata_path
        .map(|path| -> ProbeResult<String> {
            let metadata = captured_request(&PathBuf::from(path))?;
            previous_request_id(&metadata).map_err(|_| "controlled probe metadata invalid".into())
        })
        .transpose()?;
    let bodies = probe_bodies(&bytes, variant, previous.as_deref())
        .map_err(|_| "captured request reconstruction failed")?;
    let order = match std::env::var("CLAUDE_CLOAK_PROBE_ORDER") {
        Ok(value) => ProbeOrder::parse(&value).map_err(|_| "unknown probe order")?,
        Err(std::env::VarError::NotPresent) => ProbeOrder::BaselineFirst,
        Err(_) => return Err("invalid probe order".into()),
    };
    let report = bevy_tasks::block_on(async {
        rig_http::wasm_compat::timeout(
            Duration::from_secs(90),
            replay(bodies, variant, previous.as_deref(), order),
        )
        .await
        .map_err(|_| "tool result probe timed out")?
    })?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(report_path)
        .map_err(|_| "safe report creation failed")?;
    serde_json::to_writer(&mut file, &report).map_err(|_| "safe report write failed")?;
    Ok(())
}

async fn replay(
    bodies: [(Vec<u8>, Vec<u8>); 2],
    variant: Variant,
    previous: Option<&str>,
    order: ProbeOrder,
) -> ProbeResult<Value> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| "HTTP client initialization failed")?;
    let transport = OAuthTransport {
        inner: DynHttpClient::new(rig_reqwest::ReqwestClient::from(client)),
        device: device_id().map_err(|_| "stored device identity unavailable")?,
        session: uuid::Uuid::new_v4().to_string(),
        credentials: None,
        platform: None,
    };
    // Prepare and prove both bodies before forwarding either request.
    let mut requests: Vec<Request<Bytes>> = Vec::new();
    for (index, (source, expected)) in bodies.into_iter().enumerate() {
        let request = Request::post("https://api.anthropic.com/v1/messages")
            .body(Bytes::from(source))
            .map_err(|_| "probe request construction failed")?;
        let mut request = transport
            .prepare(request)
            .await
            .map_err(|_| "production request preparation failed")?;
        if matches!(variant, Variant::PreviousRequestId) && index == 1 {
            if requests
                .first()
                .is_none_or(|baseline| baseline.body() != request.body())
            {
                return Err("previous-request control did not reproduce baseline bytes".into());
            }
            let mut payload: Value =
                serde_json::from_slice(request.body()).map_err(|_| "prepared body invalid")?;
            add_previous_request_id(
                &mut payload,
                previous.ok_or("previous request metadata absent")?,
            )
            .map_err(|_| "previous request control invalid")?;
            *request.body_mut() =
                retag(js_json(&payload).map_err(|_| "controlled body encoding failed")?)
                    .map_err(|_| "controlled checksum failed")?
                    .into();
        }
        if request.headers().contains_key("content-encoding") || request.body().as_ref() != expected
        {
            return Err("prepared request differs from the controlled expected bytes".into());
        }
        if matches!(variant, Variant::RequestClassMain) {
            if request
                .headers()
                .contains_key("x-claude-code-request-class")
            {
                return Err("controlled request already contains the selected role header".into());
            }
            if index == 1 {
                request.headers_mut().insert(
                    "x-claude-code-request-class",
                    HeaderValue::from_static("main"),
                );
            }
        }
        if requests
            .first()
            .is_some_and(|previous| !controlled_headers(previous, &request, variant))
        {
            return Err("controlled request headers or destination differ".into());
        }
        requests.push(request);
    }
    let (baseline, alternative) = requests
        .first()
        .zip(requests.get(1))
        .ok_or("controlled request pair missing")?;
    let headers_equal = baseline.headers() == alternative.headers();
    let body_equal = baseline.body() == alternative.body();
    let requests = requests
        .try_into()
        .map_err(|_| "controlled request pair invalid")?;
    let mut outcomes = Vec::new();
    for (condition, request) in order.arrange(requests) {
        let mut response = transport
            .inner
            .send_streaming(request)
            .await
            .map_err(|_| "probe transport failed")?;
        let status = response.status().as_u16();
        let mut bytes = Vec::new();
        while let Some(chunk) =
            std::future::poll_fn(|cx| response.body_mut().as_mut().poll_next(cx)).await
        {
            let chunk = chunk.map_err(|_| "probe response stream failed")?;
            if bytes.len().saturating_add(chunk.len()) > 16 * 1024 * 1024 {
                return Err("probe response size exceeded".into());
            }
            bytes.extend_from_slice(&chunk);
        }
        let mut outcome = response_summary(&bytes, status)
            .map_err(|_| "probe response could not be summarized")?;
        outcome
            .as_object_mut()
            .ok_or("probe response summary invalid")?
            .insert("condition".into(), condition.into());
        outcomes.push(outcome);
    }
    Ok(
        json!({"baseline_byte_equal":true,"variant_only_declared_field_and_cch":true,
        "variant":variant.name(),"order":order.name(),"headers_equal":headers_equal,"body_equal":body_equal,
        "headers_only_declared_change":matches!(variant, Variant::RequestClassMain),
        "advertised_max_tokens":128000,"attempts":outcomes.len(),"outcomes":outcomes}),
    )
}

#[test]
fn reconstruction_preserves_every_other_field_and_rejects_extra_content() -> anyhow::Result<()> {
    let source = json!({"model":"claude-opus-5-5","max_tokens":128000,"stream":true,
        "system":[{"type":"text","text":"synthetic system"}],
        "tools":[{"name":"read","description":"read","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}],
        "messages":[{"role":"user","content":[{"type":"text","text":"read"}]},
            {"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"read","input":{"path":"absent.txt"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","is_error":true,"content":[{"type":"text","text":"no such file"}]}]}]});
    let original = body(
        &serde_json::to_vec(&source)?,
        "device",
        "session",
        Some("account"),
    )?;
    let pairs = probe_bodies(&original, Variant::ToolResultString, None)?;
    for (input, expected) in pairs {
        anyhow::ensure!(body(&input, "device", "session", Some("account"))?.as_ref() == expected);
    }
    let mut invalid: Value = serde_json::from_slice(&original)?;
    invalid
        .pointer_mut("/messages/2/content/0/content")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| anyhow::anyhow!("synthetic content"))?
        .push(json!({"type":"text","text":"extra"}));
    anyhow::ensure!(flatten_final_error(&mut invalid).is_err());
    Ok(())
}

#[test]
fn response_summary_preserves_refusal_and_checks_exact_synthetic_facts() -> anyhow::Result<()> {
    let events = [
        json!({"type":"message_start","message":{"model":"claude-opus-5-5"}}),
        json!({"type":"message_delta","delta":{"stop_reason":"refusal"},"usage":{"output_tokens":0}}),
    ];
    let encode = |events: &[Value]| {
        events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>()
    };
    let refusal = response_summary(encode(&events).as_bytes(), 200)?;
    anyhow::ensure!(refusal.get("stop_reason") == Some(&json!("refusal")));
    anyhow::ensure!(refusal.get("nonempty_text") == Some(&json!(false)));
    anyhow::ensure!(refusal.get("facts_all") == Some(&json!(false)));
    let events = [
        events
            .first()
            .ok_or_else(|| anyhow::anyhow!("synthetic start"))?
            .clone(),
        json!({"type":"content_block_delta","delta":{"type":"text_delta","text":"amber-lantern-731 violet-otter-482 silver-pine-963 copper-moon-254 indigo-crane-615"}}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}),
    ];
    let success = response_summary(encode(&events).as_bytes(), 200)?;
    anyhow::ensure!(success.get("stop_reason") == Some(&json!("end_turn")));
    anyhow::ensure!(success.get("facts_all") == Some(&json!(true)));
    Ok(())
}

#[test]
fn previous_request_control_changes_only_billing_field_and_checksum() -> anyhow::Result<()> {
    let source = json!({"model":"claude-opus-5-5","max_tokens":128000,"stream":true,
        "system":[{"type":"text","text":"synthetic system"}],
        "tools":[{"name":"read","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}],
        "messages":[{"role":"user","content":[{"type":"text","text":"read once"}]},
            {"role":"assistant","content":[{"type":"thinking","thinking":"synthetic thought","signature":"synthetic signature"},
                {"type":"tool_use","id":"call_1","name":"read","input":{"path":"absent.txt"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","is_error":true,
                "content":[{"type":"text","text":"no such file"}]}]}]});
    let original = body(
        &serde_json::to_vec(&source)?,
        "device",
        "session",
        Some("account"),
    )?;
    let id = "req_synthetic-previous_123";
    let pairs = probe_bodies(&original, Variant::PreviousRequestId, Some(id))?;
    let (baseline_source, baseline_expected) = pairs
        .first()
        .ok_or_else(|| anyhow::anyhow!("baseline missing"))?;
    let (variant_source, variant_expected) = pairs
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("variant missing"))?;
    anyhow::ensure!(baseline_expected == &original);
    anyhow::ensure!(baseline_source == variant_source);
    anyhow::ensure!(body(variant_source, "device", "session", Some("account"))? == original);
    let mut restored: Value = serde_json::from_slice(variant_expected)?;
    let text = restored
        .pointer_mut("/system/0/text")
        .ok_or_else(|| anyhow::anyhow!("billing missing"))?;
    let billing = text
        .as_str()
        .and_then(|text| text.strip_suffix(&format!(" cc_prev_req={id};")))
        .ok_or_else(|| anyhow::anyhow!("declared field absent"))?
        .to_owned();
    *text = billing.into();
    anyhow::ensure!(retag(js_json(&restored)?)? == original);
    anyhow::ensure!(retag(variant_expected.clone())? == *variant_expected);
    let mut duplicate: Value = serde_json::from_slice(variant_expected)?;
    anyhow::ensure!(add_previous_request_id(&mut duplicate, id).is_err());
    anyhow::ensure!(probe_bodies(&original, Variant::PreviousRequestId, None).is_err());
    anyhow::ensure!(probe_bodies(&original, Variant::ToolResultString, Some(id)).is_err());
    Ok(())
}

#[test]
fn previous_request_metadata_accepts_only_bounded_request_ids() -> anyhow::Result<()> {
    for id in [
        "req_synthetic-123_ok".to_owned(),
        format!("req_{}", "a".repeat(36)),
    ] {
        anyhow::ensure!(
            previous_request_id(&serde_json::to_vec(&json!({"previous_request_id":id}))?)? == id
        );
    }
    for id in [
        "req_".to_owned(),
        "msg_synthetic".to_owned(),
        "req_a; injected=true".to_owned(),
        "req_a\nb".to_owned(),
        "req_é".to_owned(),
        format!("req_{}", "a".repeat(37)),
    ] {
        anyhow::ensure!(
            previous_request_id(&serde_json::to_vec(&json!({"previous_request_id":id}))?).is_err()
        );
    }
    for value in [
        json!({}),
        json!({"previous_request_id":null}),
        json!({"previous_request_id":"req_test","extra":true}),
        json!(["req_test"]),
    ] {
        anyhow::ensure!(previous_request_id(&serde_json::to_vec(&value)?).is_err());
    }
    anyhow::ensure!(previous_request_id(&vec![b' '; 4097]).is_err());
    Ok(())
}

#[test]
fn neutral_identity_control_changes_only_the_default_prefix_and_checksum() -> anyhow::Result<()> {
    let prompt = "You are rig, a coding agent working in the user's project from their terminal.\nRead only the requested file. Keep all functional instructions.";
    let source = json!({"model":"claude-opus-5-5","max_tokens":128000,"stream":true,
        "system":[{"type":"text","text":prompt}],
        "tools":[{"name":"read","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}],
        "messages":[{"role":"user","content":[{"type":"text","text":"You are rig, a coding agent: quoted user content stays unchanged."}]}]});
    let original = body(
        &serde_json::to_vec(&source)?,
        "device",
        "session",
        Some("account"),
    )?;
    let pairs = probe_bodies(&original, Variant::NeutralDefaultIdentity, None)?;
    for (input, expected) in &pairs {
        anyhow::ensure!(&body(input, "device", "session", Some("account"))? == expected);
    }
    let (_, expected) = pairs
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("variant missing"))?;
    let mut restored: Value = serde_json::from_slice(expected)?;
    anyhow::ensure!(
        restored.pointer("/system/1/text").and_then(Value::as_str) == Some(profile::IDENTITY)
    );
    let text = restored
        .pointer_mut("/system/2/text")
        .ok_or_else(|| anyhow::anyhow!("default prompt missing"))?;
    let remainder = text
        .as_str()
        .and_then(|text| text.strip_prefix("You are a coding agent"))
        .ok_or_else(|| anyhow::anyhow!("neutral prefix missing"))?;
    *text = format!("You are rig, a coding agent{remainder}").into();
    anyhow::ensure!(retag(js_json(&restored)?)? == original);
    for system in [
        json!([]),
        json!([{"type":"text","text":"not the default prompt"}]),
        json!([{"type":"text","text":format!("quoted: {prompt}")}]),
        json!([{"type":"text","text":"You are rig, a coding agent with a different role."}]),
        json!([{"type":"text","text":prompt},{"type":"text","text":prompt}]),
        json!([{"type":"text","text":format!("{prompt}\n{prompt}")}]),
    ] {
        let mut invalid = json!({"system":system});
        anyhow::ensure!(neutral_default_identity(&mut invalid).is_err());
    }
    Ok(())
}

#[test]
fn probe_order_reverses_only_sending_and_keeps_conditions_attached() -> anyhow::Result<()> {
    for (name, expected_first) in [("baseline-first", "baseline"), ("variant-first", "variant")] {
        let baseline = Request::post("https://api.anthropic.com/v1/messages")
            .header("x-synthetic", "baseline")
            .body(Bytes::from_static(b"baseline-body"))?;
        let variant = Request::post("https://api.anthropic.com/v1/messages")
            .header("x-synthetic", "variant")
            .body(Bytes::from_static(b"variant-body"))?;
        let order = ProbeOrder::parse(name)?;
        anyhow::ensure!(order.name() == name);
        let arranged = order.arrange([baseline, variant]);
        anyhow::ensure!(arranged.first().map(|(condition, _)| *condition) == Some(expected_first));
        for (condition, request) in arranged {
            anyhow::ensure!(
                request
                    .headers()
                    .get("x-synthetic")
                    .and_then(|value| value.to_str().ok())
                    == Some(condition)
            );
            anyhow::ensure!(request.body().as_ref() == format!("{condition}-body").as_bytes());
        }
    }
    anyhow::ensure!(ProbeOrder::parse("arbitrary").is_err());
    Ok(())
}

#[test]
fn binding_variant_preserves_signed_history_and_rejects_missing_binding() -> anyhow::Result<()> {
    let mut source = json!({"model":"claude-opus-5-5","max_tokens":128000,"stream":true,
        "system":[{"type":"text","text":"synthetic system"}],
        "thinking":{"type":"adaptive","block_binding":{"prefix_mismatch_behavior":"drop_block"}},
        "tools":[{"name":"read","description":"read","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}],
        "messages":[{"role":"user","content":[{"type":"text","text":"read"}]},
            {"role":"assistant","content":[{"type":"thinking","thinking":"synthetic thought","signature":"signed-test"},
                {"type":"tool_use","id":"call_1","name":"read","input":{"path":"absent.txt"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","is_error":true,"content":[{"type":"text","text":"no such file"}]}]}]});
    let original = body(
        &serde_json::to_vec(&source)?,
        "device",
        "session",
        Some("account"),
    )?;
    let original_value: Value = serde_json::from_slice(&original)?;
    for (input, expected) in probe_bodies(&original, Variant::ThinkingBindingAbsent, None)? {
        let actual = body(&input, "device", "session", Some("account"))?;
        anyhow::ensure!(actual.as_ref() == expected);
        let value: Value = serde_json::from_slice(&actual)?;
        anyhow::ensure!(value.get("messages") == original_value.get("messages"));
        anyhow::ensure!(value.get("tools") == original_value.get("tools"));
    }
    Variant::ThinkingBindingAbsent.apply(&mut source)?;
    anyhow::ensure!(Variant::ThinkingBindingAbsent.apply(&mut source).is_err());
    anyhow::ensure!(Variant::parse("arbitrary_body_change").is_err());
    Ok(())
}

#[test]
fn final_error_cache_variant_keeps_other_anchors_and_signed_history() -> anyhow::Result<()> {
    let source = json!({"model":"claude-opus-5-5","max_tokens":128000,"stream":true,
        "system":[{"type":"text","text":"synthetic system"}],
        "tools":[{"name":"read","description":"read","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}],
        "messages":[{"role":"user","content":[{"type":"text","text":"read"}]},
            {"role":"assistant","content":[{"type":"thinking","thinking":"synthetic thought","signature":"signed-test"},
                {"type":"tool_use","id":"call_1","name":"read","input":{"path":"absent.txt"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","is_error":true,"content":[{"type":"text","text":"no such file"}]}]}]});
    let original = body(
        &serde_json::to_vec(&source)?,
        "device",
        "session",
        Some("account"),
    )?;
    let original_value: Value = serde_json::from_slice(&original)?;
    anyhow::ensure!(markers(&original_value) == 4);
    let pairs = probe_bodies(&original, Variant::FinalErrorCacheAbsent, None)?;
    for (input, expected) in &pairs {
        anyhow::ensure!(body(input, "device", "session", Some("account"))?.as_ref() == expected);
    }
    let (_, expected) = pairs
        .last()
        .ok_or_else(|| anyhow::anyhow!("synthetic variant"))?;
    let mut variant: Value = serde_json::from_slice(expected)?;
    anyhow::ensure!(markers(&variant) == 3);
    anyhow::ensure!(variant.pointer("/messages/1") == original_value.pointer("/messages/1"));
    anyhow::ensure!(variant.get("tools") == original_value.get("tools"));
    let control = original_value
        .pointer("/messages/2/content/0/cache_control")
        .ok_or_else(|| anyhow::anyhow!("synthetic final marker"))?
        .clone();
    variant
        .pointer_mut("/messages/2/content/0")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| anyhow::anyhow!("synthetic final result"))?
        .insert("cache_control".into(), control);
    anyhow::ensure!(variant.get("messages") == original_value.get("messages"));
    let mut one_anchor = json!({"messages":[{"role":"user","content":[{"type":"tool_result","is_error":true,"content":"error","cache_control":{"type":"ephemeral","ttl":"1h"}}]}]});
    anyhow::ensure!(
        Variant::FinalErrorCacheAbsent
            .apply(&mut one_anchor)
            .is_err()
    );
    one_anchor.as_object_mut().ok_or_else(|| anyhow::anyhow!("synthetic body"))?
        .insert("tools".into(), json!([{"name":"read","input_schema":{"type":"object","properties":{"cache_control":{"type":"string"}}}}]));
    anyhow::ensure!(markers(&one_anchor) == 2);
    anyhow::ensure!(
        Variant::FinalErrorCacheAbsent
            .apply(&mut one_anchor)
            .is_err()
    );
    Ok(())
}

#[test]
fn request_class_variant_requires_exact_body_and_only_declared_header_change() -> anyhow::Result<()>
{
    let baseline = Request::post("https://api.anthropic.com/v1/messages?beta=true")
        .header("anthropic-beta", "synthetic-beta")
        .header("authorization", "synthetic-authorization")
        .body(Bytes::from_static(b"unchanged body and checksum"))?;
    let mut variant = baseline.clone();
    variant.headers_mut().insert(
        "x-claude-code-request-class",
        HeaderValue::from_static("main"),
    );
    anyhow::ensure!(controlled_headers(
        &baseline,
        &variant,
        Variant::RequestClassMain
    ));
    anyhow::ensure!(!controlled_headers(
        &baseline,
        &variant,
        Variant::ToolResultString
    ));
    anyhow::ensure!(!controlled_headers(
        &variant,
        &variant,
        Variant::RequestClassMain
    ));
    *variant.body_mut() = Bytes::from_static(b"changed body");
    anyhow::ensure!(!controlled_headers(
        &baseline,
        &variant,
        Variant::RequestClassMain
    ));
    *variant.body_mut() = baseline.body().clone();
    variant
        .headers_mut()
        .insert("anthropic-beta", HeaderValue::from_static("changed-beta"));
    anyhow::ensure!(!controlled_headers(
        &baseline,
        &variant,
        Variant::RequestClassMain
    ));
    Ok(())
}

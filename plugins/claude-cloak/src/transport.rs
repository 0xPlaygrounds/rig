//! Subscriber wire compatibility for the isolated Anthropic OAuth connection.
//! Device identity persists per installation. Direct calls without an agent's
//! saved session identity use a process-local session id.

use rig_harness::rig_core::{
    http_client::{
        self, Bytes, DynHttpClient, HeaderValue, HttpClientExt, LazyBody, MultipartForm, Request,
        Response, StreamingResponse, Uri,
    },
    wasm_compat::WasmCompatSend,
};
use rig_reqwest::reqwest;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{future::Future, io::Write, path::PathBuf, sync::OnceLock};

use crate::profile::{self, BETAS, VERSION};
const MARKER: &str = "\"system\":[{\"type\":\"text\",\"text\":\"x-anthropic-billing-header:";

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("OAuth transport only permits HTTPS api.anthropic.com/v1/messages")]
    Destination,
    #[error("OAuth transport does not support multipart requests")]
    Multipart,
    #[error("invalid Anthropic request body")]
    Body,
    #[error("OAuth device identity is unavailable")]
    Identity,
    #[error("Anthropic requests permit at most four cache breakpoints")]
    CacheBudget,
    #[error("Anthropic one-hour cache breakpoints must precede five-minute breakpoints")]
    CacheOrder,
    #[error("Anthropic automatic caching and its final explicit breakpoint must use the same TTL")]
    CacheTtl,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Client(#[from] reqwest::Error),
}

#[derive(Clone)]
struct OAuthTransport {
    inner: DynHttpClient,
    device: String,
    session: String,
    #[cfg(test)]
    credentials: Option<(String, Option<String>)>,
    #[cfg(test)]
    platform: Option<(&'static str, &'static str)>,
}

fn platform() -> (&'static str, &'static str) {
    let os = match std::env::consts::OS {
        "macos" => "MacOS",
        "windows" => "Windows",
        "freebsd" => "FreeBSD",
        _ => "Linux",
    };
    let arch = match std::env::consts::ARCH {
        "aarch64" => "arm64",
        "x86" => "x86",
        _ => "x64",
    };
    (os, arch)
}

pub(crate) fn new() -> Result<DynHttpClient, Error> {
    static SESSION: OnceLock<String> = OnceLock::new();
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    Ok(DynHttpClient::new(OAuthTransport {
        inner: DynHttpClient::new(rig_reqwest::ReqwestClient::from(client)),
        #[cfg(test)]
        credentials: None,
        #[cfg(test)]
        platform: None,
        device: device_id()?,
        session: SESSION
            .get_or_init(|| uuid::Uuid::new_v4().to_string())
            .clone(),
    }))
}

fn device_id() -> Result<String, Error> {
    let home = std::env::var_os("RIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".rig")))
        .ok_or(Error::Identity)?;
    let dir = home.join("anthropic-oauth");
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
        let mut builder = std::fs::DirBuilder::new();
        builder.mode(0o700);
        match builder.create(&dir) {
            Ok(()) => (),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
            Err(e) => return Err(e.into()),
        }
        let metadata = std::fs::symlink_metadata(&dir)?;
        if !metadata.is_dir() || metadata.permissions().mode() & 0o077 != 0 {
            return Err(Error::Identity);
        }
    }
    #[cfg(not(unix))]
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("device-id");
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(&path) {
        Ok(mut file) => {
            let id = format!("{:x}", Sha256::digest(uuid::Uuid::new_v4().as_bytes()));
            file.write_all(id.as_bytes())?;
            file.sync_all()?;
            Ok(id)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let metadata = std::fs::symlink_metadata(&path)?;
            if !metadata.is_file() {
                return Err(Error::Identity);
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if metadata.permissions().mode() & 0o077 != 0 {
                    return Err(Error::Identity);
                }
            }
            let id = std::fs::read_to_string(path)?;
            if id.len() != 64 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(Error::Identity);
            }
            Ok(id)
        }
        Err(error) => Err(error.into()),
    }
}

fn validate(uri: &Uri) -> Result<(), Error> {
    if uri.scheme_str() != Some("https")
        || uri.host() != Some("api.anthropic.com")
        || uri.port_u16().is_some_and(|port| port != 443)
        || uri.authority().is_some_and(|a| a.as_str().contains('@'))
        || uri.path() != "/v1/messages"
    {
        return Err(Error::Destination);
    }
    Ok(())
}

fn prefix_name(value: &mut Value) {
    if let Some(Value::String(name)) = value.get_mut("name")
        && !matches!(
            name.to_ascii_lowercase().as_str(),
            "web_search" | "code_execution" | "text_editor" | "computer"
        )
    {
        *name = format!("_{name}");
    }
}

fn prefix_tools(value: &mut Value) {
    if let Some(tools) = value.get_mut("tools").and_then(Value::as_array_mut) {
        for tool in tools {
            if tool.get("name").and_then(Value::as_str) == Some("yield")
                && tool.get("eager_input_streaming").is_some()
            {
                tool["eager_input_streaming"] = false.into();
            }
            prefix_name(tool);
        }
    }
    if let Some(choice) = value.get_mut("tool_choice") {
        prefix_name(choice);
    }
    if let Some(messages) = value.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages {
            if let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) {
                for block in blocks {
                    if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                        prefix_name(block);
                    }
                    if block.get("type").and_then(Value::as_str) == Some("tool_result")
                        && let Some(object) = block.as_object_mut()
                    {
                        object.entry("is_error").or_insert(Value::Bool(false));
                    }
                }
            }
        }
    }
}

fn first_user_text(body: &Value) -> String {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return String::new();
    };
    for message in messages {
        if !matches!(
            message.get("role").and_then(Value::as_str),
            Some("user" | "developer")
        ) {
            continue;
        }
        let Some(content) = message.get("content") else {
            continue;
        };
        if let Some(text) = content.as_str().filter(|text| !text.trim().is_empty()) {
            return text.to_owned();
        }
        if let Some(blocks) = content.as_array() {
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(text) = block
                        .get("text")
                        .and_then(Value::as_str)
                        .filter(|text| !text.trim().is_empty())
                {
                    return text.to_owned();
                }
            }
            if blocks
                .iter()
                .any(|b| b.get("type").and_then(Value::as_str) == Some("image"))
            {
                return String::new();
            }
        }
    }
    String::new()
}

fn normalize_messages(object: &mut serde_json::Map<String, Value>) {
    let Some(messages) = object.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    let mut normalized = Vec::new();
    for mut message in std::mem::take(messages) {
        if message.get("role").and_then(Value::as_str) != Some("user") {
            normalized.push(message);
            continue;
        }
        let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) else {
            normalized.push(message);
            continue;
        };
        let mut runs: Vec<Vec<Value>> = Vec::new();
        for block in std::mem::take(blocks) {
            let result = block.get("type").and_then(Value::as_str) == Some("tool_result");
            let same = runs.last().and_then(|r| r.last()).is_some_and(|b| {
                (b.get("type").and_then(Value::as_str) == Some("tool_result")) == result
            });
            if !same {
                runs.push(Vec::new());
            }
            if let Some(run) = runs.last_mut() {
                run.push(block);
            }
        }
        for run in runs {
            let mut message = message.clone();
            let content = match run.as_slice() {
                [block]
                    if block.as_object().is_some_and(|o| o.len() == 2)
                        && block.get("type").and_then(Value::as_str) == Some("text") =>
                {
                    block.get("text").cloned().unwrap_or(Value::Null)
                }
                _ => Value::Array(run),
            };
            if let Some(message) = message.as_object_mut() {
                message.insert("content".into(), content);
            }
            normalized.push(message);
        }
    }
    *messages = normalized;
}

fn js_json(value: &Value) -> Result<Vec<u8>, Error> {
    fn encode(value: &Value, output: &mut String) -> Result<(), Error> {
        match value {
            Value::Number(number) => {
                let number = number.as_f64().ok_or(Error::Body)?;
                output.push_str(ryu_js::Buffer::new().format(number));
            }
            Value::Array(values) => {
                output.push('[');
                for (i, value) in values.iter().enumerate() {
                    if i > 0 {
                        output.push(',');
                    }
                    encode(value, output)?;
                }
                output.push(']');
            }
            Value::Object(values) => {
                let index = |key: &str| {
                    key.parse::<u32>()
                        .ok()
                        .filter(|i| *i != u32::MAX && i.to_string() == key)
                };
                let mut entries: Vec<_> = values.iter().collect();
                entries.sort_by_key(|(key, _)| index(key).map_or((1, 0), |i| (0, i)));
                output.push('{');
                for (i, (key, value)) in entries.into_iter().enumerate() {
                    if i > 0 {
                        output.push(',');
                    }
                    output.push_str(&serde_json::to_string(key)?);
                    output.push(':');
                    encode(value, output)?;
                }
                output.push('}');
            }
            _ => output.push_str(&serde_json::to_string(value)?),
        }
        Ok(())
    }
    let mut output = String::new();
    encode(value, &mut output)?;
    Ok(output.into_bytes())
}

fn cache_message(message: &mut Value, control: &Value) -> bool {
    let Some(content) = message.get_mut("content") else {
        return false;
    };
    if let Some(text) = content.as_str() {
        *content = json!([{"type":"text","text":text,"cache_control":control}]);
        return true;
    }
    let Some(blocks) = content.as_array_mut() else {
        return false;
    };
    for block in blocks.iter_mut().rev() {
        if matches!(
            block.get("type").and_then(Value::as_str),
            Some("thinking" | "redacted_thinking" | "fallback" | "tool_addition" | "tool_removal")
        ) {
            continue;
        }
        if block.get("cache_control").is_some_and(|v| !v.is_null()) {
            return false;
        }
        block["cache_control"] = control.clone();
        return true;
    }
    false
}

fn cache_blocks(object: &serde_json::Map<String, Value>) -> impl Iterator<Item = &Value> {
    ["tools", "system"]
        .into_iter()
        .filter_map(|key| object.get(key).and_then(Value::as_array))
        .flatten()
        .chain(
            object
                .get("messages")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|message| message.get("content").and_then(Value::as_array))
                .flatten(),
        )
}

fn validate_caching(object: &serde_json::Map<String, Value>) -> Result<(usize, bool), Error> {
    let mut short_seen = false;
    let mut long_seen = false;
    let mut existing = 0;
    for control in cache_blocks(object)
        .filter_map(|block| block.get("cache_control"))
        .chain(object.get("cache_control"))
        .filter(|v| !v.is_null())
    {
        existing += 1;
        if control.get("ttl").and_then(Value::as_str) == Some("1h") {
            if short_seen {
                return Err(Error::CacheOrder);
            }
            long_seen = true;
        } else {
            short_seen = true;
        }
    }
    if existing > 4 {
        return Err(Error::CacheBudget);
    }
    if let Some(root) = object.get("cache_control").filter(|v| !v.is_null()) {
        let target = object
            .get("messages")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .rev()
            .find_map(|message| {
                let content = message.get("content")?;
                if content.is_string() {
                    Some(content)
                } else {
                    content.as_array()?.iter().rev().find(|block| {
                        !matches!(
                            block.get("type").and_then(Value::as_str),
                            Some(
                                "thinking"
                                    | "redacted_thinking"
                                    | "fallback"
                                    | "tool_addition"
                                    | "tool_removal"
                            )
                        )
                    })
                }
            })
            .or_else(|| object.get("system")?.as_array()?.last())
            .or_else(|| {
                object
                    .get("tools")?
                    .as_array()?
                    .iter()
                    .rev()
                    .find(|tool| tool.get("defer_loading") != Some(&Value::Bool(true)))
            });
        if let Some(control) = target
            .and_then(|block| block.get("cache_control"))
            .filter(|v| !v.is_null())
            && (control.get("ttl").and_then(Value::as_str) == Some("1h"))
                != (root.get("ttl").and_then(Value::as_str) == Some("1h"))
        {
            return Err(Error::CacheTtl);
        }
    }
    Ok((existing, long_seen))
}

fn prompt_caching(
    object: &mut serde_json::Map<String, Value>,
    options: &Value,
) -> Result<(), Error> {
    // A raw root control delegates the moving breakpoint to the server.
    let (existing, long_seen) = validate_caching(object)?;
    if object.get("cache_control").is_some_and(|v| !v.is_null()) {
        return Ok(());
    }
    // Caller breakpoints reserve capacity before the automatic head and tail anchors.
    let mut budget = 4 - existing;
    let control = match options.get("cache_retention").and_then(Value::as_str) {
        Some("none") => return Ok(()),
        Some("short") if !long_seen => json!({"type":"ephemeral"}),
        _ => json!({"type":"ephemeral","ttl":"1h"}),
    };
    if budget > 0
        && let Some(tools) = object.get_mut("tools").and_then(Value::as_array_mut)
        && !tools
            .iter()
            .any(|t| t.get("cache_control").is_some_and(|v| !v.is_null()))
        && let Some(tool) = tools
            .iter_mut()
            .rev()
            .find(|t| t.get("defer_loading") != Some(&Value::Bool(true)))
    {
        tool["cache_control"] = control.clone();
        budget -= 1;
    }
    let short_head = object
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|block| block.get("cache_control"))
        .any(|v| !v.is_null() && v.get("ttl").and_then(Value::as_str) != Some("1h"));
    let system_control = if short_head {
        json!({"type":"ephemeral"})
    } else {
        control.clone()
    };
    if budget > 0
        && let Some(system) = object.get_mut("system").and_then(Value::as_array_mut)
    {
        let suffix = system
            .iter()
            .position(|b| {
                b.get("text").and_then(Value::as_str).is_some_and(|s| {
                    s.starts_with("<memories>") || s.starts_with("<project-context>")
                })
            })
            .unwrap_or(system.len());
        let anchor = if suffix == 0 {
            system.len().checked_sub(1)
        } else {
            Some(suffix - 1)
        };
        if let Some(anchor) = anchor
            && !system
                .iter()
                .any(|b| b.get("cache_control").is_some_and(|v| !v.is_null()))
            && let Some(block) = system.get_mut(anchor)
        {
            block["cache_control"] = system_control;
            budget -= 1;
        }
    }
    let short_head = ["tools", "system"]
        .into_iter()
        .filter_map(|key| object.get(key).and_then(Value::as_array))
        .flatten()
        .filter_map(|block| block.get("cache_control"))
        .any(|v| !v.is_null() && v.get("ttl").and_then(Value::as_str) != Some("1h"));
    let Some(messages) = object.get_mut("messages").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    let is_index = |key: &str, i: usize| {
        options
            .get(key)
            .and_then(Value::as_array)
            .is_some_and(|a| a.iter().any(|v| v.as_u64() == Some(i as u64)))
    };
    let conversational = |i: usize, m: &Value| {
        if options.get("conversational_users").is_some() {
            return is_index("conversational_users", i);
        }
        m.get("role").and_then(Value::as_str) == Some("user")
            && (m.get("content").is_some_and(Value::is_string)
                || m.get("content").and_then(Value::as_array).is_some_and(|b| {
                    b.iter().any(|b| {
                        matches!(
                            b.get("type").and_then(Value::as_str),
                            Some("text" | "image")
                        )
                    })
                }))
    };
    let mut trailing = Vec::new();
    let mut mark = None;
    let mut turns = 0usize;
    for (i, message) in messages.iter().enumerate().rev() {
        if message.get("clear_at").and_then(Value::as_str) == Some("next_user_message")
            || is_index("per_call_messages", i)
        {
            mark = Some(i);
            turns = 0;
            continue;
        }
        let tool_control = message.get("role").and_then(Value::as_str) == Some("system")
            && message
                .get("content")
                .and_then(Value::as_array)
                .is_some_and(|a| {
                    !a.is_empty()
                        && a.iter().all(|b| {
                            matches!(
                                b.get("type").and_then(Value::as_str),
                                Some("tool_addition" | "tool_removal")
                            )
                        })
                });
        if trailing.len() < 2 && !tool_control {
            trailing.push(i);
        }
        if conversational(i, message) {
            turns += 1;
        }
    }
    let mut candidates: Vec<usize> = trailing.first().copied().into_iter().collect();
    let end = mark.unwrap_or(messages.len());
    if turns >= 15 {
        let mut since = turns % 15;
        for (i, message) in messages.iter().enumerate().take(end).rev() {
            if !conversational(i, message) {
                continue;
            }
            if since > 0 {
                since -= 1;
                continue;
            }
            since = 14;
            candidates.push(i);
        }
    }
    if let Some(i) = mark.and_then(|i| i.checked_sub(1)) {
        candidates.push(i);
    }
    candidates.extend(trailing);
    for i in candidates {
        if budget == 0 {
            break;
        }
        let short_prefix = short_head
            || messages
                .iter()
                .take(i + 1)
                .filter_map(|m| m.get("content").and_then(Value::as_array))
                .flatten()
                .filter_map(|b| b.get("cache_control"))
                .any(|v| !v.is_null() && v.get("ttl").and_then(Value::as_str) != Some("1h"));
        let control = if short_prefix {
            json!({"type":"ephemeral"})
        } else {
            control.clone()
        };
        if let Some(message) = messages.get_mut(i)
            && cache_message(message, &control)
        {
            budget -= 1;
        }
    }
    Ok(())
}

fn body(bytes: &[u8], device: &str, session: &str, account: Option<&str>) -> Result<Bytes, Error> {
    let mut value: Value = serde_json::from_slice(bytes)?;
    let suffix = match value.pointer("/_omp/human_suffix") {
        Some(value) => {
            let suffix = value.as_str().ok_or(Error::Body)?;
            if suffix.len() != 3 || !suffix.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(Error::Body);
            }
            suffix.to_owned()
        }
        None => profile::suffix(&first_user_text(&value)),
    };
    let billing = format!(
        "x-anthropic-billing-header: cc_version={VERSION}.{suffix}; cc_entrypoint={}; cch=00000;",
        profile::ENTRYPOINT
    );
    crate::schema::normalize_tools(&mut value);
    prefix_tools(&mut value);
    let object = value.as_object_mut().ok_or(Error::Body)?;
    let options = object.shift_remove("_omp").unwrap_or(Value::Null);
    let had_output_config = object.contains_key("output_config");
    if options.get("cache_retention").is_some()
        && options.get("explicit_cache_control") != Some(&Value::Bool(true))
    {
        object.shift_remove("cache_control");
    }
    normalize_messages(object);
    let thinking = object
        .get("thinking")
        .and_then(|v| v.get("type"))
        .and_then(Value::as_str);
    if options.get("sampling_allowed") == Some(&Value::Bool(false))
        || thinking.is_some_and(|t| t != "disabled")
    {
        for key in ["temperature", "top_p", "top_k"] {
            object.shift_remove(key);
        }
    }
    if object.contains_key("compaction") {
        object.shift_remove("stop_sequences");
    }
    if let Some(stops) = object
        .get_mut("stop_sequences")
        .and_then(Value::as_array_mut)
    {
        stops.truncate(4);
        if stops.is_empty() {
            object.shift_remove("stop_sequences");
        }
    }
    if (options.get("forced_tools_allowed") == Some(&Value::Bool(false))
        || object.contains_key("compaction"))
        && matches!(
            object
                .get("tool_choice")
                .and_then(|v| v.get("type"))
                .and_then(Value::as_str),
            Some("any" | "tool")
        )
    {
        object.insert("tool_choice".into(), json!({"type":"auto"}));
    }
    if matches!(
        object
            .get("tool_choice")
            .and_then(|v| v.get("type"))
            .and_then(Value::as_str),
        Some("any" | "tool")
    ) {
        object.shift_remove("thinking");
        if let Some(edits) = object
            .get_mut("context_management")
            .and_then(|v| v.get_mut("edits"))
            .and_then(Value::as_array_mut)
        {
            edits.retain(|e| e.get("type").and_then(Value::as_str) == Some("compact_20260112"));
            if edits.is_empty() {
                object.shift_remove("context_management");
            }
        }
        if options.get("adaptive_only") == Some(&Value::Bool(true)) {
            object.entry("output_config").or_insert_with(|| json!({}))["effort"] = "low".into();
        } else if let Some(output) = object
            .get_mut("output_config")
            .and_then(Value::as_object_mut)
        {
            output.shift_remove("effort");
            if output.is_empty() {
                object.shift_remove("output_config");
            }
        }
    }
    if object
        .get("thinking")
        .and_then(|v| v.get("type"))
        .and_then(Value::as_str)
        == Some("enabled")
        && let Some(budget) = options
            .get("thinking_budget")
            .and_then(Value::as_u64)
            .or_else(|| {
                object
                    .get("thinking")
                    .and_then(|v| v.get("budget_tokens"))
                    .and_then(Value::as_u64)
            })
            .filter(|b| *b > 0)
    {
        let ceiling = options
            .get("max_output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(64_000);
        let current = object
            .get("max_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(ceiling)
            .min(ceiling);
        let maximum = current.max(budget.saturating_add(4000)).min(ceiling);
        let adjusted = budget.min(maximum.saturating_sub(4000));
        if adjusted == 0 {
            return Err(Error::Body);
        }
        object.insert("max_tokens".into(), maximum.into());
        if let Some(thinking) = object.get_mut("thinking") {
            thinking["budget_tokens"] = adjusted.into();
        }
    }
    let session = options
        .get("session_id")
        .and_then(Value::as_str)
        .unwrap_or(session);
    if options.get("tool_choice_explicit") == Some(&Value::Bool(false))
        && object.get("tool_choice") == Some(&json!({"type":"auto"}))
    {
        object.shift_remove("tool_choice");
    }
    if matches!(
        object
            .get("thinking")
            .and_then(|v| v.get("type"))
            .and_then(Value::as_str),
        Some("adaptive" | "enabled")
    ) && !object.contains_key("compaction")
    {
        object
            .entry("context_management")
            .or_insert_with(|| json!({"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}));
    }
    if let Some(thinking) = object.get_mut("thinking").and_then(Value::as_object_mut)
        && matches!(
            thinking.get("type").and_then(Value::as_str),
            Some("adaptive" | "enabled")
        )
    {
        thinking
            .entry("display")
            .or_insert_with(|| "updates".into());
    }
    let existing = object.shift_remove("system");
    let has_billing = match &existing {
        Some(Value::String(text)) => text.starts_with("x-anthropic-billing-header:"),
        Some(Value::Array(blocks)) => blocks.iter().any(|b| {
            b.get("text")
                .and_then(Value::as_str)
                .is_some_and(|text| text.starts_with("x-anthropic-billing-header:"))
        }),
        _ => false,
    };
    if has_billing {
        return Err(Error::Body);
    }
    // Cache placement adds the identity anchor only when caller breakpoints leave room.
    let mut system = vec![
        json!({"type":"text","text":billing}),
        json!({"type":"text","text":profile::IDENTITY}),
    ];
    match existing {
        Some(Value::Array(blocks)) => system.extend(blocks),
        Some(Value::String(text)) => system.push(json!({"type":"text","text":text})),
        Some(Value::Null) | None => (),
        _ => return Err(Error::Body),
    }
    object.insert("system".into(), Value::Array(system));
    object.entry("tools").or_insert_with(|| json!([]));
    let derived_device = match account.filter(|a| !a.is_empty()) {
        Some(account) => format!(
            "{:x}",
            Sha256::digest(format!("omp-claude-device-id-v2\0{device}\0{account}"))
        ),
        None => format!(
            "{:x}",
            Sha256::digest(format!("omp-claude-device-id-v1:{device}"))
        ),
    };
    let mut identity = json!({"device_id":derived_device,"session_id":session});
    if let Some(account) = account.filter(|s| !s.is_empty()) {
        identity
            .as_object_mut()
            .ok_or(Error::Body)?
            .insert("account_uuid".into(), account.into());
    }
    object.insert(
        "metadata".into(),
        json!({"user_id":serde_json::to_string(&identity)?}),
    );
    prompt_caching(object, &options)?;
    validate_caching(object)?;
    let mut ordered = serde_json::Map::new();
    for key in [
        "model",
        "messages",
        "system",
        "tools",
        "metadata",
        "max_tokens",
        "thinking",
        "context_management",
        "compaction",
        "output_config",
        "fallbacks",
        "anthropic_beta",
        "stream",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "speed",
        "tool_choice",
    ] {
        if key == "output_config" && !had_output_config {
            continue;
        }
        if let Some(value) = object.shift_remove(key) {
            ordered.insert(key.into(), value);
        }
    }
    if let Some(output) = object.shift_remove("output_config") {
        ordered.insert("output_config".into(), output);
    }
    ordered.extend(std::mem::take(object));
    let mut wire = js_json(&Value::Object(ordered))?;
    let anchor = wire
        .windows(MARKER.len())
        .position(|w| w == MARKER.as_bytes())
        .ok_or(Error::Body)?
        + MARKER.len();
    let offset = wire
        .get(anchor..)
        .ok_or(Error::Body)?
        .windows(9)
        .position(|w| w == b"cch=00000")
        .filter(|i| *i <= 150)
        .ok_or(Error::Body)?
        + anchor
        + 4;
    let cch = profile::checksum::compute(&wire);
    wire.get_mut(offset..offset + 5)
        .ok_or(Error::Body)?
        .copy_from_slice(cch.as_bytes());
    Ok(wire.into())
}

impl OAuthTransport {
    #[allow(clippy::result_large_err)]
    async fn prepare(&self, mut req: Request<Bytes>) -> http_client::Result<Request<Bytes>> {
        validate(req.uri()).map_err(http_client::Error::instance)?;
        #[cfg(test)]
        let credentials = match &self.credentials {
            Some(credentials) => Ok(credentials.clone()),
            None => crate::auth::credentials_for_request().await,
        };
        #[cfg(not(test))]
        let credentials = crate::auth::credentials_for_request().await;
        let (token, account) = credentials.map_err(http_client::Error::instance)?;
        let original: Value =
            serde_json::from_slice(req.body()).map_err(http_client::Error::instance)?;
        let thinking_requested = original
            .pointer("/_omp/thinking_requested")
            .and_then(Value::as_bool)
            .unwrap_or_else(|| {
                matches!(
                    original.pointer("/thinking/type").and_then(Value::as_str),
                    Some("enabled" | "adaptive")
                )
            });
        *req.body_mut() = body(req.body(), &self.device, &self.session, account.as_deref())
            .map_err(http_client::Error::instance)?;
        *req.uri_mut() = Uri::from_static("https://api.anthropic.com/v1/messages?beta=true");
        let payload: Value =
            serde_json::from_slice(req.body()).map_err(http_client::Error::instance)?;
        let agent = original
            .pointer("/_omp/agent_request")
            .and_then(Value::as_bool)
            == Some(true)
            || thinking_requested
            || payload
                .get("tools")
                .and_then(Value::as_array)
                .is_some_and(|tools| !tools.is_empty())
            || payload
                .get("thinking")
                .and_then(|v| v.get("type"))
                .and_then(Value::as_str)
                .is_some_and(|t| t != "disabled");
        let betas = if agent { BETAS } else { profile::UTILITY_BETAS };
        let mut betas: Vec<String> = betas.split(',').map(str::to_owned).collect();
        let forced = matches!(
            payload.pointer("/tool_choice/type").and_then(Value::as_str),
            Some("any" | "tool")
        );
        let thinking = matches!(
            payload.pointer("/thinking/type").and_then(Value::as_str),
            Some("enabled" | "adaptive")
        );
        let effort_beta = thinking_requested
            || (!forced && (thinking || payload.pointer("/output_config/effort").is_some()));
        if effort_beta {
            betas.push("effort-2025-11-24".into());
        }
        if payload.get("fallback_credit_token").is_some() {
            betas.push("fallback-credit-2026-06-01".into());
        }
        if thinking {
            betas.push("thinking-binding-controls-2026-08-01".into());
            if payload.pointer("/thinking/display").and_then(Value::as_str) == Some("updates") {
                betas.push("thinking-display-updates-2026-08-18".into());
            }
        }
        if !effort_beta && forced && payload.pointer("/output_config/effort").is_some() {
            betas.push("effort-2025-11-24".into());
        }
        for header in req.headers().get_all("anthropic-beta") {
            for beta in header
                .to_str()
                .map_err(http_client::Error::instance)?
                .split(',')
                .map(str::trim)
            {
                if !beta.is_empty()
                    && beta != "context-1m-2025-08-07"
                    && !betas.iter().any(|existing| existing == beta)
                {
                    betas.push(beta.to_owned());
                }
            }
        }
        let (os, arch) = platform();
        #[cfg(test)]
        let (os, arch) = self.platform.unwrap_or((os, arch));
        let headers = req.headers_mut();
        for key in [
            "x-api-key",
            "authorization",
            "content-length",
            "host",
            "cf-aig-authorization",
            "x-client-request-id",
        ] {
            headers.remove(key);
        }
        for (key, value) in [
            ("accept", "application/json"),
            ("content-type", "application/json"),
            ("user-agent", profile::USER_AGENT),
            ("x-stainless-lang", "js"),
            ("x-stainless-package-version", profile::SDK),
            ("x-stainless-retry-count", "0"),
            ("x-stainless-runtime", "node"),
            ("x-stainless-runtime-version", profile::RUNTIME),
            ("x-stainless-timeout", "600"),
            ("anthropic-dangerous-direct-browser-access", "true"),
            ("anthropic-version", "2023-06-01"),
            ("x-app", "cli"),
            ("connection", "keep-alive"),
            ("accept-encoding", "gzip, deflate, br, zstd"),
            ("x-stainless-os", os),
            ("x-stainless-arch", arch),
        ] {
            headers.insert(key, HeaderValue::from_static(value));
        }
        headers.insert(
            "anthropic-beta",
            HeaderValue::from_str(&betas.join(",")).map_err(http_client::Error::instance)?,
        );
        headers.insert(
            "x-claude-code-session-id",
            HeaderValue::from_str(
                payload
                    .pointer("/metadata/user_id")
                    .and_then(Value::as_str)
                    .and_then(|s| serde_json::from_str::<Value>(s).ok())
                    .and_then(|v| {
                        v.get("session_id")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    })
                    .as_deref()
                    .unwrap_or(&self.session),
            )
            .map_err(http_client::Error::instance)?,
        );
        let mut bearer = HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(http_client::Error::instance)?;
        bearer.set_sensitive(true);
        headers.insert("authorization", bearer);
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
        if compress_request(req.body(), custom_network) {
            let mut encoder = flate2::GzBuilder::new()
                .operating_system(3)
                .write(Vec::new(), flate2::Compression::new(6));
            encoder
                .write_all(req.body())
                .map_err(http_client::Error::instance)?;
            *req.body_mut() = encoder
                .finish()
                .map_err(http_client::Error::instance)?
                .into();
            req.headers_mut()
                .insert("content-encoding", HeaderValue::from_static("gzip"));
        }
        Ok(req)
    }
}

fn compress_request(bytes: &[u8], custom_network: bool) -> bool {
    !custom_network
        && std::str::from_utf8(bytes).is_ok_and(|text| {
            text.encode_utf16().take(profile::GZIP_THRESHOLD).count() == profile::GZIP_THRESHOLD
        })
}

impl HttpClientExt for OAuthTransport {
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
            let req = this.prepare(req).await?;
            this.inner.send(req).await
        }
    }
    #[allow(clippy::manual_async_fn)]
    fn send_multipart<U>(
        &self,
        _req: Request<MultipartForm>,
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
            let req = self.prepare(req).await?;
            self.inner.send_streaming(req).await
        }
    }
}

#[cfg(test)]
mod capture;
#[cfg(test)]
mod expiry;
#[cfg(test)]
mod live;
#[cfg(test)]
mod roundtrip;
#[cfg(test)]
mod tests;

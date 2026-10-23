use super::*;
use rig_harness::rig_core::completion::Effort;

#[cfg(target_os = "linux")]
mod tool_result;

#[test]
fn root_ttl_guard_checks_final_shape_and_limits_attempts() -> anyhow::Result<()> {
    let value = json!({
        "model":"claude-opus-5-5", "max_tokens":2048, "tools":[],
        "cache_control":{"type":"ephemeral","ttl":"5m"},
        "messages":[{"role":"user","content":[{"type":"text","text":"READY"}]}]
    });
    let attempts = AtomicUsize::new(0);
    inspect_root_ttl(&value, &attempts)?;
    inspect_root_ttl(&value, &attempts)?;
    anyhow::ensure!(inspect_root_ttl(&value, &attempts).is_err());
    let mut invalid = value;
    invalid
        .pointer_mut("/messages/0/content/0")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| anyhow::anyhow!("synthetic text block"))?
        .insert(
            "cache_control".into(),
            json!({"type":"ephemeral","ttl":"1h"}),
        );
    anyhow::ensure!(inspect_root_ttl(&invalid, &AtomicUsize::new(0)).is_err());
    Ok(())
}

#[test]
#[ignore = "requires separately authorized stored credentials, two Messages attempts and 90 seconds"]
fn real_root_ttl_regression_and_control() -> ProbeResult<()> {
    if std::env::var("CLAUDE_CLOAK_LIVE_ROOT_TTL").as_deref() != Ok("1")
        || std::env::var("CLAUDE_CLOAK_LIVE_MODEL").as_deref() != Ok(MODEL)
    {
        return Err("set CLAUDE_CLOAK_LIVE_ROOT_TTL=1 and CLAUDE_CLOAK_LIVE_MODEL=claude-cloak/claude-opus-5-5".into());
    }
    bevy_tasks::block_on(async {
        rig_http::wasm_compat::timeout(Duration::from_secs(90), root_ttl_probes())
            .await
            .map_err(|_| "root TTL live probes timed out")?
    })
}

async fn root_ttl_probes() -> ProbeResult<()> {
    let attempts = Arc::new(AtomicUsize::new(0));
    let records = Arc::new(Mutex::new(Vec::new()));
    let handler = audited_handler_with_root_ttl(records.clone(), Some(attempts.clone()))?;
    for (id, retention, name) in [
        (0, CacheRetention::Long, "raw-root-5m-long-regression"),
        (1, CacheRetention::Short, "raw-root-5m-short-control"),
    ] {
        let request = CompletionRequest::new("Reply exactly READY. Do not use tools.")
            .reasoning(Effort::Low)
            .max_tokens(2048)
            .cache(retention)
            .additional_params(json!({"cache_control":{"type":"ephemeral","ttl":"5m"}}));
        let response = complete(&handler, request, true, id).await?;
        if response.raw.get("model").and_then(Value::as_str) != Some("claude-opus-5-5")
            || response.raw.get("stop_reason").and_then(Value::as_str) != Some("end_turn")
            || response.provider_request_id.is_none()
            || response.text().trim() != "READY"
        {
            return Err(
                "root TTL probe did not return the requested model, end_turn and READY".into(),
            );
        }
        report(name, &response, 1);
    }
    let records = records.lock().map_err(|_| "root TTL audit unavailable")?;
    if attempts.load(Ordering::SeqCst) != 2
        || records.len() != 2
        || records
            .iter()
            .any(|(markers, status)| *markers != 1 || *status != 200)
    {
        return Err(
            "root TTL audit did not observe exactly two successful single-marker requests".into(),
        );
    }
    Ok(())
}

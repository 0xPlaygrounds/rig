use super::*;

/// A structural live TUI capture keeps the current profile independent of OMP fixtures.
#[tokio::test]
async fn primary_request_matches_current_profile_control() -> anyhow::Result<()> {
    let reference: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/claude-code/profile-2.1.307.json"
    )))?;
    let transport = OAuthTransport {
        inner: DynHttpClient::new(rig_reqwest::ReqwestClient::from(reqwest::Client::new())),
        device: "synthetic-device".into(),
        session: "synthetic-session".into(),
        credentials: Some(("synthetic-token".into(), None)),
        platform: Some(("Linux", "arm64")),
    };
    let source = json!({
        "_omp":{"agent_request":true},"model":"claude-opus-5-5","max_tokens":128000,
        "messages":[{"role":"user","content":"Return READY."}],
        "thinking":{"type":"adaptive"},"output_config":{"effort":"low"},"stream":true
    });
    let prepared = transport
        .prepare(
            Request::builder()
                .uri("https://api.anthropic.com/v1/messages")
                .body(Bytes::from(serde_json::to_vec(&source)?))?,
        )
        .await?;
    for (key, expected) in reference
        .get("headers")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("reference headers"))?
    {
        anyhow::ensure!(
            prepared
                .headers()
                .get(key)
                .and_then(|value| value.to_str().ok())
                == expected.as_str(),
            "profile header {key}"
        );
    }
    let mut betas: Vec<_> = prepared
        .headers()
        .get("anthropic-beta")
        .ok_or_else(|| anyhow::anyhow!("beta header"))?
        .to_str()?
        .split(',')
        .map(Value::from)
        .collect();
    betas.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
    anyhow::ensure!(reference.get("betas") == Some(&Value::Array(betas)));
    anyhow::ensure!(!prepared.headers().contains_key("content-encoding"));
    let body: Value = serde_json::from_slice(prepared.body())?;
    for key in ["thinking", "context_management", "output_config"] {
        anyhow::ensure!(body.get(key) == reference.get(key), "profile field {key}");
    }
    anyhow::ensure!(body.get("_omp").is_none());
    Ok(())
}

/// Agent provenance selects the primary beta profile even with thinking and tools off.
#[tokio::test]
async fn agent_request_remains_primary_without_thinking_or_tools() -> anyhow::Result<()> {
    let transport = OAuthTransport {
        inner: DynHttpClient::new(rig_reqwest::ReqwestClient::from(reqwest::Client::new())),
        device: "synthetic-device".into(),
        session: "synthetic-session".into(),
        credentials: Some(("synthetic-token".into(), None)),
        platform: None,
    };
    for agent in [false, true] {
        let prepared = transport.prepare(Request::builder()
            .uri("https://api.anthropic.com/v1/messages")
            .body(Bytes::from(serde_json::to_vec(&json!({
                "_omp":{"agent_request":agent},"messages":[{"role":"user","content":"hello"}]
            }))?))?).await?;
        let betas = prepared
            .headers()
            .get("anthropic-beta")
            .ok_or_else(|| anyhow::anyhow!("betas"))?
            .to_str()?;
        anyhow::ensure!(betas.split(',').any(|beta| beta == "claude-code-20250219") == agent);
        anyhow::ensure!(!betas.contains("thinking-display-updates"));
    }
    Ok(())
}

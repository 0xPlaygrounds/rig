use super::*;
use rig_harness::rig_core::message::ToolChoice;
use serde_json::json;

#[test]
fn implicit_and_explicit_auto_remain_distinguishable_after_encoding() -> anyhow::Result<()> {
    let mut implicit = CompletionRequest::new("hello");
    let mut explicit = implicit.clone();
    explicit.tool_choice = Some(ToolChoice::Auto);
    anyhow::ensure!(annotate(&mut implicit));
    anyhow::ensure!(annotate(&mut explicit));
    anyhow::ensure!(
        (implicit
            .additional_params
            .as_ref()
            .and_then(|v| v.pointer("/_omp/tool_choice_explicit")))
            == (Some(&json!(false)))
    );
    anyhow::ensure!(
        (explicit
            .additional_params
            .as_ref()
            .and_then(|v| v.pointer("/_omp/tool_choice_explicit")))
            == (Some(&json!(true)))
    );
    Ok(())
}

#[test]
fn cache_policy_preserves_explicit_retention_and_session() -> anyhow::Result<()> {
    for (retention, expected) in [
        (None, "long"),
        (Some(CacheRetention::None), "none"),
        (Some(CacheRetention::Short), "short"),
        (Some(CacheRetention::Long), "long"),
    ] {
        let mut request = CompletionRequest::new("hello");
        request.options.cache = retention;
        request.additional_params =
            Some(json!({"_omp":{"session_id":"session-fixture"},"temperature":0.4}));
        anyhow::ensure!(annotate(&mut request));
        let Some(params) = request.additional_params else {
            anyhow::bail!("annotation supplies parameters")
        };
        anyhow::ensure!((params.pointer("/_omp/cache_retention")) == (Some(&json!(expected))));
        anyhow::ensure!((params.pointer("/_omp/session_id")) == (Some(&json!("session-fixture"))));
        anyhow::ensure!((params.get("temperature")) == (Some(&json!(0.4))));
    }
    Ok(())
}

#[test]
fn raw_tool_choice_is_explicit_and_invalid_parameters_are_rejected() -> anyhow::Result<()> {
    let mut request = CompletionRequest::new("hello");
    request.additional_params = Some(json!({"tool_choice":{"type":"auto"}}));
    anyhow::ensure!(annotate(&mut request));
    anyhow::ensure!(
        (request
            .additional_params
            .as_ref()
            .and_then(|v| v.pointer("/_omp/tool_choice_explicit")))
            == (Some(&json!(true)))
    );
    request.additional_params = Some(json!([1, 2]));
    anyhow::ensure!(!annotate(&mut request));
    Ok(())
}

#[test]
fn thinking_intent_survives_forced_tools_and_raw_overrides_win() -> anyhow::Result<()> {
    let mut request = CompletionRequest::new("hello");
    request.tool_choice = Some(ToolChoice::Required);
    request.options.reasoning = Some(Reasoning::Budget { tokens: 1024 });
    anyhow::ensure!(annotate(&mut request));
    anyhow::ensure!(
        (request
            .additional_params
            .as_ref()
            .and_then(|p| p.pointer("/_omp/thinking_requested")))
            == (Some(&json!(true)))
    );
    request.additional_params = Some(json!({"thinking":{"type":"disabled"}}));
    anyhow::ensure!(annotate(&mut request));
    anyhow::ensure!(
        (request
            .additional_params
            .as_ref()
            .and_then(|p| p.pointer("/_omp/thinking_requested")))
            == (Some(&json!(false)))
    );
    Ok(())
}

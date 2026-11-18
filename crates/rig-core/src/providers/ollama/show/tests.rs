use super::ShowResponse;

/// The window `reply`, a `/api/show` body, names.
fn window(reply: &str) -> Option<u32> {
    serde_json::from_str::<ShowResponse>(reply)
        .ok()
        .and_then(|reply| reply.window())
}

#[test]
fn the_served_num_ctx_wins_over_the_trained_length() {
    let info = r#""model_info": {"general.architecture": "qwen3", "qwen3.context_length": 40960}"#;
    let set = format!(
        r#"{{"parameters": "temperature 0.6\nnum_ctx                        8192", {info}}}"#
    );
    assert_eq!(window(&set), Some(8192));
    assert_eq!(window(&format!("{{{info}}}")), Some(40960));
    let other =
        r#"{"model_info": {"general.architecture": "qwen3", "llama.context_length": 4096}}"#;
    assert_eq!(window(other), None);
    assert_eq!(window(r#"{"parameters": "num_ctx 0"}"#), None);
}

use super::*;

#[test]
fn usage_is_read_from_every_provider_shape() {
    assert_eq!(
        usage(r#"{"usage":{"prompt_tokens":10,"completion_tokens":3}}"#),
        Some((10, 3))
    );
    assert_eq!(
        usage(r#"{"usage":{"input_tokens":7,"output_tokens":2}}"#),
        Some((7, 2))
    );
    // Anthropic cache reads count as input, writes at 1.25 times.
    assert_eq!(
        usage(
            r#"{"usage":{"input_tokens":7,"cache_read_input_tokens":100,"cache_creation_input_tokens":8,"output_tokens":2}}"#
        ),
        Some((117, 2))
    );
    assert_eq!(
        usage(
            r#"{"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"thoughtsTokenCount":4}}"#
        ),
        Some((5, 5))
    );
    assert_eq!(
        usage(r#"{"prompt_eval_count":8,"eval_count":6}"#),
        Some((8, 6))
    );
    let stream = "data: {\"type\":\"response.created\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":4,\"output_tokens\":9}}}\n\n";
    assert_eq!(usage(stream), Some((4, 9)));
    assert_eq!(usage(r#"{"text":"no usage"}"#), None);
}

fn cassette(request: &str, response: &str) -> String {
    format!(
        "when:\n  path: /v1/chat/completions\n  body: '{request}'\nthen:\n  status: 200\n  body: '{response}'\n"
    )
}

#[test]
fn a_cassette_costs_its_usage_or_its_bound() {
    let known = cassette(
        "{}",
        r#"{"usage":{"prompt_tokens":1000000,"completion_tokens":0}}"#,
    );
    let (cost, unknown) = cassette_cost("openai", &known);
    assert!((cost - 2.5).abs() < 1e-9, "{cost}");
    assert_eq!(unknown, 0);

    // No usage: the request at three bytes a token plus the whole budget.
    let bounded = cassette(r#"{"max_tokens":1000000}"#, "{}");
    let (cost, unknown) = cassette_cost("openai", &bounded);
    assert!(cost >= 10.0, "{cost}");
    assert_eq!(unknown, 1);

    assert_eq!(cassette_cost("llamacpp", &bounded).0, 0.0);

    // Image routes report no tokens and are priced per image.
    let image =
        "when:\n  path: /api/v1/image/generate\n  body: '{}'\nthen:\n  status: 200\n  body: '{}'\n";
    assert!(cassette_cost("venice", image).0 >= 0.05);
    assert!(is_image_call(
        "/v1beta/models/gemini-2.5-flash-image:generateContent"
    ));
    assert!(!is_image_call(
        "/v1beta/models/gemini-2.5-flash:generateContent"
    ));
}

#[test]
fn failed_attempts_cost_at_least_the_floor_and_unrun_ones_nothing() {
    let dir = std::env::temp_dir().join(format!("xtask-spend-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("groq")).expect("dir");
    std::fs::write(
        dir.join("groq/cheap.yaml"),
        cassette(
            "{}",
            r#"{"usage":{"prompt_tokens":10,"completion_tokens":1}}"#,
        ),
    )
    .expect("fixture");
    let ledger = format!(
        "{}\n\
         groq\tgroq/cheap.yaml\tgroq::cheap\t1\tstarted\t-\n\
         groq\tgroq/cheap.yaml\tgroq::cheap\t1\t1\tfailed\n\
         groq\tgroq/cheap.yaml\tgroq::cheap\t2\tstarted\t-\n\
         groq\tgroq/cheap.yaml\tgroq::cheap\t2\t0\trecorded\n\
         groq\tgroq/cheap.yaml\tgroq::interrupted\t1\tstarted\t-\n\
         groq\tgroq/cheap.yaml\t-\t0\tskipped\towner unknown\n\
         ollama\tollama/missing.yaml\tollama::x\t1\t1\tfailed\n",
        super::super::record::LEDGER_HEADER
    );
    let (per_provider, _) = spend(&ledger, &dir);
    let groq = per_provider["groq"];
    // One failed and one interrupted attempt at the floor, one recorded at cost.
    assert!(
        groq > 2.0 * FAILED_ATTEMPT_FLOOR && groq < 2.0 * FAILED_ATTEMPT_FLOOR + 0.001,
        "{groq}"
    );
    assert_eq!(per_provider["ollama"], 0.0);
    let _ = std::fs::remove_dir_all(&dir);
}

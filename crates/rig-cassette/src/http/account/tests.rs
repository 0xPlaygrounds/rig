use super::*;

// Bodies as providers sent them in recorded replies, with account
// identifiers replaced.

#[test]
fn a_spent_workspace_quota_is_a_quota_failure_despite_its_400() {
    let anthropic = r#"{"type":"error","error":{"type":"invalid_request_error","message":"You have reached your specified workspace API usage limits. You will regain access on 2026-10-01 at 00:00 UTC."},"request_id":"req_011"}"#;
    assert_eq!(account_failure(400, anthropic), Some(AccountFailure::Quota));
    let openai = r#"{"error":{"message":"You exceeded your current quota, please check your plan and billing details.","type":"insufficient_quota","param":null,"code":"insufficient_quota"}}"#;
    assert_eq!(account_failure(429, openai), Some(AccountFailure::Quota));
    let gemini = r#"{"error":{"code":429,"message":"You exceeded your current quota, please check your plan and billing details.","status":"RESOURCE_EXHAUSTED"}}"#;
    assert_eq!(account_failure(429, gemini), Some(AccountFailure::Quota));
}

#[test]
fn per_minute_limits_are_rate_limit_failures() {
    let cases = [
        r#"{"error":{"message":"Rate limit reached for model `openai/gpt-oss-20b` in organization `org_x` service tier `on_demand` on tokens per minute (TPM)","type":"tokens","code":"rate_limit_exceeded"}}"#,
        r#"{"id":"0b02","message":"You are using a Trial key, which is limited to 20 API calls / minute."}"#,
        r#"{"error":{"message":"Request rate limit exceeded, please try again later.","type":"request_rate_limit_exceeded","code":429}}"#,
        r#"{"object":"error","message":"Rate limit exceeded","type":"rate_limited","param":null,"code":"1300"}"#,
        r#"{"type":"error","error":{"type":"rate_limit_error","message":"Number of request tokens has exceeded your per-minute rate limit"}}"#,
    ];
    for body in cases {
        assert_eq!(
            account_failure(429, body),
            Some(AccountFailure::RateLimit),
            "{body}"
        );
    }
}

#[test]
fn an_empty_balance_is_a_credit_failure() {
    let deepseek = r#"{"error":{"message":"Insufficient Balance","type":"unknown_error","param":null,"code":"invalid_request_error"}}"#;
    assert_eq!(account_failure(402, deepseek), Some(AccountFailure::Credit));
    let anthropic = r#"{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API."}}"#;
    assert_eq!(
        account_failure(400, anthropic),
        Some(AccountFailure::Credit)
    );
}

#[test]
fn a_refused_credential_is_an_auth_failure() {
    assert_eq!(
        account_failure(
            401,
            r#"{"error":{"message":"Incorrect API key provided: sk-inval***","type":"invalid_request_error","code":"invalid_api_key"}}"#
        ),
        Some(AccountFailure::Auth)
    );
    let gemini_key = r#"{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"API_KEY_INVALID"}]}}"#;
    assert_eq!(account_failure(400, gemini_key), Some(AccountFailure::Auth));
    // Doubleword's reply to a rejected key, as recorded.
    let doubleword = r#"{"error":{"code":"forbidden","message":"Forbidden","param":null,"type":"invalid_request_error"}}"#;
    assert_eq!(account_failure(403, doubleword), Some(AccountFailure::Auth));
}

#[test]
fn a_gateway_relaying_its_upstreams_limit_is_classified_by_the_upstream() {
    let openrouter = r#"{"error":{"message":"Provider returned error","code":400,"metadata":{"raw":"{\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\",\"message\":\"slow down\"}}","provider_name":"Anthropic"}}}"#;
    assert_eq!(
        account_failure(400, openrouter),
        Some(AccountFailure::RateLimit)
    );
}

#[test]
fn request_errors_are_not_account_failures() {
    let cases = [
        (
            403,
            r#"{"error":{"code":403,"message":"CachedContent not found (or permission denied)","status":"PERMISSION_DENIED"}}"#,
        ),
        (
            404,
            r#"{"type":"error","error":{"type":"not_found_error","message":"model: claude-nonexistent-rig-test"}}"#,
        ),
        (
            400,
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"Too many strict tools (21). The maximum number of strict tools supported is 20."}}"#,
        ),
        (
            400,
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"Schemas contains too many optional parameters (25), which would make grammar compilation inefficient."}}"#,
        ),
        (
            400,
            r#"{"error":{"message":"Provider returned error","code":400,"metadata":{"raw":"{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"messages.1.content.0: Invalid `signature` in `thinking` block\"}}"}}}"#,
        ),
        (
            529,
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        ),
        (
            500,
            "[json.exception.parse_error.101] parse error at line 1",
        ),
    ];
    for (status, body) in cases {
        assert_eq!(account_failure(status, body), None, "{status} {body}");
    }
}

#[test]
fn success_is_never_an_account_failure() {
    assert_eq!(
        account_failure(200, r#"{"text":"we hit a rate limit yesterday"}"#),
        None
    );
}

#[test]
fn xai_out_of_credits_is_a_credit_failure() {
    let body = r#"{"code":"Some resource has been exhausted","error":"Your team 00000000-0000-0000-0000-000000000000 has either used all available credits or reached its monthly spending limit. To continue making API requests, please purchase more credits or raise your spending limit."}"#;
    assert_eq!(account_failure(429, body), Some(AccountFailure::Credit));
}

#[test]
fn a_limit_delivered_inside_a_successful_reply_is_classified() {
    // OpenRouter reports an upstream limit as an event of a 200 stream.
    let stream = "data: {\"id\":\"gen-1\",\"choices\":[]}\n\n\
data: {\"error\":{\"code\":429,\"message\":\"Provider returned error\"}}\n\n";
    assert_eq!(
        reply_account_failure(200, stream),
        Some(AccountFailure::RateLimit)
    );
    let failed = "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_1\",\"error\":{\"code\":\"insufficient_quota\",\"message\":\"You exceeded your current quota\"}}}\n\n";
    assert_eq!(
        reply_account_failure(200, failed),
        Some(AccountFailure::Quota)
    );
    let anthropic = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\",\"message\":\"Number of request tokens has exceeded your per-minute rate limit\"}}\n\n";
    assert_eq!(
        reply_account_failure(200, anthropic),
        Some(AccountFailure::RateLimit)
    );
    // A successful Responses object carries `"error": null`; a request error
    // inside a stream is not an account failure.
    assert_eq!(
        reply_account_failure(200, r#"{"id":"resp_1","error":null,"output":[]}"#),
        None
    );
    assert_eq!(
        reply_account_failure(
            200,
            "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"
        ),
        None
    );
    // Gemini's non-SSE stream is a JSON array of chunks.
    assert_eq!(
        reply_account_failure(
            200,
            r#"[{"candidates":[]},{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"Quota exceeded for metric"}}]"#
        ),
        Some(AccountFailure::Quota)
    );
    assert_eq!(reply_account_failure(200, r#"[{"candidates":[]}]"#), None);
    // An error status defers to account_failure.
    assert_eq!(
        reply_account_failure(429, r#"{"error":{"type":"rate_limit_error"}}"#),
        Some(AccountFailure::RateLimit)
    );
}

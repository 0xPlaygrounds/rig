use super::*;

fn ids(created: &[CreatedResource]) -> Vec<(&str, &str)> {
    created
        .iter()
        .map(|resource| (resource.id.as_str(), resource.delete_url.as_str()))
        .collect()
}

#[test]
fn a_stored_response_is_created_unless_the_request_opts_out() {
    let origin = "https://api.openai.com";
    let reply = br#"{"id":"resp_1","object":"response"}"#;
    assert_eq!(
        ids(&created_resources(
            "openai",
            origin,
            "POST",
            "/v1/responses",
            b"{}",
            reply
        )),
        [("resp_1", "https://api.openai.com/v1/responses/resp_1")]
    );
    assert!(
        created_resources(
            "openai",
            origin,
            "POST",
            "/v1/responses",
            br#"{"store":false}"#,
            reply
        )
        .is_empty()
    );
    // Only OpenAI and xAI store: OpenRouter and local servers keep nothing.
    for (provider, path) in [
        ("llamacpp", "/v1/responses"),
        ("mistralrs", "/v1/responses"),
        ("copilot", "/responses"),
    ] {
        assert!(
            created_resources(provider, origin, "POST", path, b"{}", reply).is_empty(),
            "{provider}"
        );
    }
    assert!(
        created_resources(
            "openrouter",
            origin,
            "POST",
            "/api/v1/responses",
            b"{}",
            reply
        )
        .is_empty()
    );
}

#[test]
fn a_stream_names_the_response_on_its_creation_event() {
    let stream = b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_s\"}}\n\n\
event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_s\"}}\n\n";
    assert_eq!(
        ids(&created_resources(
            "xai",
            "https://api.x.ai",
            "POST",
            "/v1/responses",
            b"{}",
            stream
        )),
        [("resp_s", "https://api.x.ai/v1/responses/resp_s")]
    );
}

#[test]
fn files_caches_and_interactions_map_to_their_delete_urls() {
    assert_eq!(
        ids(&created_resources(
            "anthropic",
            "https://api.anthropic.com",
            "POST",
            "/v1/files",
            b"",
            br#"{"id":"file_01","type":"file"}"#
        )),
        [("file_01", "https://api.anthropic.com/v1/files/file_01")]
    );
    let gemini = "https://generativelanguage.googleapis.com";
    assert_eq!(
        ids(&created_resources(
            "gemini",
            gemini,
            "POST",
            "/upload/v1beta/files",
            b"",
            br#"{"file":{"name":"files/abc","mimeType":"text/plain"}}"#
        )),
        [(
            "files/abc",
            "https://generativelanguage.googleapis.com/v1beta/files/abc"
        )]
    );
    assert_eq!(
        ids(&created_resources(
            "gemini",
            gemini,
            "POST",
            "/v1beta/cachedContents",
            b"{}",
            br#"{"name":"cachedContents/xyz","model":"models/gemini-2.5-flash"}"#
        )),
        [(
            "cachedContents/xyz",
            "https://generativelanguage.googleapis.com/v1beta/cachedContents/xyz"
        )]
    );
    assert_eq!(
        ids(&created_resources(
            "gemini",
            gemini,
            "POST",
            "/v1beta/interactions",
            b"{}",
            br#"{"id":"v1_int","status":"completed"}"#
        )),
        [(
            "v1_int",
            "https://generativelanguage.googleapis.com/v1beta/interactions/v1_int"
        )]
    );
    assert!(
        created_resources(
            "gemini",
            gemini,
            "POST",
            "/v1beta/interactions",
            br#"{"store":false}"#,
            br#"{"id":"v1_int"}"#
        )
        .is_empty()
    );
}

#[test]
fn reads_and_other_routes_create_nothing() {
    let reply = br#"{"id":"resp_1"}"#;
    assert!(created_resources("openai", "o", "GET", "/v1/responses/resp_1", b"", reply).is_empty());
    assert!(
        created_resources(
            "openai",
            "o",
            "POST",
            "/v1/chat/completions",
            b"{}",
            br#"{"id":"chatcmpl-1"}"#
        )
        .is_empty()
    );
}

#[tokio::test]
async fn cleanup_deletes_what_remains_and_treats_404_as_gone() {
    let provider = httpmock::MockServer::start_async().await;
    let deleted = provider
        .mock_async(|when, then| {
            when.method("DELETE")
                .path("/v1/responses/resp_live")
                .header("authorization", "Bearer test");
            then.status(200);
        })
        .await;
    let gone = provider
        .mock_async(|when, then| {
            when.method("DELETE").path("/v1/responses/resp_gone");
            then.status(404);
        })
        .await;
    // Gemini's answer for a cache that is already deleted or expired.
    let cache_gone = provider
        .mock_async(|when, then| {
            when.method("DELETE").path("/v1beta/cachedContents/c1");
            then.status(403).body(
                r#"{"error":{"code":403,"message":"CachedContent not found (or permission denied)","status":"PERMISSION_DENIED"}}"#,
            );
        })
        .await;
    let refused = provider
        .mock_async(|when, then| {
            when.method("DELETE").path("/v1beta/cachedContents/c2");
            then.status(403).body(
                r#"{"error":{"code":403,"message":"Permission denied","status":"PERMISSION_DENIED"}}"#,
            );
        })
        .await;
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let path = dir.path().join(LEDGER_FILE);
    let created = |id: &str| {
        LedgerEntry::Created(CreatedResource {
            provider: "openai".into(),
            scenario: "cleanup/test".into(),
            kind: ResourceKind::Response,
            id: id.into(),
            delete_url: format!("{}/v1/responses/{id}", provider.base_url()),
        })
    };
    append(
        &path,
        &[
            created("resp_live"),
            created("resp_gone"),
            created("resp_live"),
        ],
    );
    let other = LedgerEntry::Created(CreatedResource {
        provider: "unknown".into(),
        scenario: "cleanup/test".into(),
        kind: ResourceKind::File,
        id: "file-x".into(),
        delete_url: format!("{}/v1/files/file-x", provider.base_url()),
    });
    append(&path, &[other]);
    let cache = |id: &str| {
        LedgerEntry::Created(CreatedResource {
            provider: "gemini".into(),
            scenario: "cleanup/test".into(),
            kind: ResourceKind::CachedContent,
            id: format!("cachedContents/{id}"),
            delete_url: format!("{}/v1beta/cachedContents/{id}", provider.base_url()),
        })
    };
    append(&path, &[cache("c1"), cache("c2")]);
    // Only Gemini's 403 "not found" means gone.
    let openai_forbidden = provider
        .mock_async(|when, then| {
            when.method("DELETE").path("/v1/files/file-403");
            then.status(403)
                .body(r#"{"error":{"message":"File not found or not yours"}}"#);
        })
        .await;
    append(
        &path,
        &[LedgerEntry::Created(CreatedResource {
            provider: "openai".into(),
            scenario: "cleanup/test".into(),
            kind: ResourceKind::File,
            id: "file-403".into(),
            delete_url: format!("{}/v1/files/file-403", provider.base_url()),
        })],
    );

    let credential = |provider: &str| {
        matches!(provider, "openai" | "gemini")
            .then(|| vec![("authorization".into(), "Bearer test".into())])
    };
    let report = clean_up(&path, credential).await;
    assert_eq!(
        report,
        CleanupReport {
            deleted: 1,
            gone: 2,
            remaining: 2,
            no_credential: 1,
        }
    );
    deleted.assert_calls_async(1).await;
    gone.assert_calls_async(1).await;
    cache_gone.assert_calls_async(1).await;
    refused.assert_calls_async(1).await;

    // Settled resources are not retried; the refused one and the one without
    // a credential are.
    let outstanding: Vec<_> = outstanding(&path).into_iter().map(|r| r.id).collect();
    assert_eq!(outstanding, ["file-x", "cachedContents/c2", "file-403"]);
    let again = clean_up(&path, credential).await;
    assert_eq!((again.remaining, again.no_credential), (2, 1));
    openai_forbidden.assert_calls_async(2).await;
    deleted.assert_calls_async(1).await;
}

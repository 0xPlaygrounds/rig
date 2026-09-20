use super::*;
use anyhow::ensure;

#[crate::cassette(Scenario::live("example/live").missing("Local endpoint captures into a fresh temporary directory"))]
#[tokio::test]
async fn allowed_recording_uses_local_endpoint_and_replays() -> anyhow::Result<()> {
    let upstream = httpmock::MockServer::start_async().await;
    let mock = upstream
        .mock_async(|when, then| {
            when.path("/answer");
            then.status(200)
                .json_body(serde_json::json!({"answer": "local only"}));
        })
        .await;
    let temp = assert_fs::TempDir::new()?;
    let root = temp.path().join("captures");
    let cassette = start_mode(
        Transport::Proxy,
        &root,
        "example",
        CassetteSpec::new("live"),
        &upstream.base_url(),
        CassetteMode::Record,
    )
    .await?;
    let client = reqwest::Client::builder().no_proxy().build()?;
    let response = client
        .get(format!("{}/answer", cassette.base_url()))
        .send()
        .await?;
    ensure!(response.status() == 200);
    ensure!(cassette.checkpoint_attempt(&root).await.is_err());
    ensure!(
        !root.exists(),
        "refused checkpoints must not create the corpus"
    );
    let attempts = temp.path().join("attempts");
    let checkpoint = cassette
        .checkpoint_attempt(&attempts)
        .await?
        .context("completed local exchange")?;
    ensure!(checkpoint == attempts.join("example/live.yaml"));
    ensure!(checkpoint.is_file());
    ensure!(!root.exists());
    ensure!(
        cassette
            .checkpoint_attempt(&attempts.join("../captures"))
            .await
            .is_err()
    );
    std::fs::create_dir_all(root.join("example"))?;
    let protected = root.join("example/derived.yaml");
    std::fs::write(&protected, "protected fixture bytes")?;
    #[cfg(unix)]
    {
        let alias = temp.path().join("corpus-alias");
        std::os::unix::fs::symlink(&root, &alias)?;
        ensure!(cassette.checkpoint_attempt(&alias).await.is_err());
        let linked = temp.path().join("linked-attempts");
        std::fs::create_dir_all(linked.join("example"))?;
        std::os::unix::fs::symlink(&protected, linked.join("example/live.yaml"))?;
        ensure!(cassette.checkpoint_attempt(&linked).await.is_err());
    }
    ensure!(std::fs::read_to_string(&protected)? == "protected fixture bytes");
    cassette.finish().await;
    mock.assert_calls_async(1).await;
    ensure!(root.join("example/live.yaml").is_file());
    let cassette = start_mode(
        Transport::Proxy,
        &root,
        "example",
        CassetteSpec::new("live"),
        &upstream.base_url(),
        CassetteMode::Replay,
    )
    .await?;
    let response = client
        .get(format!("{}/answer", cassette.base_url()))
        .send()
        .await?;
    ensure!(response.status() == 200);
    ensure!(cassette.checkpoint_attempt(&root).await?.is_none());
    cassette.finish().await;
    mock.assert_calls_async(1).await;
    Ok(())
}

#[crate::cassette(
        Scenario::derived("example/derived", &["example/live"], "Deliberately corrupt bytes", "Rewrite the local source response"),
        Scenario::synthetic("example/synthetic", "Hand-written fault"),
    )]
#[tokio::test]
async fn denial_precedes_upstream_and_filesystem_mutation() -> anyhow::Result<()> {
    let upstream = httpmock::MockServer::start_async().await;
    let mock = upstream
        .mock_async(|when, then| {
            when.any_request();
            then.status(200);
        })
        .await;
    let temp = assert_fs::TempDir::new()?;
    let root = temp.path().join("must-not-be-created");
    for scenario in ["derived", "synthetic", "unknown"] {
        for transport in [Transport::Proxy, Transport::Direct] {
            let result = start_mode(
                transport,
                &root,
                "example",
                CassetteSpec::new(scenario),
                &upstream.base_url(),
                CassetteMode::Record,
            )
            .await;
            ensure!(result.is_err(), "{scenario} must not start recording");
            ensure!(
                !root.exists(),
                "denial must not create a directory or fixture"
            );
            mock.assert_calls_async(0).await;
        }
    }
    // Invalid upstream syntax is deliberately untouched: policy runs first.
    let result = start_mode(
        Transport::Proxy,
        &root,
        "example",
        CassetteSpec::new("derived"),
        "not a URL",
        CassetteMode::Record,
    )
    .await;
    ensure!(result.is_err());
    Ok(())
}

#[cfg(unix)]
#[tokio::test]
async fn recording_refuses_a_fixture_symlink_before_upstream_contact() -> anyhow::Result<()> {
    let upstream = httpmock::MockServer::start_async().await;
    let mock = upstream
        .mock_async(|when, then| {
            when.any_request();
            then.status(200);
        })
        .await;
    let temp = assert_fs::TempDir::new()?;
    let root = temp.path().join("captures");
    std::fs::create_dir_all(root.join("example"))?;
    let protected = root.join("example/derived.yaml");
    std::fs::write(&protected, "protected bytes")?;
    std::os::unix::fs::symlink(&protected, root.join("example/live.yaml"))?;
    let result = start_mode(
        Transport::Proxy,
        &root,
        "example",
        CassetteSpec::new("live"),
        &upstream.base_url(),
        CassetteMode::Record,
    )
    .await;
    ensure!(result.is_err());
    mock.assert_calls_async(0).await;
    ensure!(std::fs::read_to_string(&protected)? == "protected bytes");
    Ok(())
}

#[test]
fn scripted_family_allows_only_its_declared_sources() {
    let family = ScriptedFamily::new("openai", &["streaming/streaming_smoke"]);
    assert!(
        !family
            .recorded_sse_frames("streaming/streaming_smoke", 0)
            .is_empty()
    );
    assert!(
        std::panic::catch_unwind(
            || family.recorded_sse_frames("streaming_tools/streaming_tools_smoke", 0)
        )
        .is_err()
    );
    assert!(std::panic::catch_unwind(|| family.status_reply("undeclared", 503, false)).is_err());
    assert!(
        std::panic::catch_unwind(|| family.recorded_statuses_and_bodies("undeclared")).is_err()
    );
}

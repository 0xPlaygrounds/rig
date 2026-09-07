//! Explicit-root regression tests using independently owned fixture layouts.

use super::*;

#[test]
fn independent_downstream_graph_excludes_smithy_and_agent_runtimes() {
    let scratch = assert_fs::TempDir::new().expect("temporary downstream package");
    fs::create_dir(scratch.path().join("src")).expect("create source directory");
    fs::write(
        scratch.path().join("src/lib.rs"),
        "pub use rig_cassette::CassetteSpec;\n",
    )
    .expect("write downstream source");
    let path = serde_json::to_string(env!("CARGO_MANIFEST_DIR")).expect("manifest path string");
    fs::write(scratch.path().join("Cargo.toml"), format!(
        "[package]\nname = \"cassette-downstream-probe\"\nversion = \"0.0.0\"\nedition = \"2024\"\n[workspace]\n[dependencies]\nrig-cassette = {{ path = {path}, default-features = false }}\n"
    )).expect("write downstream manifest");
    // Reuse the repository's resolved versions, including any subsequently
    // yanked release already locked by the build. The separate manifest still
    // computes its own no-default-feature graph; it must not resolve a new
    // dependency universe from whichever sparse-index entries CI has cached.
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.lock"),
        scratch.path().join("Cargo.lock"),
    )
    .expect("seed downstream dependency versions");
    let output =
        std::process::Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
            .current_dir(scratch.path())
            .args(["tree", "--offline", "-e", "normal", "--prefix", "none"])
            .output()
            .expect("inspect downstream graph");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let graph = String::from_utf8(output.stdout).expect("dependency graph UTF-8");
    let names: Vec<_> = graph
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .collect();
    assert!(names.contains(&"rig-cassette"));
    assert!(names.contains(&"rig-core"));
    for forbidden in [
        "rig",
        "rig-agent",
        "rig-ecs",
        "aws-smithy-eventstream",
        "aws-smithy-types",
    ] {
        assert!(
            !names.contains(&forbidden),
            "unexpected dependency {forbidden}: {graph}"
        );
    }
}

fn write_fixture(root: &Path, scenario: &str, response: &str) {
    let path = cassette_path(root, "example", scenario);
    fs::create_dir_all(path.parent().expect("fixture parent")).expect("create fixture directory");
    let interactions = [CassetteInteraction {
        when: CassetteRequest {
            path: "/v1/answer".into(),
            method: "POST".into(),
            query_param: Vec::new(),
            header: vec![NameValue {
                name: "Content-Type".into(),
                value: "application/json".into(),
            }],
            body: Some(r#"{"input":"hello"}"#.into()),
            body_encoding: BodyEncoding::Utf8,
        },
        then: CassetteResponse {
            status: 200,
            header: Vec::new(),
            body: Some(response.into()),
            body_encoding: BodyEncoding::Utf8,
        },
    }];
    fs::write(path, serialize_cassette_interactions(&interactions)).expect("write fixture");
}

#[test]
fn every_reader_uses_the_supplied_repository_or_downstream_root() {
    let scratch = assert_fs::TempDir::new().expect("temporary fixtures");
    for layout in ["rig/tests/cassettes", "downstream/fixtures/cassettes"] {
        let root = scratch.path().join(layout);
        write_fixture(&root, "nested/unary", r#"{"answer":"ok"}"#);
        write_fixture(
            &root,
            "nested/stream",
            "data: {\"answer\":\"ok\"}\n\ndata: [DONE]\n\n",
        );
        assert_eq!(
            cassette_path(&root, "example", "nested/unary"),
            root.join("example/nested/unary.yaml")
        );
        assert_eq!(
            recorded_interaction_bodies(&root, "example", "nested/unary"),
            vec![(r#"{"input":"hello"}"#.into(), r#"{"answer":"ok"}"#.into())]
        );
        assert_eq!(
            recorded_json_request(&root, "example", "nested/unary"),
            json!({"input":"hello"})
        );
        assert_eq!(
            recorded_json_response(&root, "example", "nested/unary"),
            json!({"answer":"ok"})
        );
        assert_eq!(
            recorded_request_header_pairs(&root, "example", "nested/unary"),
            vec![vec![("content-type".into(), "application/json".into())]]
        );
        assert_eq!(
            recorded_request_paths(&root, "example", "nested/unary"),
            vec!["/v1/answer"]
        );
        assert_eq!(
            recorded_statuses_and_bodies(&root, "example", "nested/unary"),
            vec![(200, r#"{"answer":"ok"}"#.into())]
        );
        assert_eq!(
            recorded_sse_json_frames(&root, "example", "nested/stream"),
            vec![json!({"answer":"ok"})]
        );
    }
}

#[tokio::test]
async fn both_constructors_replay_from_an_external_fixture_root() {
    // The test runner invokes this suite in replay mode; no environment mutation
    // is needed, so it remains safe alongside other parallel cassette tests.
    assert_eq!(CassetteMode::current(), CassetteMode::Replay);
    let scratch = assert_fs::TempDir::new().expect("temporary fixtures");
    let root = scratch.path().join("downstream/fixtures/cassettes");
    write_fixture(&root, "unary", r#"{"answer":"ok"}"#);
    for direct in [false, true] {
        let cassette = if direct {
            ProviderCassette::start_direct_recording(
                &root,
                "example",
                "unary",
                "https://example.invalid/v1",
            )
            .await
        } else {
            ProviderCassette::start(&root, "example", "unary", "https://example.invalid/v1").await
        };
        let response = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("HTTP client")
            .post(format!("{}/answer", cassette.base_url()))
            .json(&json!({"input":"hello"}))
            .send()
            .await
            .expect("local replay response");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.json::<Value>().await.expect("response JSON"),
            json!({"answer":"ok"})
        );
        cassette.finish().await;
    }
}

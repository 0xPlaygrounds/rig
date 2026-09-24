use super::*;

fn owner_of(source: &str, scenario: &str) -> Option<Owner> {
    owners_in_source(source, scenario, "llamacpp::cassette::tool_matrix")
        .expect("valid Rust")
        .into_iter()
        .next()
}

#[test]
fn the_enclosing_test_owns_a_wrapper_call_not_a_nested_tool_fn() {
    // A resolver that takes the nearest enclosing fn picks the tool's `call` here.
    let source = r#"
        #[tokio::test]
        async fn a_one_argument_tool_round_trips_its_value() {
            struct Lookup;
            impl Tool for Lookup {
                async fn call(&self, args: Args) -> Result<String, Error> { Ok(args.city) }
            }
            with_llamacpp_competent_cassette("tool_matrix/one_argument_tool", move |client| async move {
                let _ = client;
            })
            .await;
        }
    "#;
    assert_eq!(
        owner_of(source, "tool_matrix/one_argument_tool"),
        Some(Owner::Test(
            "llamacpp::cassette::tool_matrix::a_one_argument_tool_round_trips_its_value".into()
        ))
    );
}

#[test]
fn a_helper_fn_is_never_the_owner() {
    let source = r#"
        async fn shared_body(scenario: &str) {
            with_openai_cassette("helpers/untested", |c| async {}).await;
        }
        #[tokio::test]
        async fn outer() {
            async fn inner() {
                with_openai_cassette("helpers/nested", |c| async {}).await;
            }
            inner().await;
        }
    "#;
    assert_eq!(owner_of(source, "helpers/untested"), None);
    assert_eq!(
        owner_of(source, "helpers/nested"),
        Some(Owner::Test("llamacpp::cassette::tool_matrix::outer".into()))
    );
}

#[test]
fn a_spec_chain_and_a_matrix_row_name_their_owner() {
    let source = r#"
        #[tokio::test]
        async fn batch() {
            with_openai_cassette(CassetteSpec::new("multi_extract/batch").unordered(), |c| async {}).await;
        }
        crate::matrix::case_matrix! {
            wrapper: with_openai_cassette, family: agent_tool_sessions_case;
            #[tokio::test]
            parallel_calls: ("agent_tool_sessions/parallel_calls", parallel_calls_3);
        }
    "#;
    assert_eq!(
        owner_of(source, "multi_extract/batch"),
        Some(Owner::Test("llamacpp::cassette::tool_matrix::batch".into()))
    );
    assert_eq!(
        owner_of(source, "agent_tool_sessions/parallel_calls"),
        Some(Owner::Test(
            "llamacpp::cassette::tool_matrix::parallel_calls".into()
        ))
    );
}

#[test]
fn a_replay_only_reader_is_not_an_owner_and_a_skipping_test_is_hand_derived() {
    let source = r#"
        #[test]
        fn the_smoke_tier_round_trip_is_covered_elsewhere() {
            let calls = recorded_statuses_and_bodies("llamacpp", "tools/tools_roundtrip");
        }
        #[tokio::test]
        async fn object_top_p() {
            if crate::cassettes::skip_when_recording("hand-derived") { return; }
            with_openai_cassette("response_metadata_matrix/object_top_p", |c| async {}).await;
        }
    "#;
    assert_eq!(owner_of(source, "tools/tools_roundtrip"), None);
    assert_eq!(
        owner_of(source, "response_metadata_matrix/object_top_p"),
        Some(Owner::HandDerived(
            "llamacpp::cassette::tool_matrix::object_top_p".into()
        ))
    );
}

#[test]
fn inline_modules_join_the_path() {
    let source = r#"
        mod nested {
            #[tokio::test]
            async fn cell() { with_x_cassette("a/b", |c| async {}).await; }
        }
    "#;
    assert_eq!(
        owner_of(source, "a/b"),
        Some(Owner::Test(
            "llamacpp::cassette::tool_matrix::nested::cell".into()
        ))
    );
}

#[test]
fn module_paths_follow_the_file_layout() {
    let base = Path::new("/r/crates/rig-cassette/tests/providers");
    assert_eq!(
        module_path(base, &base.join("openai/cassette/raw_capture_matrix.rs")),
        "openai::cassette::raw_capture_matrix"
    );
    assert_eq!(module_path(base, &base.join("groq/mod.rs")), "groq");
}

#[test]
fn path_attributes_decide_the_module_path() {
    let dir = std::env::temp_dir().join(format!("xtask-modules-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("cassette")).expect("dir");
    std::fs::write(
        dir.join("mod.rs"),
        "#[path = \"cassette/corpus_faults.rs\"]\nmod corpus_faults;\nmod cassette;\n",
    )
    .expect("mod.rs");
    std::fs::write(dir.join("cassette/corpus_faults.rs"), "").expect("faults");
    std::fs::write(dir.join("cassette.rs"), "mod streaming;\n").expect("cassette.rs");
    std::fs::write(dir.join("cassette/streaming.rs"), "").expect("streaming");
    let map = module_map(&dir.join("mod.rs"), "deepseek");
    assert_eq!(
        map.get(&dir.join("cassette/corpus_faults.rs"))
            .map(String::as_str),
        Some("deepseek::corpus_faults")
    );
    assert_eq!(
        map.get(&dir.join("cassette/streaming.rs"))
            .map(String::as_str),
        Some("deepseek::cassette::streaming")
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_shared_fixture_lists_every_producer() {
    let source = r#"
        #[tokio::test]
        async fn first() {
            with_openai_cassette("agent/completion_smoke", |c| async {}).await;
        }
        #[tokio::test]
        async fn second() {
            with_openai_cassette("agent/completion_smoke", |c| async {}).await;
        }
    "#;
    assert_eq!(
        owners_in_source(source, "agent/completion_smoke", "openai").expect("valid Rust"),
        [
            Owner::Test("openai::first".into()),
            Owner::Test("openai::second".into())
        ]
    );
}

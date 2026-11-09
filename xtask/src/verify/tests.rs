use super::*;

use std::collections::{BTreeMap, BTreeSet};

#[test]
fn check_ids_are_unique() {
    let all = checks::all();
    assert_eq!(
        all.len(),
        all.iter().map(|c| &c.id).collect::<BTreeSet<_>>().len()
    );
}

#[test]
fn telemetry_retry_policy_is_not_overridden() {
    let all = checks::all();
    let c = all.iter().find(|c| c.id == "core-all").unwrap();
    assert!(c.steps[0].args.contains(&"guards".into()));
    assert!(!c.steps[0].args.contains(&"--retries".into()));
}

#[test]
fn verification_commands_force_replay_shape_matching_and_snapshot_checks_and_preserve_retry_contracts()
 {
    let step = Step::new("cargo", &["test"])
        .env("RIG_PROVIDER_TEST_MODE", "record")
        .env("RIG_CASSETTE_MATCHING", "exact")
        .env("RIG_REGENERATE_GOLDEN", "1")
        .env("RIG_CASSETTE_SNAPSHOTS", "write")
        .env("NEXTEST_RETRIES", "5");
    let command = execute::command(Path::new("/repo"), &step, Path::new("/repo/target"));
    let env: BTreeMap<_, _> = command.get_envs().collect();
    assert_eq!(
        env[std::ffi::OsStr::new("RIG_PROVIDER_TEST_MODE")],
        Some(std::ffi::OsStr::new("replay"))
    );
    assert_eq!(env[std::ffi::OsStr::new("RIG_REGENERATE_GOLDEN")], None);
    assert_eq!(
        env[std::ffi::OsStr::new("RIG_CASSETTE_SNAPSHOTS")],
        Some(std::ffi::OsStr::new("check"))
    );
    assert_eq!(
        env[std::ffi::OsStr::new("RIG_CASSETTE_MATCHING")],
        Some(std::ffi::OsStr::new("shape"))
    );
    assert_eq!(env[std::ffi::OsStr::new("NEXTEST_RETRIES")], None);
}

#[test]
fn conformance_compiles_exactly_its_executed_targets() {
    let all = checks::all();
    let c = all.iter().find(|c| c.id == "conformance").unwrap();
    let targets: BTreeSet<_> = c.steps[0]
        .args
        .windows(2)
        .filter(|w| w[0] == "--test")
        .map(|w| w[1].as_str())
        .collect();
    assert_eq!(
        targets,
        BTreeSet::from(["integration", "history_conformance"])
    );
    assert!(c.steps[0].args.windows(2).any(|w| w == ["--retries", "0"]));
    assert!(c.steps[0].args.contains(&"--all-features".into()));
}

#[test]
fn full_covers_workspace_and_example_targets() {
    // `tests` owns the locked-version compile of every member's lib, bin,
    // test and example targets: Cargo's default `test` target selection,
    // which any narrowing flag would silently drop.
    let all = checks::all();
    let c = all.iter().find(|c| c.id == "tests").unwrap();
    for flag in ["--workspace", "--all-features"] {
        assert!(c.steps[0].args.contains(&flag.into()));
    }
    for flag in [
        "--lib",
        "--bins",
        "--bin",
        "--tests",
        "--test",
        "--examples",
        "--example",
        "--benches",
    ] {
        assert!(!c.steps[0].args.contains(&flag.into()), "{flag}");
    }
    assert!(!all.iter().any(|c| c.id == "workspace-check"));
    // Bench targets are outside cargo test's default selection. Each needs
    // an explicit locked compilation owner that does not execute it.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let metadata: Value = serde_json::from_str(
        &output(
            root,
            "cargo",
            &["metadata", "--locked", "--no-deps", "--format-version", "1"],
        )
        .expect("cargo metadata --no-deps for the bench guard"),
    )
    .unwrap();
    let packages = metadata["packages"].as_array().unwrap();
    assert!(packages.len() > 20, "workspace members not found");
    for package in packages {
        for bench in package["targets"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|target| {
                target["kind"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|kind| kind == "bench")
            })
        {
            let package_name = package["name"].as_str().unwrap();
            let bench_name = bench["name"].as_str().unwrap();
            let owned = all.iter().flat_map(|check| &check.steps).any(|step| {
                step.program == "cargo"
                    && step.args.first().is_some_and(|arg| arg == "check")
                    && step.args.iter().any(|arg| arg == "--locked")
                    && step
                        .args
                        .windows(2)
                        .any(|pair| pair == ["-p", package_name])
                    && step
                        .args
                        .windows(2)
                        .any(|pair| pair == ["--bench", bench_name])
            });
            assert!(
                owned,
                "{package_name} bench {bench_name} has no locked compilation owner"
            );
        }
    }
}

struct Repo(std::path::PathBuf);
impl Repo {
    fn new() -> Self {
        static SEQUENCE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "rig-planner-repo-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        output(&path, "git", &["init", "--quiet"]).unwrap();
        let hooks = path.join(".git/empty-hooks");
        std::fs::create_dir(&hooks).unwrap();
        output(&path, "git", &["config", "commit.gpgsign", "false"]).unwrap();
        output(
            &path,
            "git",
            &["config", "core.hooksPath", hooks.to_str().unwrap()],
        )
        .unwrap();
        std::fs::write(path.join(".gitignore"), "target/\n").unwrap();
        std::fs::write(path.join("file.rs"), "original").unwrap();
        output(&path, "git", &["add", "."]).unwrap();
        output(
            &path,
            "git",
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
        )
        .unwrap();
        Self(path)
    }
}

#[test]
fn staged_then_undone_and_untracked_inputs_are_not_lost() {
    let repo = Repo::new();
    std::fs::write(repo.0.join("file.rs"), "staged").unwrap();
    output(&repo.0, "git", &["add", "file.rs"]).unwrap();
    std::fs::write(repo.0.join("file.rs"), "original").unwrap();
    std::fs::write(repo.0.join("new.rs"), "new").unwrap();
    std::fs::create_dir_all(repo.0.join("target")).unwrap();
    std::fs::write(repo.0.join("target/out"), "ignored").unwrap();
    let paths =
        selection::changes(&repo.0, &Options::parse(vec!["--full".into()]).unwrap()).unwrap();
    assert!(paths.contains("file.rs"));
    assert!(paths.contains("new.rs"));
    assert!(!paths.iter().any(|p| p.starts_with("target")));
}

fn fake_check(id: &str, script: &str) -> Check {
    Check {
        id: id.into(),
        reason: "fake execution".into(),
        steps: vec![Step::new("bash", &["-c", script])],
    }
}

#[test]
fn failed_required_command_stops_later_checks() {
    let repo = Repo::new();
    let metadata = serde_json::json!({"target_directory": repo.0.join("target")});
    let later = fake_check("later", "touch target/later");
    std::fs::create_dir_all(repo.0.join("target")).unwrap();
    let error = execute::run(
        &repo.0,
        &metadata,
        &[fake_check("first", "exit 3"), later.clone()],
        false,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("required check first failed"),
        "{error}"
    );
    assert!(!repo.0.join("target/later").exists());
    execute::run(&repo.0, &metadata, &[later], false).unwrap();
    assert!(repo.0.join("target/later").exists());
}

#[test]
fn missing_prerequisite_stops_before_any_check() {
    let repo = Repo::new();
    let metadata = serde_json::json!({"target_directory": repo.0.join("target")});
    let first = fake_check("first", "touch target/executed");
    let missing = Check {
        id: "missing".into(),
        reason: "test".into(),
        steps: vec![Step::new("rig-nonexistent-tool-for-test", &[])],
    };
    let error = execute::run(&repo.0, &metadata, &[first, missing], false).unwrap_err();
    assert!(error.to_string().contains("no checks executed"));
    assert!(error.to_string().contains("rig-nonexistent-tool-for-test"));
    assert!(!repo.0.join("target/executed").exists());
}

impl Drop for Repo {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn tool_versions_require_the_exact_pinned_version() {
    assert!(preflight::version_matches(
        "rustc 1.95.0 (abc 2026-04-14)",
        "1.95.0"
    ));
    assert!(!preflight::version_matches(
        "rustc 1.95.0 (abc 2026-04-14)",
        "1.95"
    ));
}

#[test]
fn workflows_carry_no_toolchain_copy() {
    // rust-toolchain.toml is the single toolchain source. A literal
    // `toolchain:` or a RUST_VERSION copy in any workflow or action would
    // build one job on a stale compiler; jobs that run no check (release-plz,
    // the plan job) never reach the planner's rustc probe, so scan the text.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut files: Vec<_> = std::fs::read_dir(root.join(".github/workflows"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .collect();
    files.push(root.join(".github/actions/rust-setup/action.yml"));
    for file in files {
        let text = std::fs::read_to_string(&file).unwrap();
        for line in text.lines() {
            let trimmed = line.trim();
            assert!(
                !trimmed.starts_with("RUST_VERSION:"),
                "{}: RUST_VERSION copy",
                file.display()
            );
            if let Some(value) = trimmed.strip_prefix("toolchain:") {
                assert!(
                    value.contains("${{"),
                    "{}: literal toolchain pin {value:?}",
                    file.display()
                );
            }
        }
    }
}

#[test]
fn default_check_compiles_extracted_regressions_without_extra_features() {
    let all = checks::all();
    let check = all
        .iter()
        .find(|check| check.id == "default-check")
        .unwrap();
    let args = &check.steps[0].args;
    for package in ["rig", "rig-test-support"] {
        assert!(args.windows(2).any(|pair| pair == ["-p", package]));
    }
    assert!(args.iter().any(|arg| arg == "--tests"));
    for flag in ["--features", "--all-features", "--no-default-features"] {
        assert!(!args.iter().any(|arg| arg == flag), "{flag}");
    }
}

/// The findings of the typed-options guards in `source`, as a listed
/// completion-wire file.
fn guarded(source: &str) -> Vec<String> {
    super::guards::offenders("crates/rig-core/src/providers/openai/wire/chat.rs", source)
        .unwrap_or_else(|error| panic!("the source parses: {error}"))
}

#[test]
fn an_option_fields_pattern_must_name_every_field() {
    for source in [
        "use crate::completion::options::OptionFields;
         fn map(fields: OptionFields<'_>) { let OptionFields { reasoning, .. } = fields; }",
        "use crate::completion::options::{OptionFields as F};
         fn map(fields: F<'_>) { let F { reasoning, .. } = fields; }",
        "use crate::completion::options::OptionFields;
         fn map(fields: OptionFields<'_>) { let OptionFields { stop: _stop, reasoning } = fields; }",
        "use crate::completion::options::OptionFields;
         fn map(fields: OptionFields<'_>) { let OptionFields { stop: _, reasoning } = fields; }",
    ] {
        let findings = guarded(source);
        assert_eq!(findings.len(), 1, "{source}: {findings:?}");
        assert!(findings[0].starts_with("options-mapping: "), "{findings:?}");
    }
    assert!(
        guarded(
            "use crate::completion::options::OptionFields;
             fn map(fields: OptionFields<'_>) { let OptionFields { reasoning, stop } = fields; }"
        )
        .is_empty()
    );
}

#[test]
fn an_option_map_literal_takes_no_base() {
    let findings = guarded(
        "use crate::completion::options::OptionMap as Answers;
         fn map(base: Answers) -> Answers { Answers { seed: Mapping::Nothing, ..base } }",
    );
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(findings[0].contains("'..'"), "{findings:?}");
}

#[test]
fn a_wire_neither_builds_nor_rewrites_a_requests_options() {
    for source in [
        "fn encode(request: CompletionRequest) { let copy = CompletionRequest::new(\"hi\"); }",
        "fn encode(request: CompletionRequest) { let copy = request.options(GenerationOptions::default()); }",
        "fn encode(mut request: CompletionRequest) { request.options = GenerationOptions::default(); }",
        "fn encode(request: &CompletionRequest) -> bool { request.options.seed.is_some() }",
    ] {
        let findings = guarded(source);
        assert!(
            findings
                .iter()
                .all(|finding| finding.starts_with("options-mapping: ")),
            "{source}: {findings:?}"
        );
        assert!(!findings.is_empty(), "{source}");
    }
}

#[test]
fn a_mapping_cannot_defer_to_a_raw_key() {
    let findings = guarded(
        "use crate::completion::options::{self, OptionFields};
         fn map(request: &CompletionRequest, fields: OptionFields<'_>) {
             let OptionFields { reasoning, stop } = fields;
             let raw = options::param(self, request, \"thinking\");
         }",
    );
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(findings[0].contains("options::param"), "{findings:?}");
    assert!(
        guarded(
            "use crate::completion::options;
             fn continues_stored(request: &CompletionRequest) -> bool {
                 options::param(self, request, \"store\").is_some()
             }"
        )
        .is_empty()
    );
}

#[test]
fn a_wire_reads_no_raw_params_and_builds_no_body_of_its_own() {
    for source in [
        "fn encode(request: CompletionRequest) { let raw = request.additional_params.clone(); }",
        "use crate::wire::Body;
         fn encode(body: Value) -> Body { Body::Bytes(serde_json::to_vec(&body).unwrap_or_default()) }",
        "use crate::wire::{Body as B};
         fn encode(form: Form) -> B { B::Multipart(form) }",
        "fn encode(builder: Builder, body: Vec<u8>) { builder.body(body); }",
        "fn encode(body: FinalBody) { let copy = body.deserialize::<serde_json::Value>(); }",
        "fn encode(request: &CompletionRequest) { let bytes = serde_json::to_vec(request); }",
    ] {
        let findings = guarded(source);
        assert!(
            findings.iter().any(|finding| finding.starts_with("options-precedence: ")),
            "{source}: {findings:?}"
        );
    }
    for source in [
        "fn encode(builder: Builder, body: FinalBody) { builder.body(body.into_body()); }",
        "fn encode(builder: Builder) { builder.body(crate::wire::Body::empty()); }",
    ] {
        assert!(guarded(source).is_empty(), "{source}");
    }
    // A document part's own `additional_params` is another field, allowed
    // where a wire reads it.
    let document =
        "fn part(document: &Document) { let params = document.additional_params.as_ref(); }";
    let anthropic = super::guards::offenders(
        "crates/rig-core/src/providers/anthropic/completion.rs",
        document,
    );
    assert_eq!(anthropic, Ok(Vec::new()));
    assert_eq!(guarded(document).len(), 1);
}

/// Whether `source`, as a listed completion-wire file, fails the precedence
/// guard.
fn fails_precedence(source: &str) -> bool {
    guarded(source)
        .iter()
        .any(|finding| finding.starts_with("options-precedence: "))
}

#[test]
fn a_wire_opens_no_body_in_a_pattern() {
    for source in [
        "fn rewrite(body: &mut crate::wire::Body) {
             if let crate::wire::Body::Bytes(bytes) = body { bytes.clear(); }
         }",
        "use crate::wire::{Body as B};
         fn rewrite(body: B) -> usize { match body { B::Multipart(_) => 0, _ => 1 } }",
        "fn rewrite(body: &crate::wire::Body) -> bool { matches!(body, crate::wire::Body::Bytes(_)) }",
        "use crate::wire::Body;
         type Sent = Body;
         fn rewrite(body: Sent) -> bool { if let Sent::Bytes(_) = body { true } else { false } }",
    ] {
        assert!(fails_precedence(source), "{source}");
    }
    assert!(!fails_precedence(
        "use crate::wire::Body;
         fn send(body: FinalBody) -> Body { body.into_body() }"
    ));
}

#[test]
fn a_wire_writes_to_no_built_body() {
    assert!(fails_precedence(
        "fn rewrite(mut built: http::Request<Body>) { let body = built.body_mut(); }"
    ));
}

#[test]
fn a_wire_destructures_no_raw_params() {
    for source in [
        "fn encode(request: CompletionRequest) { let CompletionRequest { additional_params, .. } = request; }",
        "fn encode(request: CompletionRequest) { let CompletionRequest { additional_params: raw, .. } = request; }",
        "fn encode(requests: Vec<CompletionRequest>) { requests.into_iter().map(|CompletionRequest { additional_params, .. }| additional_params); }",
    ] {
        assert!(fails_precedence(source), "{source}");
    }
    // The allowlist holds for a pattern as for a field read: a document
    // part's own `additional_params` is another field.
    let document =
        "fn part(document: Document) { let Document { additional_params, .. } = document; }";
    let anthropic = super::guards::offenders(
        "crates/rig-core/src/providers/anthropic/completion.rs",
        document,
    );
    assert_eq!(anthropic, Ok(Vec::new()));
    assert!(fails_precedence(document));
}

#[test]
fn a_glob_import_of_body_resolves() {
    for source in [
        "use crate::wire::Body::*;
         fn rewrite(body: &mut crate::wire::Body) { if let Bytes(bytes) = body { bytes.clear(); } }",
        "use crate::wire::Body::*;
         fn encode(bytes: Vec<u8>) -> crate::wire::Body { Bytes(bytes) }",
        "use crate::wire::{Body as B};
         use B::*;
         fn encode(form: Form) -> B { Multipart(form) }",
    ] {
        assert!(fails_precedence(source), "{source}");
    }
    // A glob of another module does not make every `Bytes` a body.
    assert!(!fails_precedence(
        "use crate::types::*;
         fn size(bytes: Bytes) -> usize { bytes.len() }"
    ));
}

/// `Self` in an `impl` of `Body`, a qualified `<Body>::X` and a macro that
/// splices a variant after `Body` all reach `Body::Bytes` without naming it.
#[test]
fn a_body_reached_without_its_variant_path_fails() {
    for source in [
        "impl crate::wire::Body {
             pub(crate) fn raw(bytes: Vec<u8>) -> Self { Self::Bytes(bytes) }
         }",
        "use crate::wire::Body;
         impl From<Vec<u8>> for Body { fn from(bytes: Vec<u8>) -> Self { Self::Bytes(bytes) } }",
        "fn encode(bytes: Vec<u8>) -> crate::wire::Body { <crate::wire::Body>::Bytes(bytes) }",
        "use crate::wire::{Body as B};
         fn encode(form: Form) -> B { <B>::Multipart(form) }",
        "macro_rules! b { ($v:ident, $x:expr) => { crate::wire::Body::$v($x) } }
         fn encode(bytes: Vec<u8>) -> crate::wire::Body { b!(Bytes, bytes) }",
        "macro_rules! b { ($t:path, $x:expr) => { $t($x) } }
         fn encode(bytes: Vec<u8>) { b!(crate::wire::Body, bytes); }",
    ] {
        assert!(fails_precedence(source), "{source}");
    }
    // A macro that names `Body::empty()` sends no body, and another type's
    // `impl` is not `Body`'s.
    for source in [
        "fn send(builder: Builder) { builder.body(crate::wire::Body::empty()); }",
        "macro_rules! on_route { ($c:expr) => { match $c { Self::Chat(w) => w } } }",
        "impl Payload { fn bytes(&self) -> usize { 0 } }",
    ] {
        assert!(!fails_precedence(source), "{source}");
    }
}

#[test]
fn test_items_are_not_guarded() {
    assert!(
        guarded(
            "#[cfg(test)]
             mod tests { fn build() { let raw = request.additional_params.clone(); } }"
        )
        .is_empty()
    );
}

#[test]
fn a_file_with_a_completion_wire_is_found() {
    use super::guards::holds_a_completion_wire;
    let wire = "impl Wire for Chat { type Op = crate::operation::Completion; }";
    assert_eq!(holds_a_completion_wire(wire), Ok(true));
    let target = "impl crate::completion::ReplayTarget for Chat {}";
    assert_eq!(holds_a_completion_wire(target), Ok(true));
    let embeddings = "impl Wire for Embeddings { type Op = crate::operation::Embed; }";
    assert_eq!(holds_a_completion_wire(embeddings), Ok(false));
    let tested = "#[cfg(test)] impl ReplayTarget for Fake {}";
    assert_eq!(holds_a_completion_wire(tested), Ok(false));
}

#[test]
fn a_wire_reads_and_sets_no_provider_options() {
    for source in [
        "fn encode(request: &CompletionRequest) -> bool { request.provider_options.is_empty() }",
        "fn encode(request: CompletionRequest) { let CompletionRequest { provider_options, .. } = request; }",
        "fn encode(request: CompletionRequest) { let copy = request.provider_options(ProviderOptions::new()); }",
        "fn encode(mut request: CompletionRequest) { request.provider_options = ProviderOptions::new(); }",
    ] {
        assert!(fails_precedence(source), "{source}");
    }
}

/// The findings of the `extras-off-decode-path` guard in `source`, as the
/// decode-path file `file`.
fn decoding(file: &str, source: &str) -> Vec<String> {
    super::extras::offenders(file, source, true)
        .unwrap_or_else(|error| panic!("the source parses: {error}"))
}

const CHAT: &str = "crates/rig-core/src/providers/openai/wire/chat.rs";

#[test]
fn a_decoder_names_no_extension_module_in_any_import_form() {
    for source in [
        "use crate::providers::openrouter::{extension as ext};",
        "use crate::providers::openrouter::{self, extension::*};",
        "use super::super::openrouter::extension;",
        "use crate::providers::openrouter::extension::OpenRouterExtras as Extras;",
        "use crate::providers::openrouter as or;
         fn read(extras: or::extension::OpenRouterExtras) {}",
        "use crate::providers::*;
         fn read(extras: openrouter::extension::OpenRouterExtras) {}",
        "type Extras = crate::providers::openrouter::extension::OpenRouterExtras;",
        "fn read(reply: &CompletionResponse) {
             let extras = reply.extras::<crate::providers::openrouter::extension::OpenRouterExt>();
         }",
        "macro_rules! extras { ($i:ident) => { crate::providers::openrouter::extension::$i } }",
        "macro_rules! extras { ($i:ident) => { super::extension::$i } }",
        "pub use super::extension::OpenRouterExt;",
    ] {
        let findings = decoding(CHAT, source);
        assert!(!findings.is_empty(), "{source}");
        assert!(
            findings
                .iter()
                .all(|finding| finding.starts_with("extras-off-decode-path: ")),
            "{findings:?}"
        );
    }
}

#[test]
fn a_decoder_names_no_extension_trait_in_any_import_form() {
    for source in [
        "use crate::completion::provider_options::ProviderExtension as Ext;",
        "use crate::completion::{ReplyExtras, CompletionResponse};",
        "use crate::completion::provider_options::{self as po};
         fn read<P: po::ProviderExtension>(raw: &Value) {}",
        "use crate::completion::*;
         fn read<P: ProviderExtension>(raw: &Value) {}",
        "use crate::completion::*;
         impl ReplyExtras for Usage {}",
    ] {
        assert!(!decoding(CHAT, source).is_empty(), "{source}");
    }
    // An alias is followed to its use.
    let aliased = decoding(
        CHAT,
        "use crate::completion::provider_options::ProviderExtension as Ext;
         fn read<P: Ext>(raw: &Value) {}",
    );
    assert_eq!(aliased.len(), 2, "{aliased:?}");
}

#[test]
fn a_decoder_writes_no_citation_past_the_fold() {
    for source in [
        "use crate::message::Citation;
         fn cite() -> Citation { Citation { span: None, sources: Vec::new() } }",
        "use crate::message::{citation::Span as S};
         fn span() -> S { S { start: 0, end: 4 } }",
        "use crate::message::Text;
         fn cite(text: Text) -> Text { Text::with_citations(text, []) }",
        "use crate::message::Text as T;
         fn span(text: &T) { let _ = T::span(text, 0..4); }",
        "fn spans(texts: &[crate::message::Text]) { texts.iter().map(crate::message::Text::span); }",
        "fn cite(text: Text) -> Text { text.with_citations([]) }",
        "fn span(text: &Text) { let _ = text.span(0..4); }",
        "use crate::completion::message::citation;
         fn cite(text: &mut Text) { citation::attach(text, Vec::new(), Vec::new(), \"p\", 0); }",
    ] {
        let findings = decoding(CHAT, source);
        assert!(!findings.is_empty(), "{source}");
        assert!(
            findings
                .iter()
                .all(|finding| finding.starts_with("extras-off-decode-path: ")),
            "{findings:?}"
        );
    }
    // Handing a citation to the fold, or a tracing span, passes.
    for source in [
        "use crate::wire::{SpanUnit, WireCitation, WireSpan};
         fn cite(out: &mut Out<'_, Completion>) {
             out.cite(0, WireCitation::new(Some(WireSpan::new(0, 4, SpanUnit::Chars)), Vec::new()));
         }",
        "fn trace() { let span = tracing::Span::current(); let _ = span.id(); }",
    ] {
        assert_eq!(decoding(CHAT, source), Vec::<String>::new(), "{source}");
    }
}

#[test]
fn a_companion_crates_extension_is_off_its_decode_path_too() {
    let findings = decoding(
        "crates/rig-bedrock/src/completion.rs",
        "use crate::extension::BedrockExt;",
    );
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(
        findings[0].contains("crates/rig-bedrock/src/completion.rs:1"),
        "{findings:?}"
    );
    use super::extras::in_extension_module;
    for file in [
        "crates/rig-bedrock/src/extension.rs",
        "crates/rig-bedrock/src/extension/options.rs",
        "crates/rig-core/src/providers/openrouter/extension.rs",
        "crates/rig-core/src/providers/openai/extension/responses.rs",
    ] {
        assert!(in_extension_module(file), "{file}");
    }
    assert!(!in_extension_module(
        "crates/rig-core/src/providers/openrouter/mod.rs"
    ));
}

#[test]
fn what_the_decode_path_may_still_write() {
    for source in [
        "pub mod extension;",
        "mod extension {
             use crate::completion::{ProviderExtension, ReplyExtras};
             pub struct OpenRouter;
         }",
        "fn suffix(path: &std::path::Path) -> bool {
             let extension = path.extension();
             extension.is_some()
         }",
        "#[cfg(test)]
         mod tests { use super::extension::OpenRouterExt; }",
        "use crate::completion::{CompletionResponse, ProviderOptions};",
    ] {
        assert_eq!(decoding(CHAT, source), Vec::<String>::new(), "{source}");
    }
}

#[test]
fn an_extension_item_is_re_exported_nowhere() {
    let reexports = |source: &str| {
        super::extras::offenders("src/lib.rs", source, false)
            .unwrap_or_else(|error| panic!("the source parses: {error}"))
    };
    for source in [
        "pub use rig_core::providers::openrouter::extension::OpenRouterExt;",
        "pub use rig_core::providers::openrouter::{extension::*};",
        "pub(crate) use rig_core::providers::openrouter::extension;",
        "pub type OpenRouter = rig_core::providers::openrouter::extension::OpenRouterExt;",
    ] {
        assert_eq!(reexports(source).len(), 1, "{source}");
    }
    for source in [
        "use rig_core::providers::openrouter::extension::OpenRouterExt;",
        "pub use rig_core::completion::{ProviderExtension, ReplyExtras};",
        "pub use rig_core::providers;",
    ] {
        assert!(reexports(source).is_empty(), "{source}");
    }
}

/// The directory names under the workspace's `folder`.
fn member_dirs(folder: &str) -> Vec<String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(folder);
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect()
}

#[test]
fn every_plugin_crate_is_a_harness_package() {
    let names = member_dirs("plugins");
    assert!(names.len() > 3, "found {names:?}");
    for name in &names {
        assert!(checks::HARNESS_PACKAGES.contains(&name.as_str()), "{name}");
    }
}

/// CI selects the corpus by the glob [`checks::CORPUS_PACKAGES`], so every
/// corpus crate must be named by it, and none may be published.
#[test]
fn every_corpus_crate_is_selected_by_the_corpus_glob_and_unpublished() {
    let prefix = checks::CORPUS_PACKAGES.trim_end_matches('*');
    let dirs = member_dirs("corpus");
    assert!(dirs.len() > 5, "found {dirs:?}");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../corpus");
    for dir in &dirs {
        let manifest = std::fs::read_to_string(root.join(dir).join("Cargo.toml"))
            .unwrap_or_else(|error| panic!("{dir}: {error}"));
        let package = manifest.split("\n[").next().unwrap_or_default();
        let lines: Vec<&str> = package.lines().map(str::trim).collect();
        assert!(
            lines.contains(&format!("name = \"{prefix}{dir}\"").as_str()),
            "corpus/{dir} must be the package `{prefix}{dir}`"
        );
        assert!(
            lines.contains(&"publish = false"),
            "corpus/{dir} must be `publish = false`"
        );
    }
}

/// The redaction guard finds the library named however an import is laid
/// out, and nothing that only looks like it.
#[test]
fn the_redact_guard_finds_every_way_to_name_the_library() {
    use super::execute::names_redact;

    let named = [
        "use rig_core::redact;",
        "let shown = rig_core::redact::text(line);",
        "use rig_core::{message, redact};",
        "use rig_core::{\n    message,\n    redact,\n};",
        "use rig_core::{\n    completion::{Message, Usage},\n    redact as secrets,\n};",
        "use rig_harness::rig_core::{\n    effect::Answered,\n    redact\n};",
        "let found = crate::redact::find(text);",
    ];
    for source in named {
        assert!(names_redact(source).is_some(), "{source}");
    }
    let not = [
        "fn redact(text: &str) {}",
        "self.redact_model_feedback()",
        "/// See [`rig_core::redact`].",
        "use rig_core::{message, tool};\nlet redact = 1;",
        ".after(rig_basics::redact::RedactSystems)",
    ];
    for source in not {
        assert_eq!(names_redact(source), None, "{source}");
    }
    assert_eq!(
        names_redact("use std::io;\nuse rig_core::{\n    redact,\n};"),
        Some(3)
    );
}

use super::*;

fn project() -> assert_fs::TempDir {
    let project = assert_fs::TempDir::new().expect("project");
    for (path, contents) in [
        (
            "Cargo.toml",
            include_str!("../../../fixtures/repository_repair/Cargo.toml"),
        ),
        (
            "Cargo.lock",
            include_str!("../../../fixtures/repository_repair/Cargo.lock"),
        ),
        (
            "src/lib.rs",
            include_str!("../../../fixtures/repository_repair/src/lib.rs"),
        ),
        (
            "tests/pagination.rs",
            include_str!("../../../fixtures/repository_repair/tests/pagination.rs"),
        ),
    ] {
        let path = project.path().join(path);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("directories");
        std::fs::write(path, contents).expect("fixture");
    }
    project
}

fn compile_probe(sandbox: &Sandbox, project: &Path, source: &str) -> PathBuf {
    let input = project.join("probe.rs");
    let binary = sandbox.target().join("probe");
    std::fs::write(&input, source).expect("probe source");
    let output = sandbox
        .run(
            project,
            &sandbox.binary("rustc"),
            &[
                "--edition=2024".into(),
                input.display().to_string(),
                "-o".into(),
                binary.display().to_string(),
            ],
            Mode::Compile,
            Duration::from_secs(30),
        )
        .expect("isolated compilation");
    assert_eq!(output.code, Some(0), "{output:?}");
    binary
}

#[test]
fn the_real_project_compiles_and_reports_its_behavioral_failure_in_isolation() {
    let project = project();
    let sandbox = Sandbox::new().expect("sandbox");
    let output = sandbox
        .run(
            project.path(),
            &sandbox.binary("cargo"),
            &[
                "test".into(),
                "--offline".into(),
                "--locked".into(),
                "--no-run".into(),
                "--tests".into(),
                "--message-format=json".into(),
            ],
            Mode::Compile,
            Duration::from_secs(30),
        )
        .expect("compile");
    assert_eq!(output.code, Some(0), "{output:?}");
    let artifact = output
        .stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|value| {
            value["reason"] == "compiler-artifact" && value["target"]["name"] == "pagination"
        })
        .expect("integration-test artifact");
    let binary = PathBuf::from(artifact["executable"].as_str().expect("executable"));
    assert!(
        binary
            .canonicalize()
            .expect("binary")
            .starts_with(sandbox.target().canonicalize().expect("target"))
    );
    let output = sandbox
        .run(
            project.path(),
            &binary,
            &["--test-threads=1".into()],
            Mode::Test,
            Duration::from_secs(10),
        )
        .expect("run tests");
    assert_eq!(output.code, Some(101), "{output:?}");
    assert!(
        output
            .stdout
            .contains("an_offset_past_the_end_returns_an_empty_page ... FAILED")
    );
    assert!(output.stdout.contains("2 passed; 1 failed; 0 ignored"));
    assert!(!output.timed_out && !output.output_limit);
}

#[test]
fn executed_code_cannot_read_host_data_write_project_or_artifacts_or_use_network() {
    let project = project();
    let outside = assert_fs::TempDir::new().expect("outside");
    let canary = outside.path().join("private.txt");
    std::fs::write(&canary, "private canary").expect("canary");
    let canonical_canary = canary.canonicalize().expect("canonical canary");
    let data_alias = format!("/System/Volumes/Data{}", canonical_canary.display());
    let sandbox = Sandbox::new().expect("sandbox");
    let target_canary = sandbox.target().join("protected.txt");
    std::fs::write(&target_canary, "compiled artifact").expect("artifact");
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("host listener");
    listener
        .set_nonblocking(true)
        .expect("nonblocking listener");
    let address = listener.local_addr().expect("host address").to_string();
    let source = format!(
        r#"
fn main() {{
    assert!(std::fs::read_to_string({canary:?}).is_err());
    assert!(std::fs::read_to_string({data_alias:?}).is_err());
    assert!(std::fs::write("src/lib.rs", "changed").is_err());
    assert!(std::fs::write({target_canary:?}, "changed").is_err());
    let address: std::net::SocketAddr = {address:?}.parse().expect("probe address");
    assert!(std::net::TcpStream::connect_timeout(&address, std::time::Duration::from_millis(100)).is_err());
    for name in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "RIG_PROVIDER_TEST_MODE", "CARGO_MANIFEST_DIR"] {{
        assert!(std::env::var_os(name).is_none(), "inherited {{name}}");
    }}
    #[cfg(target_os = "macos")]
    {{
        use std::os::unix::process::CommandExt;
        let denied_exec = std::process::Command::new("/usr/bin/true").exec();
        assert_eq!(denied_exec.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(std::process::Command::new("/usr/bin/true").status().is_err());
        unsafe extern "C" {{ fn fork() -> i32; }}
        assert_eq!(unsafe {{ fork() }}, -1);
    }}
    println!("containment checked");
}}
"#
    );
    let binary = compile_probe(&sandbox, project.path(), &source);
    let output = sandbox
        .run(
            project.path(),
            &binary,
            &[],
            Mode::Test,
            Duration::from_secs(10),
        )
        .expect("run probe");
    assert_eq!(output.code, Some(0), "{output:?}");
    assert_eq!(output.stdout.trim(), "containment checked");
    assert!(
        matches!(listener.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock)
    );
    assert_eq!(
        std::fs::read_to_string(target_canary).expect("artifact"),
        "compiled artifact"
    );
    assert_eq!(
        std::fs::read_to_string(project.path().join("src/lib.rs")).expect("source"),
        include_str!("../../../fixtures/repository_repair/src/lib.rs")
    );
}

#[test]
fn compile_time_file_reads_cannot_reach_host_data() {
    let project = project();
    let outside = assert_fs::TempDir::new().expect("outside");
    let canary = outside.path().join("private.txt");
    std::fs::write(&canary, "DO_NOT_READ_COMPILER_CANARY").expect("canary");
    let sandbox = Sandbox::new().expect("sandbox");
    std::fs::write(
        project.path().join("probe.rs"),
        format!("fn main() {{ println!(\"{{}}\", include_str!({canary:?})); }}"),
    )
    .expect("source");
    let output = sandbox
        .run(
            project.path(),
            &sandbox.binary("rustc"),
            &[
                "--edition=2024".into(),
                "probe.rs".into(),
                "-o".into(),
                sandbox.target().join("probe").display().to_string(),
            ],
            Mode::Compile,
            Duration::from_secs(30),
        )
        .expect("compile");
    assert_ne!(output.code, Some(0), "{output:?}");
    assert!(output.stderr.contains("couldn't read"), "{output:?}");
    assert!(!output.stdout.contains("DO_NOT_READ_COMPILER_CANARY"));
    assert!(!output.stderr.contains("DO_NOT_READ_COMPILER_CANARY"));
}

#[test]
fn a_timed_out_test_is_killed_and_reaped() {
    let project = project();
    let sandbox = Sandbox::new().expect("sandbox");
    let binary = compile_probe(
        &sandbox,
        project.path(),
        "fn main() { loop { std::thread::sleep(std::time::Duration::from_millis(5)); } }",
    );
    let before = Instant::now();
    let output = sandbox
        .run(
            project.path(),
            &binary,
            &[],
            Mode::Test,
            Duration::from_millis(100),
        )
        .expect("bounded run");
    assert!(output.timed_out, "{output:?}");
    assert!(before.elapsed() < Duration::from_secs(5));
    assert_ne!(output.code, Some(0));
}

#[test]
fn excessive_output_stops_the_test_without_growing_the_capture() {
    let project = project();
    let sandbox = Sandbox::new().expect("sandbox");
    let binary = compile_probe(
        &sandbox,
        project.path(),
        "fn main() { loop { println!(\"{}\", \"x\".repeat(8192)); } }",
    );
    let output = sandbox
        .run(
            project.path(),
            &binary,
            &[],
            Mode::Test,
            Duration::from_secs(5),
        )
        .expect("bounded output");
    assert!(output.output_limit, "{output:?}");
    assert!(output.stdout.len() <= OUTPUT_LIMIT);
    assert!(!output.timed_out);
    assert_ne!(output.code, Some(0));
}

#[test]
fn a_successful_exit_does_not_hide_truncated_output() {
    let project = project();
    let sandbox = Sandbox::new().expect("sandbox");
    let binary = compile_probe(
        &sandbox,
        project.path(),
        "fn main() { print!(\"{}\", \"x\".repeat(128 * 1024)); }",
    );
    let output = sandbox
        .run(
            project.path(),
            &binary,
            &[],
            Mode::Test,
            Duration::from_secs(5),
        )
        .expect("bounded output");
    assert!(
        output.output_limit,
        "truncated evidence cannot be accepted: {output:?}"
    );
    assert!(output.stdout.len() <= OUTPUT_LIMIT);
}

#[cfg(target_os = "linux")]
#[test]
fn timeout_closes_pipes_from_descendants_that_created_another_session() {
    let project = project();
    let sandbox = Sandbox::new().expect("sandbox");
    let binary = compile_probe(
        &sandbox,
        project.path(),
        r#"
unsafe extern "C" { fn fork() -> i32; fn setsid() -> i32; }
fn main() {
    let child = unsafe { fork() };
    assert!(child >= 0);
    if child == 0 {
        assert!(unsafe { setsid() } >= 0);
        println!("descendant owns another session and retains output pipes");
    }
    loop { std::thread::sleep(std::time::Duration::from_millis(5)); }
}
"#,
    );
    let before = Instant::now();
    let output = sandbox
        .run(
            project.path(),
            &binary,
            &[],
            Mode::Test,
            Duration::from_millis(250),
        )
        .expect("namespace cleanup closes descendant pipes");
    assert!(output.timed_out, "{output:?}");
    assert!(
        output.stdout.contains("descendant owns another session"),
        "{output:?}"
    );
    assert!(before.elapsed() < Duration::from_secs(5));
}

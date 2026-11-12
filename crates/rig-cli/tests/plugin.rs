#![allow(
    clippy::disallowed_methods,
    reason = "test fixtures run programs and touch files of their own"
)]

use std::fs;
use std::path::Path;
use std::process::{Command, Output, Stdio};

/// `rig <args>` in `dir`, with `home` as `RIG_HOME`.
fn rig(home: &Path, dir: &Path, args: &[&str]) -> std::io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_rig"))
        .env("RIG_HOME", home)
        .env_remove("RIG_SOURCE")
        .current_dir(dir)
        .args(args)
        .output()
}

fn plugin(home: &Path, args: &[&str]) -> std::io::Result<Output> {
    let dir = home.parent().unwrap_or(home);
    rig(home, dir, &[&["plugin"], args].concat())
}

#[test]
fn help_and_refused_commands_use_their_streams_and_leave_the_home_untouched() {
    let result = (|| -> anyhow::Result<()> {
        let root = assert_fs::TempDir::new()?;
        let home = root.path().join("home");
        let home = home.as_path();
        for args in [
            vec!["--help"],
            vec!["new", "--help"],
            vec!["enable", "-h"],
            vec!["enable", "example::Plugin", "--help"],
            vec!["-h", "new", "example"],
            vec!["unknown", "--help"],
        ] {
            let output = plugin(home, &args)?;
            anyhow::ensure!(output.status.success(), "{args:?}: {output:?}");
            anyhow::ensure!(output.stderr.is_empty(), "{args:?}: {output:?}");
            anyhow::ensure!(String::from_utf8(output.stdout)?.contains("rig plugin new <name>"));
            anyhow::ensure!(!home.exists());
        }
        for args in [vec![], vec!["unknown"]] {
            let output = plugin(home, &args)?;
            anyhow::ensure!(output.status.code() == Some(2), "{args:?}: {output:?}");
            anyhow::ensure!(output.stdout.is_empty(), "{args:?}: {output:?}");
            anyhow::ensure!(String::from_utf8(output.stderr)?.starts_with("usage:\n"));
            anyhow::ensure!(!home.exists());
        }
        fs::create_dir(home)?;
        let original = "an invalid settings file\n";
        fs::write(home.join("settings.toml"), original)?;
        let output = plugin(home, &["new", "--help"])?;
        anyhow::ensure!(output.status.success(), "{output:?}");
        anyhow::ensure!(fs::read_to_string(home.join("settings.toml"))? == original);
        Ok(())
    })();
    assert!(result.is_ok(), "{result:?}");
}

#[test]
fn a_fresh_home_lists_the_defaults_with_what_they_add_and_writes_nothing() {
    let result = (|| -> anyhow::Result<()> {
        let root = assert_fs::TempDir::new()?;
        let home = root.path().join("home");
        let output = plugin(&home, &["list"])?;
        anyhow::ensure!(output.status.success(), "{output:?}");
        let text = String::from_utf8(output.stdout)?;
        let row = |plugin: &str| {
            text.lines()
                .find(|line| line.split_whitespace().nth(1) == Some(plugin))
                .map(str::to_owned)
                .unwrap_or_default()
        };
        let tui = row("rig_tui::TuiPlugin");
        anyhow::ensure!(
            tui.starts_with("on ") && tui.contains("The terminal view"),
            "{tui}"
        );
        let mcp = row("rig_mcp_client::McpPlugin");
        anyhow::ensure!(
            mcp.starts_with("off ") && mcp.contains("MCP servers:"),
            "{mcp}"
        );
        anyhow::ensure!(!home.exists(), "a list writes no file");
        Ok(())
    })();
    assert!(result.is_ok(), "{result:?}");
}

/// Writes the shell script `body` to `path` as an executable.
#[cfg(unix)]
fn script(path: &Path, body: &str) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    if let Some(directory) = path.parent() {
        fs::create_dir_all(directory)?;
    }
    fs::write(path, format!("#!/bin/sh\n{body}\n"))?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))
}

/// `rig -p` refuses an agent that does not compile, with the compiler's
/// errors on stderr, and starts no build. A run without `-p` starts the
/// previous build, even with stdin off a terminal. The previous build
/// takes `--json`, which no report of it says yet.
#[cfg(unix)]
#[test]
fn a_print_run_refuses_a_build_that_does_not_compile_and_other_runs_start_the_previous_one() {
    let result = (|| -> anyhow::Result<()> {
        let root = assert_fs::TempDir::new()?;
        let home = root.path().join("home");
        let tools = root.path().join("tools");
        script(
            &tools.join("cargo"),
            "[ \"$1\" = build ] || exit 0\n\
             echo 'error[E0425]: cannot find value `oops` in this scope' >&2\nexit 101",
        )?;
        script(
            &home.join("bin/good"),
            "[ -z \"$RIG_DESCRIBE\" ] || { printf 'rig-describe 1\\nflag\\tjson\\t-\\tJSON\\n' \
             > \"$RIG_DESCRIBE\"; exit 0; }\necho good >> \"$RIG_HOME/ran\"",
        )?;
        let path = std::env::join_paths(std::iter::once(tools).chain(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        )))?;
        let start = |args: &[&str]| {
            Command::new(env!("CARGO_BIN_EXE_rig"))
                .env("RIG_HOME", &home)
                .env("PATH", &path)
                .env_remove("RIG_SOURCE")
                .current_dir(root.path())
                .args(args)
                .stdin(Stdio::null())
                .output()
        };
        for args in [&["-p", "--json", "hi"][..], &["-p", "hi"]] {
            let printed = start(args)?;
            let ran = fs::read_to_string(home.join("ran")).unwrap_or_default();
            anyhow::ensure!(
                ran.is_empty(),
                "a -p run started the previous build: {args:?} {printed:?}"
            );
            anyhow::ensure!(!printed.status.success(), "{printed:?}");
            let said = String::from_utf8(printed.stderr)?;
            anyhow::ensure!(said.contains("error[E0425]"), "{said}");
        }
        let piped = start(&[])?;
        let ran = fs::read_to_string(home.join("ran")).unwrap_or_default();
        anyhow::ensure!(ran == "good\n", "{piped:?}");
        // A `-p` that is not the option, such as a word after `--`.
        let word = start(&["--", "-p"])?;
        let said = String::from_utf8(word.stderr)?;
        anyhow::ensure!(
            said.contains("starting the previous build"),
            "a word `-p` refused the previous build: {said}"
        );
        Ok(())
    })();
    assert!(result.is_ok(), "{result:?}");
}

/// A start refused over a plugin name rig does not know names the command
/// that takes it out, `rig plugin disable`, which works while the name is
/// in either list.
#[test]
fn disable_takes_out_an_unknown_plugin_the_start_refuses_over() {
    let result = (|| -> anyhow::Result<()> {
        let root = assert_fs::TempDir::new()?;
        let home = root.path().join("home");
        let home = home.as_path();
        fs::create_dir(home)?;
        fs::write(
            home.join("settings.toml"),
            "[plugins]\nenable = [\"gone::GonePlugin\"]\ndisable = [\"lost::LostPlugin\"]\n",
        )?;
        for name in ["gone::GonePlugin", "lost::LostPlugin"] {
            let refused = rig(home, root.path(), &["-p", "hi"])?;
            let said = String::from_utf8(refused.stderr)?;
            let fix = format!("`rig plugin disable {name}`");
            anyhow::ensure!(
                said.contains(&fix),
                "the refusal does not name {fix}: {said}"
            );
            let output = plugin(home, &["disable", name])?;
            anyhow::ensure!(output.status.success(), "{output:?}");
        }
        let settings = fs::read_to_string(home.join("settings.toml"))?;
        anyhow::ensure!(!settings.contains("Plugin"), "{settings}");
        let output = plugin(home, &["list"])?;
        anyhow::ensure!(output.status.success(), "{output:?}");
        Ok(())
    })();
    assert!(result.is_ok(), "{result:?}");
}

/// `rig plugin check --build` refuses settings a start would refuse: the
/// built agent reads the home's settings file as it describes itself, and
/// the check passes on what it says is wrong with them, failing, not
/// saying to restart.
#[cfg(unix)]
#[test]
fn check_build_refuses_settings_the_built_agent_would_refuse() {
    let result = (|| -> anyhow::Result<()> {
        let root = assert_fs::TempDir::new()?;
        let home = root.path().join("home");
        let tools = root.path().join("tools");
        script(&tools.join("cargo"), "exit 0")?;
        script(
            &home.join("target/debug/rig-harness-agent"),
            "if [ -n \"$RIG_DESCRIBE_SETTINGS\" ] && grep -q max_depth \"$RIG_DESCRIBE_SETTINGS\"; then\n\
             echo \"rig: $RIG_DESCRIBE_SETTINGS [subagents]: unknown field \\`max_depth\\`\" >&2; exit 2\n\
             fi\nprintf 'rig-describe 1\\n' > \"$RIG_DESCRIBE\"",
        )?;
        let path = std::env::join_paths(std::iter::once(tools).chain(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        )))?;
        let check = || {
            Command::new(env!("CARGO_BIN_EXE_rig"))
                .env("RIG_HOME", &home)
                .env("PATH", &path)
                .env_remove("RIG_SOURCE")
                .current_dir(root.path())
                .args(["plugin", "check", "--build"])
                .stdin(Stdio::null())
                .output()
        };
        let settings = home.join("settings.toml");
        fs::write(&settings, "[subagents]\nmodel = \"ollama/qwen3\"\n")?;
        let good = check()?;
        anyhow::ensure!(good.status.success(), "{good:?}");
        fs::write(&settings, "[subagents]\nmax_depth = 3\n")?;
        let refused = check()?;
        let said = String::from_utf8(refused.stderr)?;
        let printed = String::from_utf8(refused.stdout)?;
        anyhow::ensure!(!refused.status.success(), "{printed}{said}");
        anyhow::ensure!(
            said.contains(&format!("{} [subagents]", settings.display()))
                && said.contains("max_depth")
                && !printed.contains("/reload"),
            "{printed}{said}"
        );
        Ok(())
    })();
    assert!(result.is_ok(), "{result:?}");
}

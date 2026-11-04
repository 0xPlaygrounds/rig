use std::fs;
use std::path::Path;
use std::process::{Command, Output};

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

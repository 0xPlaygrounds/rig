use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn plugin(home: &Path, args: &[&str]) -> std::io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_rig"))
        .env("RIG_HOME", home)
        .env_remove("RIG_SOURCE")
        .arg("plugin")
        .args(args)
        .output()
}

#[test]
fn help_and_refused_commands_use_their_streams_and_leave_the_home_untouched() {
    let result = (|| -> anyhow::Result<()> {
        let root = assert_fs::TempDir::new()?;
        let home = root.path().join("home");
        let home = home.as_path();
        for args in [
            vec!["--help"],
            vec!["add", "--help"],
            vec!["enable", "-h"],
            vec!["add", "example::Plugin", "--path", "--help"],
            vec!["-h", "new", "example"],
            vec!["unknown", "--help"],
        ] {
            let output = plugin(home, &args)?;
            anyhow::ensure!(output.status.success(), "{args:?}: {output:?}");
            anyhow::ensure!(output.stderr.is_empty(), "{args:?}: {output:?}");
            anyhow::ensure!(String::from_utf8(output.stdout)?.contains("rig plugin add <type>"));
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
        let original = "an invalid plugin list\n";
        fs::write(home.join("plugins.toml"), original)?;
        let output = plugin(home, &["add", "--help"])?;
        anyhow::ensure!(output.status.success(), "{output:?}");
        anyhow::ensure!(fs::read_to_string(home.join("plugins.toml"))? == original);
        Ok(())
    })();
    assert!(result.is_ok(), "{result:?}");
}

#[test]
fn list_exposes_a_missing_shipped_plugin_that_the_picker_can_enable() {
    let result = (|| -> anyhow::Result<()> {
        let root = assert_fs::TempDir::new()?;
        let home = root.path();
        let original = "# An older home.\n";
        fs::write(home.join("plugins.toml"), original)?;
        let type_path = "rig_mcp_client::McpPlugin";
        let output = plugin(home, &["list"])?;
        anyhow::ensure!(output.status.success(), "{output:?}");
        let text = String::from_utf8(output.stdout)?;
        let row = text
            .lines()
            .find(|line| line.split_whitespace().nth(1) == Some(type_path));
        anyhow::ensure!(
            row.is_some_and(|row| {
                let mut words = row.split_whitespace();
                words.next() == Some("off")
                    && words.next() == Some(type_path)
                    && words.next() == Some("rig-mcp-client")
                    && row.contains("MCP servers:")
                    && row.contains("not in plugins.toml")
                    && row.contains("rig plugin enable")
            }),
            "{row:?}"
        );
        anyhow::ensure!(fs::read_to_string(home.join("plugins.toml"))? == original);
        let enabled = plugin(home, &["enable", type_path])?;
        anyhow::ensure!(enabled.status.success(), "{enabled:?}");
        anyhow::ensure!(String::from_utf8(enabled.stdout)?.contains("Added"));
        let output = plugin(home, &["list"])?;
        anyhow::ensure!(output.status.success(), "{output:?}");
        let text = String::from_utf8(output.stdout)?;
        let rows: Vec<_> = text
            .lines()
            .filter(|line| line.split_whitespace().nth(1) == Some(type_path))
            .collect();
        anyhow::ensure!(rows.len() == 1);
        anyhow::ensure!(
            rows.first()
                .is_some_and(|row| row.starts_with("on ") && !row.contains("not in plugins.toml"))
        );
        Ok(())
    })();
    assert!(result.is_ok(), "{result:?}");
}

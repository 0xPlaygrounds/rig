use std::sync::atomic::{AtomicBool, AtomicU32};

use super::{ShellArgs, run};

fn shell(command: &str, spill: Option<crate::Spill>) -> String {
    let args = ShellArgs {
        command: command.to_owned(),
        timeout_secs: None,
    };
    run(
        args,
        &[],
        spill,
        &AtomicBool::new(false),
        &AtomicU32::new(0),
    )
    .unwrap_or_default()
}

/// A process that leaves the command's process group keeps the output pipe
/// open after the kill; the call must still return.
#[cfg(target_os = "linux")]
#[test]
fn returns_when_a_process_outside_the_group_holds_the_pipe() {
    use std::time::{Duration, Instant};

    let started = Instant::now();
    let output = shell("setsid sleep 5 & echo started", None);
    assert!(started.elapsed() < Duration::from_secs(3));
    assert!(output.contains("started"), "{output}");
    assert!(output.ends_with("[exit code 0]"), "{output}");
}

/// Output cut by lines or by bytes is kept whole, and `read` reads it back
/// by the path the cut output names.
#[cfg(unix)]
#[test]
fn cut_output_is_read_back_by_its_handle() {
    use rig_core::tool::PortableTool;

    let dir = std::env::temp_dir().join(format!("rig-tools-spill-{}", std::process::id()));
    let spill = Some(crate::Spill(dir.clone()));
    for (command, lines) in [("seq 1 3000", 3000), ("seq 1 300000", 300000)] {
        let output = shell(command, spill.clone());
        let handle = output
            .lines()
            .next()
            .and_then(|line| line.split_once(" is in "))
            .and_then(|(_, rest)| rest.split_once(':'))
            .map(|(path, _)| path.to_owned())
            .unwrap_or_default();
        let args = serde_json::json!({ "path": handle, "offset": lines / 2, "limit": 1 });
        let read = serde_json::from_value(args)
            .map(|args| futures::executor::block_on(crate::Read.call(args)));
        let wanted = format!("{}\t{}\n[", lines / 2, lines / 2);
        assert!(
            matches!(&read, Ok(Ok(text)) if text.contains(&wanted)),
            "{output:.200} {read:?}"
        );
    }
    assert!(shell("seq 1 3", spill).starts_with("1\n2\n3\n"));
    assert_eq!(std::fs::read_dir(&dir).map(Iterator::count).ok(), Some(2));
    std::fs::remove_dir_all(dir).ok();
}

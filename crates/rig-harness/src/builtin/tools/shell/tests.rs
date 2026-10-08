/// A process that leaves the command's process group keeps the output pipe
/// open after the kill; the call must still return.
#[cfg(target_os = "linux")]
#[test]
fn returns_when_a_process_outside_the_group_holds_the_pipe() {
    use std::sync::atomic::{AtomicBool, AtomicU32};
    use std::time::{Duration, Instant};

    use super::{ShellArgs, run};

    let started = Instant::now();
    let output = run(
        ShellArgs {
            command: "setsid sleep 5 & echo started".to_owned(),
            timeout_secs: None,
        },
        &AtomicBool::new(false),
        &AtomicU32::new(0),
    );
    assert!(started.elapsed() < Duration::from_secs(3));
    let output = output.unwrap_or_default();
    assert!(output.contains("started"), "{output}");
    assert!(output.ends_with("[exit code 0]"), "{output}");
}

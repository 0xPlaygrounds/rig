use rig_harness::prelude::{Processes, Tool};
use rig_harness::rig_core::tool::ToolContext;
use rig_tools::{Shell, ShellArgs};

use super::{SharedShell, ShellOutput};

/// The shell tool on `processes`, cutting as `output` says.
fn shell(processes: &Processes, output: &ShellOutput) -> SharedShell {
    SharedShell {
        shell: Shell::default(),
        processes: processes.clone(),
        output: output.clone(),
    }
}

/// What `shell` answers `command`.
fn run(shell: &SharedShell, command: &str) -> String {
    let args = ShellArgs::new(command);
    bevy_tasks::block_on(shell.call(&mut ToolContext::new(), args)).unwrap_or_default()
}

#[test]
fn each_shell_call_starts_as_the_processes_say_at_the_time() {
    let processes = Processes::default();
    let shell = shell(&processes, &ShellOutput::default());
    let greeting = "echo \"greeting: $GREETING\"";
    assert!(run(&shell, greeting).contains("greeting: \n"));
    processes.set(processes.launcher().with_env("GREETING", "hello"));
    let output = run(&shell, greeting);
    assert!(output.contains("greeting: hello"), "{output}");
}

#[test]
fn without_a_cut_the_shell_returns_every_line() {
    let output = ShellOutput::default();
    let shell = shell(&Processes::default(), &output);
    let lines = rig_tools::MAX_LINES + 500;
    let count = format!("seq 1 {lines}");
    let numbers = |text: &str| {
        text.lines()
            .filter(|line| line.parse::<usize>().is_ok())
            .count()
    };
    assert!(numbers(&run(&shell, &count)) <= rig_tools::MAX_LINES);
    output.set(None);
    let whole = run(&shell, &count);
    assert_eq!(numbers(&whole), lines, "{whole:.200}");
}

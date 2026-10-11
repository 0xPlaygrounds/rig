use rig_harness::prelude::{Processes, Tool};
use rig_harness::rig_core::tool::ToolContext;
use rig_tools::{Shell, ShellArgs};

use super::SharedShell;

#[test]
fn each_shell_call_starts_as_the_processes_say_at_the_time() {
    let processes = Processes::default();
    let shell = SharedShell {
        shell: Shell::default(),
        processes: processes.clone(),
    };
    let run = |shell: &SharedShell| {
        let args = ShellArgs::new("echo \"greeting: $GREETING\"");
        bevy_tasks::block_on(shell.call(&mut ToolContext::new(), args)).unwrap_or_default()
    };
    assert!(run(&shell).contains("greeting: \n"), "{}", run(&shell));
    processes.set(processes.launcher().with_env("GREETING", "hello"));
    assert!(run(&shell).contains("greeting: hello"), "{}", run(&shell));
}

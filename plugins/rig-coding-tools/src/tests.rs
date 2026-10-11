use std::time::{Duration, Instant};

use rig_harness::prelude::{Agent, AgentPlugin, App, Processes, TaskPoolPlugin, Tool, WorkDir};
use rig_harness::rig_core::serde_json::json;
use rig_harness::rig_core::tool::ToolContext;
use rig_harness_test_support::{ToolOutputs, log_tool_outputs, start_call};
use rig_tools::{Shell, ShellArgs};

use super::{ReadTool, SharedShell, ShellOutput, ShellTool};

/// The shell tool on `processes`, cutting as `output` says.
fn shell(processes: &Processes, output: &ShellOutput) -> SharedShell {
    SharedShell {
        shell: Shell::default(),
        processes: processes.clone(),
        output: output.clone(),
        dir: None,
    }
}

#[test]
fn an_agent_with_a_work_dir_reads_and_runs_commands_there() {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("work-dir-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(dir.join("note.txt"), "in the work dir").ok();
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, ReadTool, ShellTool))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.spawn((Agent, WorkDir(dir.clone()))).id();
    let calls = [
        ("read", json!({ "path": "note.txt" })),
        ("read", json!({ "path": "../note.txt" })),
        ("shell", json!({ "command": "pwd" })),
    ];
    for (n, (tool, args)) in calls.into_iter().enumerate() {
        start_call(world, agent, (tool, &format!("c{n}")), args);
    }
    let started = Instant::now();
    let output = |app: &App, call: &str| app.world().resource::<ToolOutputs>().0.get(call).cloned();
    while output(&app, "c2").is_none() && started.elapsed() < Duration::from_secs(10) {
        app.update();
        std::thread::sleep(Duration::from_millis(5));
    }
    let [read, outside, pwd] =
        ["c0", "c1", "c2"].map(|call| output(&app, call).unwrap_or_default());
    assert!(read.contains("in the work dir"), "{read}");
    assert!(
        !outside.is_empty() && !outside.contains("in the work dir"),
        "{outside}"
    );
    let canonical = dir.canonicalize().unwrap_or_default();
    assert!(pwd.contains(&*canonical.to_string_lossy()), "{pwd}");
    std::fs::remove_dir_all(&dir).ok();
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

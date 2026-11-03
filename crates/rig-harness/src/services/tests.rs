#![allow(clippy::disallowed_methods, reason = "test fixtures")]

use std::path::{Path, PathBuf};

use bevy_app::prelude::*;
use rig_harness_protocol::{Home, env};

use super::{Processes, Project, ProjectRoot, RigHome, ServicesPlugin, TokioRuntime};

#[test]
fn what_the_app_inserted_stays_and_the_rest_comes_from_the_process() {
    let mut app = App::new();
    app.insert_resource(RigHome(Home::at("/srv/rig-home")))
        .insert_resource(ProjectRoot(PathBuf::from("/srv/project")))
        .add_plugins(ServicesPlugin);
    let world = app.world();
    assert_eq!(
        world.resource::<RigHome>().0.root(),
        PathBuf::from("/srv/rig-home")
    );
    assert_eq!(
        world.resource::<ProjectRoot>().0,
        PathBuf::from("/srv/project")
    );
    // No child process acts as this agent.
    let processes = world.resource::<Processes>();
    let command = processes.launcher().command("true");
    let unset: Vec<_> = command
        .get_envs()
        .filter(|(_, value)| value.is_none())
        .map(|(name, _)| name.to_string_lossy().into_owned())
        .collect();
    assert!(
        env::AGENT_ONLY
            .iter()
            .all(|name| unset.contains(&(*name).to_owned()))
    );
    // A change reaches every holder of the resource, a tool's too.
    let held = processes.clone();
    processes.set(processes.launcher().wrapped(["nice"]));
    assert_eq!(held.launcher().prefix, ["nice"]);
    // Plugins run futures on rig's own runtime.
    let answer = world
        .get_resource::<TokioRuntime>()
        .map(|runtime| runtime.0.block_on(async { 6 * 7 }));
    assert_eq!(answer, Some(42));
}

/// A scratch directory standing for the user's home: holding `.git`, as a
/// home kept in git does, and `.rig`, which is `RIG_HOME`.
fn scratch_user(name: &str) -> Option<PathBuf> {
    let user = std::env::temp_dir().join(format!("rig-project-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&user).ok();
    std::fs::create_dir_all(user.join(".git")).ok()?;
    std::fs::create_dir_all(user.join(".rig")).ok()?;
    user.canonicalize().ok()
}

/// Makes each of `dirs` under `base`; a name ending in `!` is made a file.
fn make(base: &Path, dirs: &[&str]) {
    for dir in dirs {
        match dir.strip_suffix('!') {
            Some(file) => {
                let file = base.join(file);
                file.parent().map(std::fs::create_dir_all);
                std::fs::write(file, "").ok();
            }
            None => {
                std::fs::create_dir_all(base.join(dir)).ok();
            }
        }
    }
}

/// The root is the nearest directory holding `.git` or a `.rig` directory,
/// below the user's home, else the working directory; the config is its
/// `.rig`, but for the home folder and a `.rig` that is `RIG_HOME`.
#[test]
fn the_project_is_the_nearest_marked_directory_below_the_users_home() {
    let made = scratch_user("rows");
    assert!(made.is_some(), "a scratch home");
    let Some(user) = made else {
        return;
    };
    make(
        &user,
        &[
            "r/.git",
            "r/a/b",
            "r/v/.git",
            "r/v/s",
            "r/m/.git!",
            "r/m/s",
            "r/pkg/.rig",
            "r/pkg/src",
            "r/other",
            "x/y",
            "z/.rig",
            "z/y",
            "f/.rig!",
            "f/y",
            "t/.git",
            "t/.rig",
            "t/a",
        ],
    );
    let rig_home = user.join(".rig");
    let find = |dir: &str| Project::find(&user.join(dir), &rig_home, Some(&user));
    let at = |root: &str| Project {
        root: user.join(root),
        config: Some(user.join(root).join(".rig")),
    };
    let rows = [
        ("r", at("r")),
        ("r/a/b", at("r")),
        ("x/y", at("x/y")),
        ("z/y", at("z")),
        ("r/v/s", at("r/v")),
        ("r/m/s", at("r/m")),
        ("r/pkg/src", at("r/pkg")),
        ("r/other", at("r")),
        ("t/a", at("t")),
        ("f/y", at("f/y")),
        ("gone", at("gone")),
    ];
    for (dir, expected) in rows {
        assert_eq!(find(dir), expected, "{dir}");
    }
    let home = Project {
        root: user.clone(),
        config: None,
    };
    assert_eq!(find(""), home);
    // A RIG_HOME in a project's `.rig` leaves it no config, and marks nothing.
    make(&user, &["r/.rig", "x/.rig"]);
    let custom = Project::find(&user.join("r/a"), &user.join("r/.rig"), Some(&user));
    assert_eq!(custom.config, None);
    let custom = Project::find(&user.join("x/y"), &user.join("x/.rig"), Some(&user));
    assert_eq!(custom, at("x/y"));
    std::fs::remove_dir_all(user).ok();
}

/// A working directory reached through a link is in the project its target
/// is in, not the link's.
#[cfg(unix)]
#[test]
fn a_linked_working_directory_is_in_its_targets_project() {
    let made = scratch_user("links");
    assert!(made.is_some(), "a scratch home");
    let Some(user) = made else {
        return;
    };
    make(&user, &["r/.git", "r/a", "q/.git", "x"]);
    let link =
        |target: &str, link: &str| std::os::unix::fs::symlink(user.join(target), user.join(link));
    assert!(link("r/a", "q/into-repo").is_ok() && link("x", "q/out").is_ok());
    let find = |dir: &str| Project::find(&user.join(dir), &user.join(".rig"), Some(&user)).root;
    assert_eq!(find("q/into-repo"), user.join("r"));
    assert_eq!(find("q/out"), user.join("x"));
    let project = Project::find(&user.join("q/into-repo"), &user.join(".rig"), Some(&user));
    assert_eq!(
        project.levels(&user.join("q/into-repo")),
        [user.join("r/a"), user.join("r")]
    );
    std::fs::remove_dir_all(user).ok();
}

#[test]
fn without_one_the_project_is_the_working_directory() {
    let mut app = App::new();
    app.add_plugins(ServicesPlugin);
    let here = std::env::current_dir().ok();
    assert_eq!(
        app.world()
            .get_resource::<ProjectRoot>()
            .map(|root| root.0.clone()),
        here
    );
}

/// The secret this file's tests run a copy of themselves with.
const PROBE: &str = "RIG_SECRET_PROBE";

/// Runs the test `name` of this binary again with a secret in its
/// environment, and whether it ran and passed.
#[cfg(unix)]
fn with_a_secret(name: &str) -> bool {
    let output = std::env::current_exe().and_then(|test| {
        std::process::Command::new(test)
            .args(["--exact", name, "--nocapture", "--test-threads", "1"])
            .env(PROBE, "hunter2")
            .output()
    });
    output.is_ok_and(|output| {
        output.status.success() && String::from_utf8_lossy(&output.stdout).contains("1 passed")
    })
}

#[cfg(unix)]
#[test]
fn no_secret_reaches_a_process_started_through_processes() {
    if std::env::var_os(PROBE).is_none() {
        let name = "services::tests::no_secret_reaches_a_process_started_through_processes";
        assert!(with_a_secret(name), "the run with a secret failed");
        return;
    }
    // A sandbox replaces the whole launcher; the secret is still kept out.
    let processes = Processes::default();
    processes.set(rig_tools::process::Launcher::default().wrapped(["env"]));
    let output = processes
        .launcher()
        .command("sh")
        .args(["-c", "printf %s \"${RIG_SECRET_PROBE-none}\""])
        .output();
    let seen = output.map(|output| String::from_utf8_lossy(&output.stdout).into_owned());
    assert_eq!(seen.ok().as_deref(), Some("none"));
}

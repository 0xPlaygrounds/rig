//! [`RemoteHostsPlugin`]: paths written `HOST:PATH` reach the hosts
//! `[remote.hosts]` names, through the transport it gives each.

use std::collections::BTreeMap;
use std::sync::Arc;

use rig_harness::prelude::{
    App, AppSettingsExt, Messages, Name, Notice, Plugin, Processes, ProjectWorkspace,
    PromptSection, Resource, Tool, ToolDef,
};
use rig_harness::rig_core::serde::Deserialize;
use rig_tools::process::Launcher;
use rig_tools::remote::{Hosts, Remote};
use rig_tools::{Edit, Read, Search, Shell, Write};

/// `[remote]` in `settings.toml`, read by [`RemoteHostsPlugin`]. Only the
/// user's settings are read, never a project's: a transport is a command
/// rig runs.
///
/// ```toml
/// [remote.hosts]
/// bench = ["koh", "exec", "bench", "--sh"]
/// box = ["ssh", "-o", "BatchMode=yes", "dev@box"]
/// ```
#[derive(Resource, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
#[serde(crate = "rig_harness::rig_core::serde", default, deny_unknown_fields)]
pub struct RemoteSettings {
    /// Each host's transport, by the name paths call it: a command that
    /// runs its last argument, a shell line, there with the remote user's
    /// shell in their home, keeps stdout and stderr apart, and ends with
    /// 255 when it failed itself, as `ssh HOST` and `koh exec HOST --sh` do.
    /// Options of its own, such as ssh's `-o BatchMode=yes`, go in it.
    pub hosts: BTreeMap<String, Vec<String>>,
}

impl RemoteSettings {
    /// Each host's [`Remote`], its transport started under `launcher`'s
    /// policy, and a line about each host that is left out, such as one
    /// with no transport.
    pub fn remotes(&self, launcher: &Launcher) -> (Vec<Remote>, Vec<String>) {
        let mut remotes = Vec::new();
        let mut problems = Vec::new();
        for (name, transport) in &self.hosts {
            match Remote::new(name, transport, launcher.clone()) {
                Ok(remote) => remotes.push(remote),
                Err(error) => problems.push(format!("[remote.hosts] {name} is left out: {error}")),
            }
        }
        (remotes, problems)
    }
}

/// Paths written `HOST:PATH`, `HOST` one of `[remote.hosts]`
/// ([`RemoteSettings`]), are that host's: `read bench:~/app/src/lib.rs`,
/// `edit`, `write`, `search` with `path: "bench:~/app"`, and `shell` with
/// `cwd: "bench:~/app"` act there through the host's transport, and every
/// other path stays on this machine. The tools name remote files in the
/// same form, so the model passes them back as they are. When the app
/// finishes building, it wraps the [`ProjectWorkspace`] in a
/// [`Hosts`], adds a system-prompt section naming the hosts, and says so
/// in the `path` of `read`, `edit`, `write` and `search` and the `cwd` of
/// `shell`, and in `read`'s description that remote files are worked on in place; without hosts it does none of these.
#[derive(Default)]
pub struct RemoteHostsPlugin;

impl Plugin for RemoteHostsPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<RemoteSettings>("remote")
            .init_resource::<Processes>()
            .init_resource::<ProjectWorkspace>();
    }

    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let launcher = world.resource::<Processes>().launcher();
        let (remotes, problems) = world.resource::<RemoteSettings>().remotes(&launcher);
        if world.contains_resource::<Messages<Notice>>() {
            for problem in problems {
                world.write_message(Notice::error(None, problem));
            }
        }
        let names: Vec<String> = remotes
            .iter()
            .map(|remote| remote.name().to_owned())
            .collect();
        let Some(first) = names.first() else {
            return;
        };
        let text = prompt(first, &names);
        let inner = world.resource::<ProjectWorkspace>().0.clone();
        world.insert_resource(ProjectWorkspace(Arc::new(Hosts::new(inner, remotes))));
        world.spawn((
            Name::new("prompt:remote_hosts"),
            PromptSection::new(PromptSection::ORDER_PROJECT - 50, "remote_hosts", text),
        ));
        let hosts = names.join(", ");
        let mut tools = world.query::<&mut ToolDef>();
        for mut def in tools.iter_mut(world) {
            let def = Arc::make_mut(&mut def.0);
            let name = def.name.as_str();
            let (argument, note) = if name == Shell::NAME {
                let note = format!(
                    " Or HOST:DIR, to run the command on that remote host ({hosts}), such as \
                     `{first}:~/app`."
                );
                ("cwd", note)
            } else if [Read::NAME, Edit::NAME, Write::NAME, Search::NAME].contains(&name) {
                let note = format!(
                    " Or HOST:PATH, a file on that remote host ({hosts}), such as \
                     `{first}:~/app/src/lib.rs`."
                );
                ("path", note)
            } else {
                continue;
            };
            if name == Read::NAME {
                def.description.push_str(&format!(
                    " Work on the files of a remote host ({hosts}) in place, by HOST:PATH paths \
                     in `read`, `edit`, `write` and `search` and `shell` with `cwd` HOST:DIR, \
                     not on copies here."
                ));
            }
            let pointer = format!("/properties/{argument}/description");
            if let Some(rig_harness::rig_core::serde_json::Value::String(text)) =
                def.parameters.pointer_mut(&pointer)
            {
                text.push_str(&note);
            }
        }
    }
}

/// The system prompt's section on the hosts `names`, its examples on
/// `first`.
fn prompt(first: &str, names: &[String]) -> String {
    let names: Vec<String> = names.iter().map(|name| format!("`{name}`")).collect();
    format!(
        "Remote hosts: {}. A path written HOST:PATH is that host's file, as scp reads it: \
         `{first}:~/app/src/lib.rs`, `{first}:/srv/app` (a relative PATH is under the remote \
         home). `read`, `edit`, `write` and `search` act on it there, and `shell` with `cwd` \
         set to such a directory, such as `{first}:~/app`, runs the command there. Every \
         other path is on this machine. Read and edit remote files in place this way, not \
         through copies here. The tools name remote files in the same form: pass those paths \
         back as they are. A `shell` exit code of 255 from a host can mean its connection \
         failed. A `shell` timeout stops the connection on this machine, which may leave the \
         command on the host running: to bound a long command there, run it under `timeout` \
         there.",
        names.join(", ")
    )
}

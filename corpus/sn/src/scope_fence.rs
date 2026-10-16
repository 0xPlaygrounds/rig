//! SN-01: a scope fence for file changes. `/scope <dir>…` fences the
//! session's file changes to those directories of the project; `/scope`
//! shows the fence and `/scope off` lifts it.
//!
//! A Gate system reads each tool call's `Footprint`, the paths the call
//! writes as its tool resolved them (`..` folded, links followed), so one
//! rule covers `edit`, `write`, `delete`, `move` and any tool that declares
//! the paths it writes. A call that writes outside the fence is refused and
//! the turn goes on. Calls that only read are never fenced, and neither is
//! a tool that may change anything (`Writes(Anything)`), such as `shell`:
//! the fence guards tool edits, it is no sandbox. The fence is a saved
//! resource of the session, so every agent of the session, subagents
//! included, is fenced by it, and it comes back on `/reload` and `/resume`.

use std::path::{Path, PathBuf};

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_harness::rig_tools::workspace::{Stat, Workspace};

#[cfg(all(test, feature = "tui"))]
mod tests;

/// The decider named on a refusal.
const NAME: &str = "scope-fence";

/// The tag of the prompt section that tells agents about the fence.
const SECTION: &str = "scope_fence";

/// The fence of this session: directories relative to the project root,
/// each ending in `/`; empty when no fence is set. Saved with the session.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Saved)]
pub struct ScopeFence {
    /// The fenced directories, such as `src/parser/`.
    pub dirs: Vec<String>,
}

impl ScopeFence {
    /// The directories as one line: `src/parser/, src/lexer/`.
    pub fn line(&self) -> String {
        self.dirs.join(", ")
    }

    /// The refusal a call that writes outside the fence gets.
    pub fn refusal(&self) -> String {
        format!(
            "Outside the current scope ({}). Ask the user to widen the scope with /scope.",
            self.line()
        )
    }
}

/// The plugin's Gate system, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeFenceSystems;

/// `/scope`, the fence on file-changing tool calls, its prompt section and
/// its status-line item.
#[derive(Default)]
pub struct ScopeFencePlugin;

impl Plugin for ScopeFencePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScopeFence>()
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .configure_sets(AgentPass, ScopeFenceSystems.in_set(Stage::Gate))
            .add_systems(AgentPass, fence.in_set(ScopeFenceSystems))
            .add_systems(
                Update,
                tell_agents
                    .run_if(resource_changed::<ScopeFence>)
                    .before(AgentLoop),
            )
            .add_observer(checked)
            .add_command(
                "scope",
                "Fence file changes to directories: /scope <dir>…, /scope off",
                scope,
            );
        app.world_mut().spawn((
            ScopeSection,
            PromptSection::new(PromptSection::ORDER_ROLE + 50, SECTION, ""),
        ));
        #[cfg(feature = "tui")]
        status::add(app);
    }
}

/// On the fence's prompt section.
#[derive(Component)]
struct ScopeSection;

/// Refuses each tool call about to run that writes a path outside the
/// fence, as the agent's tools resolve the fenced directories.
fn fence(
    mut calls: Query<(&Footprint, &ForAgent, &mut Decision), (ReadyToStart, ToolCalls)>,
    fence: Res<ScopeFence>,
    dirs: Query<&WorkDir>,
    root: Res<ProjectRoot>,
    workspace: Res<ProjectWorkspace>,
) {
    if fence.dirs.is_empty() {
        return;
    }
    for (Footprint(touches), &ForAgent(agent), mut decision) in &mut calls {
        let Touches::Writes(Scope::Paths(paths)) = touches else {
            continue;
        };
        let base = dirs.get(agent).map_or(&root.0, |dir| &dir.0);
        // `canonical` may run on the main thread: footprints are made with it.
        let allowed: Vec<PathBuf> = fence
            .dirs
            .iter()
            .map(|dir| workspace.0.resolve(Some(base), dir))
            .collect();
        let inside = |path: &PathBuf| allowed.iter().any(|dir| path.starts_with(dir));
        if !paths.iter().all(inside) {
            decision.decide(Decider::plugin(NAME), Outcome::refused(fence.refusal()));
        }
    }
}

/// `/scope`, `/scope off` and `/scope <dir>…`.
fn scope(
    In(args): In<CommandArgs>,
    mut fence: ResMut<ScopeFence>,
    root: Res<ProjectRoot>,
    workspace: Res<ProjectWorkspace>,
    wake: Res<Wake>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let words: Vec<String> = args.args.split_whitespace().map(str::to_owned).collect();
    match words.as_slice() {
        [] if fence.dirs.is_empty() => {
            notices.write(Notice::info(args.agent, "No scope is set."));
        }
        [] => {
            notices.write(Notice::info(args.agent, format!("Scope: {}", fence.line())));
        }
        [off] if off == "off" => {
            fence.dirs.clear();
            notices.write(Notice::info(args.agent, "Scope removed."));
        }
        _ => {
            let (workspace, root) = (workspace.0.clone(), root.0.clone());
            if workspace.host().is_none() {
                apply(
                    check(&*workspace, &root, &words),
                    args.agent,
                    &mut fence,
                    &mut notices,
                );
                return;
            }
            let pool = IoTaskPool::get_or_init(TaskPool::default);
            let checking = blocking(move || Ok(check(&*workspace, &root, &words)));
            let work = async move {
                Checked(
                    checking
                        .await
                        .unwrap_or_else(|failed| Err(failed.to_string())),
                )
            };
            commands.spawn((
                Name::new("checking the scope"),
                ScopeCheck(args.agent),
                Running::spawn(pool, &wake, work),
            ));
        }
    }
}

/// A `/scope` check in another workspace, for the agent it was typed for.
#[derive(Component)]
struct ScopeCheck(Entity);

/// What a check found: the fence's directories, or why there is none.
struct Checked(Result<Vec<String>, String>);

/// Sets the fence a check in another workspace found.
fn checked(
    done: On<Insert<Done<Checked>>>,
    checks: Query<(&ScopeCheck, &Done<Checked>)>,
    mut fence: ResMut<ScopeFence>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if let Ok((&ScopeCheck(agent), Done(Checked(found)))) = checks.get(done.entity) {
        apply(found.clone(), agent, &mut fence, &mut notices);
        commands.entity(done.entity).despawn();
    }
}

/// Sets the fence to `found`'s directories, or says why it is not set.
fn apply(
    found: Result<Vec<String>, String>,
    agent: Entity,
    fence: &mut ScopeFence,
    notices: &mut MessageWriter<Notice>,
) {
    match found {
        Ok(dirs) => {
            fence.dirs = dirs;
            notices.write(Notice::info(agent, format!("Scope: {}", fence.line())));
        }
        Err(why) => {
            notices.write(Notice::error(agent, why));
        }
    }
}

/// Each of `dirs` as the fence keeps it, relative to the project `root`
/// and ending in `/`, or why one cannot be fenced: it is missing, not a
/// directory, or outside the project.
fn check(workspace: &dyn Workspace, root: &Path, dirs: &[String]) -> Result<Vec<String>, String> {
    let top = workspace.resolve(Some(root), ".");
    let mut fenced = Vec::new();
    for dir in dirs {
        let resolved = workspace.resolve(Some(root), dir);
        let Ok(relative) = resolved.strip_prefix(&top) else {
            return Err(format!("/scope: {dir} is outside the project"));
        };
        match workspace.stat(&resolved) {
            Ok(Stat::Dir) => {}
            Ok(Stat::Missing) | Err(_) => return Err(format!("/scope: {dir} does not exist")),
            Ok(_) => return Err(format!("/scope: {dir} is not a directory")),
        }
        let relative = relative.to_string_lossy();
        let shown = match relative.is_empty() {
            true => "./".to_owned(),
            false => format!("{relative}/"),
        };
        if !fenced.contains(&shown) {
            fenced.push(shown);
        }
    }
    Ok(fenced)
}

/// Tells every agent about the fence in its system prompt, or nothing
/// without one.
fn tell_agents(
    fence: Res<ScopeFence>,
    mut sections: Query<&mut PromptSection, With<ScopeSection>>,
) {
    let text = match fence.dirs.is_empty() {
        true => String::new(),
        false => format!(
            "The user fenced file changes to {}. Change files only there; read anything.",
            fence.line()
        ),
    };
    for mut section in &mut sections {
        section.text.clone_from(&text);
    }
}

/// The `scope: src/parser/` item on the terminal view's status line.
#[cfg(feature = "tui")]
mod status {
    use rig_harness::prelude::*;
    use rig_tui::{AppStatus, Side, StatusItem, StatusSystems, Tone};

    use super::ScopeFence;

    /// After the other agents at work; gone before the turn's spending.
    pub const ITEM: StatusItem = StatusItem::new("sn-scope-fence", Side::Left, 75, 10);

    pub(super) fn add(app: &mut App) {
        app.init_resource::<AppStatus>().add_systems(
            PostUpdate,
            show.in_set(StatusSystems::Show)
                .run_if(resource_changed::<ScopeFence>),
        );
    }

    fn show(fence: Res<ScopeFence>, mut status: ResMut<AppStatus>) {
        let text = match fence.dirs.is_empty() {
            true => String::new(),
            false => format!("scope: {}", fence.dirs.join(" ")),
        };
        status.0.show(ITEM.says(text, Tone::Yellow));
    }
}

#[cfg(feature = "tui")]
pub use status::ITEM;

//! Other coding agents' programs as model providers. [`BridgePlugin`] files
//! the models of each installed harness (Claude Code, Grok Build,
//! Antigravity, Codex) in the catalog under a vendor of its own, so
//! `--model claude-code/<id>` answers through the `claude` program and its
//! sign-in instead of an API key.
//!
//! A harness is run for inference alone: without its own tools,
//! instructions, memory, skills or extensions. It gets Rig's system prompt
//! and reaches Rig's tools through one MCP server the plugin serves on
//! loopback. A tool the model calls is run by Rig like any model's tool
//! call, so approvals, the transcript and replay see it.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! App::new().add_plugins(rig_bridge::BridgePlugin);
//! ```

#[cfg(target_family = "wasm")]
compile_error!("rig-bridge runs other agents' programs: it needs a native target");

#[cfg(not(target_family = "wasm"))]
mod acp;
#[cfg(not(target_family = "wasm"))]
mod antigravity;
#[cfg(not(target_family = "wasm"))]
mod claude;
#[cfg(not(target_family = "wasm"))]
mod codex;
#[cfg(not(target_family = "wasm"))]
mod grok;
#[cfg(not(target_family = "wasm"))]
mod harness;
#[cfg(not(target_family = "wasm"))]
mod mcp;
#[cfg(not(target_family = "wasm"))]
mod session;

#[cfg(not(target_family = "wasm"))]
pub use plugin::BridgeSettings;

/// Adds the models of the harnesses in [`BridgeSettings`].
#[cfg(not(target_family = "wasm"))]
#[derive(Default)]
pub struct BridgePlugin;

#[cfg(not(target_family = "wasm"))]
mod plugin {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Duration;

    use rig_harness::prelude::*;
    use rig_harness::rig_core::catalog::{Catalog, ModelSpec, ReasoningSupport, Transport};
    use rig_harness::rig_core::providers::anthropic;
    use rig_harness::rig_core::providers::registry::{ConnectError, ProviderId};
    use rig_harness::rig_core::serve::ErasedHandler;
    use rig_harness::rig_tools::process::{Process, Spawn};
    use serde::Deserialize;

    use crate::BridgePlugin;
    use crate::acp::Acp;
    use crate::harness::Harness;
    use crate::session::{Bridge, Model};

    /// How long a program has to list its models as the agent starts.
    const LISTING: Duration = Duration::from_secs(5);

    /// `[bridge]` in `settings.toml`.
    #[derive(Resource, Deserialize, Debug, Clone)]
    #[serde(default, deny_unknown_fields)]
    pub struct BridgeSettings {
        /// The Claude Code program: a name on `PATH` or a path.
        pub claude: PathBuf,
        /// The Grok Build program.
        pub grok: PathBuf,
        /// The Antigravity program.
        pub antigravity: PathBuf,
        /// The Codex program.
        pub codex: PathBuf,
        /// How many harness processes may wait for their conversation's
        /// next request. The one unused longest ends first; its
        /// conversation starts a new one when it goes on.
        pub sessions: usize,
    }

    impl Default for BridgeSettings {
        fn default() -> Self {
            Self {
                claude: "claude".into(),
                grok: "grok".into(),
                antigravity: "agy".into(),
                codex: "codex".into(),
                sessions: 8,
            }
        }
    }

    impl Plugin for BridgePlugin {
        fn build(&self, app: &mut App) {
            app.add_setting::<BridgeSettings>("bridge");
        }

        /// Files the models once every plugin's `build` ran, so the process
        /// launcher and tokio runtime the harnesses run with exist.
        fn finish(&self, app: &mut App) {
            let world = app.world_mut();
            let settings = world.get_resource_or_init::<BridgeSettings>().clone();
            let Some(runtime) = world.get_resource::<TokioRuntime>().cloned() else {
                return;
            };
            let processes = world.get_resource_or_init::<Processes>().clone();
            let project = world
                .get_resource::<ProjectRoot>()
                .map(|root| root.0.clone());
            let bridge = Arc::new(Bridge::new(
                runtime.0,
                processes.clone(),
                settings.sessions,
                project,
            ));
            let mut models = world.get_resource_or_init::<Models>();
            let mut file = |vendor: &str, program: PathBuf, harness: Arc<dyn Harness>, ids: Ids| {
                let Ok(vendor) = ProviderId::declare(vendor) else {
                    return;
                };
                let specs: Vec<ModelSpec> = match ids {
                    Ids::Of(other) => models
                        .0
                        .catalog()
                        .iter()
                        .filter(|spec| spec.provider.vendor() == other)
                        .cloned()
                        .map(|mut spec| {
                            spec.provider = vendor;
                            spec
                        })
                        .collect(),
                    Ids::Listed(ids) => ids
                        .iter()
                        .map(|id| {
                            ModelSpec::new(vendor, id)
                                .with_tools(true)
                                .with_reasoning(ReasoningSupport::Unknown { default: None })
                        })
                        .collect(),
                    Ids::Specs(specs) => specs.into_iter().map(|spec| spec(vendor)).collect(),
                };
                for spec in specs {
                    models.0.catalog_mut().insert(spec);
                }
                let provider = Provider {
                    bridge: bridge.clone(),
                    harness,
                    program,
                };
                models.0.provide(vendor, provider);
            };

            // Claude Code takes the ids of Anthropic's API.
            file(
                "claude-code",
                settings.claude,
                Arc::new(crate::claude::Claude),
                Ids::Of(anthropic::PROVIDER_NAME),
            );
            let listed = listing(&processes, &settings.grok, &["models"]);
            file(
                "grok-build",
                settings.grok,
                Arc::new(Acp(Arc::new(crate::grok::Grok))),
                Ids::Listed(crate::grok::models(&listed)),
            );
            let listed = listing(&processes, &settings.antigravity, &["models"]);
            file(
                "antigravity",
                settings.antigravity,
                Arc::new(Acp(Arc::new(crate::antigravity::Antigravity))),
                Ids::Listed(crate::antigravity::models(&listed)),
            );
            let listed = listing(&processes, &settings.codex, &["debug", "models"]);
            if let Some((codex, listed)) = crate::codex::Codex::new(&listed) {
                let specs = listed.into_iter().map(|model| {
                    let spec = move |vendor| {
                        let spec = ModelSpec::new(vendor, &model.id)
                            .with_display_name(model.name)
                            .with_tools(true)
                            .with_reasoning(ReasoningSupport::Unknown { default: None });
                        match model.context_window {
                            Some(tokens) => spec.with_context_window(tokens),
                            None => spec,
                        }
                    };
                    Box::new(spec) as Box<dyn FnOnce(ProviderId) -> ModelSpec>
                });
                file(
                    "codex",
                    settings.codex,
                    Arc::new(codex),
                    Ids::Specs(specs.collect()),
                );
            }
        }
    }

    /// Where a harness's model ids come from.
    enum Ids {
        /// Another vendor's models in the catalog, under the same ids.
        Of(&'static str),
        /// The ids the program listed.
        Listed(Vec<String>),
        /// The models the program listed, as specs of the vendor given.
        Specs(Vec<Box<dyn FnOnce(ProviderId) -> ModelSpec>>),
    }

    /// What `program` prints when run with `args`; nothing when it is not
    /// installed, fails to start or takes too long.
    fn listing(processes: &Processes, program: &Path, args: &[&str]) -> String {
        let Some(program) = installed(program) else {
            return String::new();
        };
        let mut command = processes.launcher().command(program);
        command.args(args);
        Process::start(command, Spawn::default())
            .and_then(|process| process.output(Some(LISTING)))
            .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
            .unwrap_or_default()
    }

    /// How a harness's models are reached.
    struct Provider {
        bridge: Arc<Bridge>,
        harness: Arc<dyn Harness>,
        program: PathBuf,
    }

    impl rig_harness::rig_core::catalog::Provide for Provider {
        /// A model of the harness. A program that is not installed refuses
        /// it, so pickers leave its models out.
        fn handler(
            &self,
            spec: &ModelSpec,
            _catalog: &Catalog,
            _transport: &Transport,
        ) -> Result<ErasedHandler, ConnectError> {
            let program = installed(&self.program).ok_or_else(|| {
                ConnectError::Provider(format!("`{}` is not installed", self.program.display()))
            })?;
            Ok(ErasedHandler::new(Model {
                bridge: self.bridge.clone(),
                harness: self.harness.clone(),
                program,
                vendor: spec.provider.vendor(),
                model: spec.id.to_string(),
            }))
        }
    }

    /// The file `program` names: itself when it is a path, or else the
    /// first of that name on `PATH`.
    fn installed(program: &Path) -> Option<PathBuf> {
        if program.components().count() > 1 {
            return program.is_file().then(|| program.to_path_buf());
        }
        let path = std::env::var_os("PATH")?;
        std::env::split_paths(&path)
            .map(|dir| dir.join(program))
            .find(|candidate| candidate.is_file())
    }
}

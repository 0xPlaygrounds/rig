//! Other coding agents' programs as model providers. [`BridgePlugin`] files
//! the models of each installed harness (Claude Code, Grok Build,
//! Antigravity, Codex, Meta AI Muse, GitHub Copilot CLI) in the catalog under
//! a vendor of its own, so `--model claude-code/<id>` answers through the
//! `claude` program and its sign-in instead of an API key.
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
mod commands;
#[cfg(not(target_family = "wasm"))]
mod copilot;
#[cfg(not(target_family = "wasm"))]
mod grok;
#[cfg(not(target_family = "wasm"))]
mod harness;
#[cfg(not(target_family = "wasm"))]
mod mcp;
#[cfg(not(target_family = "wasm"))]
mod muse;
#[cfg(not(target_family = "wasm"))]
mod rpc;
#[cfg(not(target_family = "wasm"))]
mod session;
#[cfg(all(test, not(target_family = "wasm")))]
mod tests;

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

    use tokio::sync::mpsc;

    use crate::BridgePlugin;
    use crate::acp::Acp;
    use crate::commands::{BridgeHandle, BridgeSignal, HarnessEntry};
    use crate::harness::{Harness, Vendor};
    use crate::session::{Bridge, Model, reap_stale_dirs};

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
        /// The Meta AI Muse program.
        pub muse: PathBuf,
        /// The GitHub Copilot CLI program.
        pub copilot: PathBuf,
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
                muse: "muse".into(),
                copilot: "copilot".into(),
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
            let (bridge, processes, http, entries, signal_tx, signal_rx) = {
                let world = app.world_mut();
                let settings = world.get_resource_or_init::<BridgeSettings>().clone();
                let Some(runtime) = world.get_resource::<TokioRuntime>().cloned() else {
                    return;
                };
                let processes = world.get_resource_or_init::<Processes>().clone();
                let project = world
                    .get_resource::<ProjectRoot>()
                    .map(|root| root.0.clone());
                let scratch_root = RigHome::of(world).root().join("run").join("bridge");
                reap_stale_dirs(&scratch_root);

                let (signal_tx, signal_rx) = mpsc::unbounded_channel::<BridgeSignal>();
                let bridge = Arc::new(
                    Bridge::new(
                        runtime.0,
                        processes.clone(),
                        settings.sessions,
                        project,
                        scratch_root.clone(),
                    )
                    .with_signals(signal_tx.clone()),
                );

                // Discover installed harness model catalogs in parallel so startup
                // stays bounded by a single `LISTING` timeout.
                let (grok_out, agy_out, codex_out, muse_out, copilot_models) =
                    std::thread::scope(|scope| {
                        let grok_h =
                            scope.spawn(|| listing(&processes, &settings.grok, &["models"]));
                        let agy_h =
                            scope.spawn(|| listing(&processes, &settings.antigravity, &["models"]));
                        let codex_h = scope
                            .spawn(|| listing(&processes, &settings.codex, &["debug", "models"]));
                        let muse_h = scope.spawn(|| {
                            listing(&processes, &settings.muse, &["exec", "--list-models"])
                        });
                        let copilot_h = scope.spawn(|| {
                            installed(&settings.copilot)
                                .map(|program| {
                                    crate::copilot::discover_models(
                                        &processes,
                                        &program,
                                        &scratch_root,
                                        LISTING,
                                    )
                                })
                                .unwrap_or_default()
                        });
                        (
                            grok_h.join().unwrap_or_default(),
                            agy_h.join().unwrap_or_default(),
                            codex_h.join().unwrap_or_default(),
                            muse_h.join().unwrap_or_default(),
                            copilot_h.join().unwrap_or_default(),
                        )
                    });

                let mut models = world.get_resource_or_init::<Models>();
                let mut file = |vendor: Vendor,
                                program: &Path,
                                harness: Arc<dyn Harness>,
                                ids: Ids|
                 -> HarnessEntry {
                    let count = ProviderId::declare(vendor.as_str())
                        .map(|pid| {
                            let specs: Vec<ModelSpec> = match ids {
                                Ids::Of(other) => models
                                    .0
                                    .catalog()
                                    .iter()
                                    .filter(|spec| spec.provider.vendor() == other)
                                    .cloned()
                                    .map(|mut spec| {
                                        spec.provider = pid;
                                        spec.pricing = None;
                                        spec
                                    })
                                    .collect(),
                                Ids::Described(listed) => listed
                                    .into_iter()
                                    .map(|(id, name, window)| {
                                        let mut spec = ModelSpec::new(pid, id)
                                            .with_tools(true)
                                            .with_reasoning(ReasoningSupport::Unknown {
                                                default: None,
                                            });
                                        if let Some(name) = name {
                                            spec = spec.with_display_name(name);
                                        }
                                        if let Some(tokens) = window {
                                            spec = spec.with_context_window(tokens);
                                        }
                                        enrich_limits(models.0.catalog(), vendor, spec)
                                    })
                                    .collect(),
                            };
                            let count = specs.len();
                            if count > 0 {
                                for spec in specs {
                                    models.0.catalog_mut().insert(spec);
                                }
                                models.0.provide(
                                    pid,
                                    Provider {
                                        bridge: bridge.clone(),
                                        harness,
                                        program: program.to_path_buf(),
                                        vendor,
                                    },
                                );
                            }
                            count
                        })
                        .unwrap_or(0);
                    HarnessEntry {
                        vendor,
                        program: installed(program),
                        model_count: count,
                    }
                };
                let listed = |ids: Vec<String>| {
                    Ids::Described(ids.into_iter().map(|id| (id, None, None)).collect())
                };

                // Claude Code takes the ids of Anthropic's API.
                let claude_entry = file(
                    Vendor::ClaudeCode,
                    &settings.claude,
                    Arc::new(crate::claude::Claude),
                    Ids::Of(anthropic::PROVIDER_NAME),
                );
                let grok_entry = file(
                    Vendor::GrokBuild,
                    &settings.grok,
                    Arc::new(Acp(Arc::new(crate::grok::Grok))),
                    listed(crate::grok::models(&grok_out)),
                );
                let agy_entry = file(
                    Vendor::Antigravity,
                    &settings.antigravity,
                    Arc::new(Acp(Arc::new(crate::antigravity::Antigravity))),
                    Ids::Described(
                        crate::antigravity::models(&agy_out)
                            .into_iter()
                            .map(|(id, name)| (id, Some(name), None))
                            .collect(),
                    ),
                );
                let codex_entry =
                    if let Some((codex, models_list)) = crate::codex::Codex::new(&codex_out) {
                        file(
                            Vendor::Codex,
                            &settings.codex,
                            Arc::new(codex),
                            Ids::Described(
                                models_list
                                    .into_iter()
                                    .map(|m| (m.id, Some(m.name), m.context_window))
                                    .collect(),
                            ),
                        )
                    } else {
                        HarnessEntry {
                            vendor: Vendor::Codex,
                            program: installed(&settings.codex),
                            model_count: 0,
                        }
                    };
                let muse_entry = file(
                    Vendor::Muse,
                    &settings.muse,
                    Arc::new(crate::muse::Muse),
                    listed(crate::muse::models(&muse_out)),
                );
                let copilot_entry = file(
                    Vendor::Copilot,
                    &settings.copilot,
                    Arc::new(Acp(Arc::new(crate::copilot::Copilot))),
                    Ids::Described(
                        copilot_models
                            .into_iter()
                            .map(|m| (m.id, Some(m.name), None))
                            .collect(),
                    ),
                );
                let http = models.0.transport().shared();
                let entries = vec![
                    claude_entry,
                    grok_entry,
                    agy_entry,
                    codex_entry,
                    muse_entry,
                    copilot_entry,
                ];
                (bridge, processes, http, entries, signal_tx, signal_rx)
            };

            let handle = BridgeHandle {
                bridge,
                processes,
                http,
                installed: entries,
                signals: signal_tx,
            };
            crate::commands::register(app, handle, signal_rx);
        }
    }

    /// Where a harness's model ids come from.
    enum Ids {
        /// Another vendor's models in the catalog, under the same ids.
        Of(&'static str),
        /// `(id, display_name, context_window)` tuples reported by the program.
        Described(Vec<(String, Option<String>, Option<u32>)>),
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
        vendor: Vendor,
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
                vendor: self.vendor,
                model: spec.id.to_string(),
                context_limit: spec.input_limit(),
            }))
        }
    }

    struct Limits {
        context_window: Option<u32>,
        max_input_tokens: Option<u32>,
        max_output_tokens: Option<u32>,
    }

    impl Limits {
        const fn window(tokens: u32) -> Self {
            Self {
                context_window: Some(tokens),
                max_input_tokens: None,
                max_output_tokens: None,
            }
        }
    }

    /// Populates missing context/token limits for bridged models from Rig's
    /// catalog and verified harness defaults so the status meter, `/usage`,
    /// and compaction know the model's context window.
    pub(super) fn enrich_limits(
        catalog: &Catalog,
        vendor: Vendor,
        mut spec: ModelSpec,
    ) -> ModelSpec {
        spec.pricing = None;
        if spec.input_limit().is_some() {
            return spec;
        }
        let Some(limits) = lookup_limits(catalog, vendor, &spec.id) else {
            return spec;
        };
        spec.context_window = spec.context_window.or(limits.context_window);
        spec.max_input_tokens = spec.max_input_tokens.or(limits.max_input_tokens);
        spec.max_output_tokens = spec.max_output_tokens.or(limits.max_output_tokens);
        spec
    }

    fn catalog_limits(catalog: &Catalog, vendor: &str, id: &str) -> Option<Limits> {
        let found = catalog
            .iter()
            .find(|spec| spec.provider.vendor() == vendor && spec.id == id)?;
        (found.context_window.is_some() || found.max_input_tokens.is_some()).then_some(Limits {
            context_window: found.context_window,
            max_input_tokens: found.max_input_tokens,
            max_output_tokens: found.max_output_tokens,
        })
    }

    fn lookup_limits(catalog: &Catalog, vendor: Vendor, id: &str) -> Option<Limits> {
        match vendor {
            Vendor::ClaudeCode | Vendor::Codex => None,
            Vendor::Copilot => catalog_limits(catalog, "copilot", id).or_else(|| {
                matches!(id, "mai-code-2" | "mai-code-2-flash").then_some(Limits::window(200_000))
            }),
            Vendor::GrokBuild => {
                let mut prefix = id;
                loop {
                    if let Some(limits) = catalog_limits(catalog, "xai", prefix) {
                        return Some(limits);
                    }
                    let (head, _) = prefix.rsplit_once('-')?;
                    prefix = head;
                }
            }
            Vendor::Antigravity => {
                let base = ["-xhigh", "-high", "-medium", "-low"]
                    .into_iter()
                    .find_map(|suffix| id.strip_suffix(suffix))
                    .unwrap_or(id);
                ["gcp.gemini", "anthropic", "openai", "openrouter"]
                    .into_iter()
                    .find_map(|source| catalog_limits(catalog, source, base))
                    .or_else(|| catalog_limits(catalog, "openrouter", &format!("openai/{base}")))
                    .or_else(|| {
                        (base.starts_with("gemini-4-neon")
                            || base.starts_with("gemini-pro-orchestrator"))
                        .then_some(Limits::window(1_048_576))
                    })
            }
            Vendor::Muse => ["opencode", "venice", "ollama"]
                .into_iter()
                .find_map(|source| catalog_limits(catalog, source, id))
                .or_else(|| {
                    let base = ["-full-1.0", "-full", "-1.0"]
                        .into_iter()
                        .find_map(|suffix| id.strip_suffix(suffix))
                        .unwrap_or(id);
                    catalog_limits(catalog, "openrouter", &format!("meta/{base}"))
                }),
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

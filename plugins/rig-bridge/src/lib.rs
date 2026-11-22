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

    use tokio::sync::mpsc;

    use crate::BridgePlugin;
    use crate::acp::Acp;
    use crate::commands::{BridgeHandle, BridgeNotice, HarnessEntry};
    use crate::harness::Harness;
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
            let (bridge, processes, entries, notice_rx) = {
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

                let (notice_tx, notice_rx) = mpsc::unbounded_channel::<BridgeNotice>();
                let (warn_tx, mut warn_rx) = mpsc::unbounded_channel::<String>();
                let forward_tx = notice_tx.clone();
                runtime.0.spawn(async move {
                    while let Some(text) = warn_rx.recv().await {
                        if forward_tx
                            .send(BridgeNotice {
                                agent: None,
                                warn: true,
                                text,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                });

                let bridge = Arc::new(
                    Bridge::new(
                        runtime.0,
                        processes.clone(),
                        settings.sessions,
                        project,
                        scratch_root.clone(),
                    )
                    .with_notices(warn_tx),
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
                let mut file = |vendor: &str,
                                program: PathBuf,
                                harness: Arc<dyn Harness>,
                                ids: Ids|
                 -> usize {
                    let Ok(vendor) = ProviderId::declare(vendor) else {
                        return 0;
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
                                spec.pricing = None;
                                spec
                            })
                            .collect(),
                        Ids::Listed(ids) => ids
                            .iter()
                            .map(|id| {
                                let spec = ModelSpec::new(vendor, id)
                                    .with_tools(true)
                                    .with_reasoning(ReasoningSupport::Unknown { default: None });
                                enrich_limits(models.0.catalog(), vendor.vendor(), spec)
                            })
                            .collect(),
                        Ids::Specs(specs) => specs
                            .into_iter()
                            .map(|spec| {
                                enrich_limits(models.0.catalog(), vendor.vendor(), spec(vendor))
                            })
                            .collect(),
                    };
                    let count = specs.len();
                    for spec in specs {
                        models.0.catalog_mut().insert(spec);
                    }
                    let provider = Provider {
                        bridge: bridge.clone(),
                        harness,
                        program,
                    };
                    models.0.provide(vendor, provider);
                    count
                };

                // Claude Code takes the ids of Anthropic's API.
                let claude_count = file(
                    "claude-code",
                    settings.claude.clone(),
                    Arc::new(crate::claude::Claude),
                    Ids::Of(anthropic::PROVIDER_NAME),
                );
                let grok_count = file(
                    "grok-build",
                    settings.grok.clone(),
                    Arc::new(Acp(Arc::new(crate::grok::Grok))),
                    Ids::Listed(crate::grok::models(&grok_out)),
                );
                let agy_count = file(
                    "antigravity",
                    settings.antigravity.clone(),
                    Arc::new(Acp(Arc::new(crate::antigravity::Antigravity))),
                    Ids::Listed(crate::antigravity::models(&agy_out)),
                );
                let codex_count =
                    if let Some((codex, listed)) = crate::codex::Codex::new(&codex_out) {
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
                            settings.codex.clone(),
                            Arc::new(codex),
                            Ids::Specs(specs.collect()),
                        )
                    } else {
                        0
                    };
                let muse_count = file(
                    "muse",
                    settings.muse.clone(),
                    Arc::new(crate::muse::Muse),
                    Ids::Listed(crate::muse::models(&muse_out)),
                );
                let copilot_count = if !copilot_models.is_empty() {
                    let specs = copilot_models.into_iter().map(|model| {
                        let spec = move |vendor| {
                            ModelSpec::new(vendor, &model.id)
                                .with_display_name(model.name)
                                .with_tools(true)
                                .with_reasoning(ReasoningSupport::Unknown { default: None })
                        };
                        Box::new(spec) as Box<dyn FnOnce(ProviderId) -> ModelSpec>
                    });
                    file(
                        "copilot",
                        settings.copilot.clone(),
                        Arc::new(Acp(Arc::new(crate::copilot::Copilot))),
                        Ids::Specs(specs.collect()),
                    )
                } else {
                    0
                };

                let entries = vec![
                    HarnessEntry {
                        vendor: "claude-code",
                        label: "Claude Code",
                        program: installed(&settings.claude),
                        model_count: claude_count,
                    },
                    HarnessEntry {
                        vendor: "grok-build",
                        label: "Grok Build",
                        program: installed(&settings.grok),
                        model_count: grok_count,
                    },
                    HarnessEntry {
                        vendor: "antigravity",
                        label: "Antigravity",
                        program: installed(&settings.antigravity),
                        model_count: agy_count,
                    },
                    HarnessEntry {
                        vendor: "codex",
                        label: "Codex",
                        program: installed(&settings.codex),
                        model_count: codex_count,
                    },
                    HarnessEntry {
                        vendor: "muse",
                        label: "Meta AI Muse",
                        program: installed(&settings.muse),
                        model_count: muse_count,
                    },
                    HarnessEntry {
                        vendor: "copilot",
                        label: "GitHub Copilot",
                        program: installed(&settings.copilot),
                        model_count: copilot_count,
                    },
                ];
                (bridge, processes, entries, notice_rx)
            };

            let handle = BridgeHandle {
                bridge,
                processes,
                installed: entries,
            };
            crate::commands::register(app, handle, notice_rx);
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
                context_limit: spec.input_limit(),
            }))
        }
    }

    /// Context and output token limits copied from a matching catalog row or
    /// verified harness default.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Limits {
        context_window: Option<u32>,
        max_input_tokens: Option<u32>,
        max_output_tokens: Option<u32>,
    }

    /// Fills in missing context/input/output token limits on `spec` from Rig's
    /// catalog (and verified harness defaults for models not yet in the
    /// built-in catalog) so the status meter, `/usage`, and compaction know
    /// the model's context window.
    fn enrich_limits(catalog: &Catalog, vendor: &str, mut spec: ModelSpec) -> ModelSpec {
        spec.pricing = None;
        if spec.input_limit().is_some() {
            return spec;
        }
        let Some(limits) = lookup_limits(catalog, vendor, &spec.id) else {
            return spec;
        };
        if spec.context_window.is_none() {
            spec.context_window = limits.context_window;
        }
        if spec.max_input_tokens.is_none() {
            spec.max_input_tokens = limits.max_input_tokens;
        }
        if spec.max_output_tokens.is_none() {
            spec.max_output_tokens = limits.max_output_tokens;
        }
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

    fn lookup_limits(catalog: &Catalog, vendor: &str, id: &str) -> Option<Limits> {
        match vendor {
            "copilot" => {
                if let Some(limits) = catalog_limits(catalog, "copilot", id) {
                    return Some(limits);
                }
                let dashed = id.replace('.', "-");
                for source in ["openai", "anthropic", "gcp.gemini", "xai", "openrouter"] {
                    if let Some(limits) = catalog_limits(catalog, source, id)
                        .or_else(|| catalog_limits(catalog, source, &dashed))
                    {
                        return Some(limits);
                    }
                }
                if matches!(id, "mai-code-2" | "mai-code-2-flash") {
                    return Some(Limits {
                        context_window: Some(200_000),
                        max_input_tokens: None,
                        max_output_tokens: None,
                    });
                }
                None
            }
            "grok-build" => {
                if let Some(limits) = catalog_limits(catalog, "xai", id) {
                    return Some(limits);
                }
                let mut prefix = id;
                while let Some((head, _)) = prefix.rsplit_once('-') {
                    if let Some(limits) = catalog_limits(catalog, "xai", head) {
                        return Some(limits);
                    }
                    prefix = head;
                }
                None
            }
            "antigravity" => {
                let base = id
                    .strip_suffix("-xhigh")
                    .or_else(|| id.strip_suffix("-high"))
                    .or_else(|| id.strip_suffix("-medium"))
                    .or_else(|| id.strip_suffix("-low"))
                    .unwrap_or(id);
                for source in ["gcp.gemini", "anthropic", "openai", "openrouter"] {
                    if let Some(limits) = catalog_limits(catalog, source, base) {
                        return Some(limits);
                    }
                }
                if let Some(limits) =
                    catalog_limits(catalog, "openrouter", &format!("openai/{base}"))
                {
                    return Some(limits);
                }
                if base.starts_with("gemini-4-neon") || base.starts_with("gemini-pro-orchestrator")
                {
                    return Some(Limits {
                        context_window: Some(1_048_576),
                        max_input_tokens: None,
                        max_output_tokens: None,
                    });
                }
                None
            }
            "muse" => {
                for source in ["opencode", "venice", "ollama"] {
                    if let Some(limits) = catalog_limits(catalog, source, id) {
                        return Some(limits);
                    }
                }
                let base = id
                    .strip_suffix("-full-1.0")
                    .or_else(|| id.strip_suffix("-full"))
                    .or_else(|| id.strip_suffix("-1.0"))
                    .unwrap_or(id);
                catalog_limits(catalog, "openrouter", &format!("meta/{base}"))
            }
            _ => None,
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

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn enrich_limits_populates_context_windows_across_bridge_vendors() {
            let catalog = Catalog::builtin();
            let check = |vendor: &str, id: &str| {
                let Ok(pid) = ProviderId::declare(vendor) else {
                    return None;
                };
                enrich_limits(catalog, vendor, ModelSpec::new(pid, id)).input_limit()
            };

            assert_eq!(check("grok-build", "grok-4.7"), Some(500_000));
            assert_eq!(check("grok-build", "grok-4.7-build-fast"), Some(500_000));
            assert_eq!(check("antigravity", "gemini-4-neon-high"), Some(1_048_576));
            assert_eq!(
                check("antigravity", "gemini-4-carbon-medium"),
                Some(1_048_576)
            );
            assert_eq!(
                check("antigravity", "gemini-pro-orchestrator-neon"),
                Some(1_048_576)
            );
            assert_eq!(
                check("antigravity", "claude-opus-5-5-medium"),
                Some(1_000_000)
            );
            assert_eq!(check("antigravity", "gpt-oss-120b-medium"), Some(131_072));
            assert_eq!(check("muse", "muse-wonder-1.0"), Some(1_048_576));
            assert_eq!(check("muse", "muse-wonder-full-1.0"), Some(1_048_576));
            assert_eq!(check("copilot", "gpt-6.1-terra"), Some(922_000));
            assert_eq!(check("copilot", "claude-sonnet-5.5"), Some(1_000_000));
            assert_eq!(check("copilot", "mai-code-2"), Some(200_000));
            assert_eq!(check("copilot", "auto"), None);
        }
    }
}

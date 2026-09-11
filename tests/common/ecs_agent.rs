//! Native ECS execution for provider cassette comparisons.
//!
//! This support owns a World and drives its public schedule. It does not use
//! rig-agent's builder, runner, policy, or recorded effect answers.

#[path = "ecs_agent/tests.rs"]
mod tests;

use std::{sync::Arc, time::Duration};

use bevy_app::{App, Update};
use bevy_ecs::prelude::*;
use rig_core::{
    completion::CompletionModel,
    effect::{EffectKind, HandlerDescriptor},
    serve::{
        Dispatch, Reply, Serve, ServingPolicy,
        adapters::{CompletionAdapter, ToolAdapter},
    },
    tool::Tool,
};
use rig_ecs::{
    agent::{DefaultMaxTurns, Failed, Failure, RunResult, Settled},
    bus::{Handlers, Recording, run_to_quiescence},
    commands::{Agent, Prompt, install},
    lifecycle::grant_tool,
};
use rig_effect_log::EffectLogRecorder;

/// Enter the supplied runtime on each handler poll, without spawning detached
/// work. The ECS task still owns the future, so dropping it drops transport IO.
pub(crate) struct RuntimeHandler<S> {
    pub(crate) inner: Arc<S>,
    pub(crate) runtime: tokio::runtime::Handle,
}

impl<S: Serve + 'static> Serve for RuntimeHandler<S> {
    type Family = S::Family;

    fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let future = self.inner.serve(kind, dispatch);
        futures::pin_mut!(future);
        let reply = std::future::poll_fn(|context| {
            let _entered = self.runtime.enter();
            future.as_mut().poll(context)
        })
        .await;
        match reply {
            Reply::Outcome(outcome) => Reply::Outcome(outcome),
            Reply::Stream(mut stream) => {
                let runtime = self.runtime.clone();
                Reply::Stream(Box::pin(futures::stream::poll_fn(move |context| {
                    let _entered = runtime.enter();
                    stream.as_mut().poll_next(context)
                })))
            }
        }
    }
}

/// Test-owned application. Callers can install ordinary systems and inspect
/// the complete world; there is no second configuration or policy engine.
pub(crate) struct EcsAgent {
    pub app: App,
    pub agent: Entity,
    tool_count: u64,
    recorder: EffectLogRecorder,
    golden_identity: bool,
    pub declared_policies: Vec<String>,
    /// Whether the producer declares the bus policy as agent-owned metadata.
    pub declare_bus_policy: bool,
}

impl EcsAgent {
    pub fn new(model: impl CompletionModel + 'static, preamble: &str, turns: usize) -> Self {
        Self::configured(model, preamble, turns, false, true, |_| {})
    }

    pub fn for_golden(
        model: impl CompletionModel + 'static,
        preamble: &str,
        keep_events: bool,
    ) -> Self {
        Self::for_golden_with_setup(model, preamble, keep_events, |_| {})
    }

    /// Register application-owned handlers before the model, preserving the
    /// producer's handler registration order in the complete recorder header.
    pub fn for_golden_with_setup(
        model: impl CompletionModel + 'static,
        preamble: &str,
        keep_events: bool,
        setup: impl FnOnce(&mut World),
    ) -> Self {
        Self::configured(model, preamble, 1, true, keep_events, setup)
    }

    fn configured(
        model: impl CompletionModel + 'static,
        preamble: &str,
        turns: usize,
        golden_identity: bool,
        keep_events: bool,
        setup: impl FnOnce(&mut World),
    ) -> Self {
        let mut app = App::new();
        install(app.world_mut(), ServingPolicy::default()).expect("fresh runtime");
        app.add_systems(Update, run_to_quiescence);
        app.finish();
        app.cleanup();
        let recorder = if keep_events {
            EffectLogRecorder::keeping_stream_events()
        } else {
            EffectLogRecorder::new()
        };
        Recording::install(app.world_mut(), recorder.clone());
        setup(app.world_mut());
        let model = Handlers::register_in(
            app.world_mut(),
            if golden_identity {
                "golden/model:default"
            } else {
                "parity/model"
            },
            RuntimeHandler {
                inner: Arc::new(CompletionAdapter::new(
                    if golden_identity { "default" } else { "parity" },
                    model,
                )),
                runtime: tokio::runtime::Handle::current(),
            },
        )
        .expect("fresh model key");
        let agent = Agent::new(model)
            .owner(if golden_identity { "golden" } else { "parity" })
            .preamble(preamble)
            .max_turns(turns)
            .spawn(app.world_mut())
            .expect("registered model");
        if golden_identity {
            // Legacy corpus identity distinguishes an absent declared default
            // from the runtime's effective one-turn budget.
            app.world_mut()
                .entity_mut(agent)
                .insert(DefaultMaxTurns(None));
        }
        Self {
            app,
            agent,
            tool_count: 0,
            recorder,
            golden_identity,
            declared_policies: vec![],
            declare_bus_policy: true,
        }
    }

    pub fn tool<T: Tool + 'static>(&mut self, tool: T) {
        let order = self.tool_count;
        let handler = Handlers::register_in(
            self.app.world_mut(),
            if self.golden_identity {
                format!("golden/tool:{}#{order}", T::NAME)
            } else {
                format!("parity/tool#{order}")
            },
            RuntimeHandler {
                inner: Arc::new(ToolAdapter::new(tool)),
                runtime: tokio::runtime::Handle::current(),
            },
        )
        .expect("fresh tool key");
        grant_tool(self.app.world_mut(), self.agent, handler).expect("live agent and tool");
        self.tool_count += 1;
    }

    /// Execute with ordinary runtime defaults, yielding to transport IO between
    /// updates. The deadline is a failing test guard, never a successful ending.
    pub async fn prompt(&mut self, prompt: &str, streamed: bool) -> String {
        self.prompt_with_max_turns(prompt, streamed, None).await
    }

    /// Apply an optional run override without changing the agent's defaults.
    pub async fn prompt_with_max_turns(
        &mut self,
        prompt: &str,
        streamed: bool,
        max_turns: Option<usize>,
    ) -> String {
        let mut request = Prompt::new(self.agent, prompt);
        if streamed {
            request = request.streaming();
        }
        if let Some(limit) = max_turns {
            request = request.max_turns(limit);
        }
        let run = request.spawn(self.app.world_mut()).expect("live agent");
        self.wait_for_success(run).await
    }

    /// Drive a caller-configured native run to the same success boundary.
    pub async fn wait_for_success(&mut self, run: Entity) -> String {
        self.wait_for_outcome(run)
            .await
            .expect("native ECS run must succeed")
    }

    pub fn effect_log(&self) -> rig_effect_log::EffectLog {
        self.recorder.log()
    }

    /// The legacy response boundary includes the memory append. Native Settled
    /// exposes the answer before that effect necessarily completes, so this
    /// consumer also awaits the actual append acknowledgement, without changing
    /// the runtime's phase or manufacturing an outcome.
    fn memory_append_complete(&mut self, run: Entity) -> bool {
        use rig_core::effect::{MemoryOp, MemoryOutcome, Outcome};
        use rig_ecs::{
            agent::Remembering,
            bus::{EffectOutcome, PendingEffect},
        };
        if self.app.world().get::<Remembering>(run).is_none() {
            return true;
        }
        let world = self.app.world_mut();
        let mut effects = world.query::<(&ChildOf, &PendingEffect, Option<&EffectOutcome>)>();
        let appends: Vec<_> = effects
            .iter(world)
            .filter(|(parent, pending, _)| {
                parent.parent() == run
                    && matches!(
                        pending.kind,
                        EffectKind::Memory {
                            op: MemoryOp::Append { .. }
                        }
                    )
            })
            .collect();
        assert!(appends.len() <= 1, "one append for a successful run");
        let Some((_, _, Some(outcome))) = appends.first() else {
            return false;
        };
        assert!(
            matches!(
                outcome.0,
                Ok(Outcome::Memory(MemoryOutcome::Appended)) | Err(_)
            ),
            "memory append must finish with its acknowledgement or recorded error: {:?}",
            outcome.0
        );
        true
    }

    pub async fn wait_for_outcome(&mut self, run: Entity) -> Result<String, Failure> {
        if self.golden_identity {
            let bus = self.app.world().resource::<rig_ecs::bus::Policy>().0;
            rig_ecs::replay::stamp_header(
                self.app.world_mut(),
                self.agent,
                &self.recorder,
                self.declare_bus_policy.then_some(bus),
                self.declared_policies.clone(),
            );
            rig_ecs::replay::stamp_run(self.app.world_mut(), run, &self.recorder);
        }
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                self.app.update();
                if let Some(failed) = self.app.world().get::<Failed>(run) {
                    return Err(failed.0.clone());
                }
                if self.app.world().get::<Settled>(run).is_some()
                    && self.memory_append_complete(run)
                {
                    // A successful fold alone does not rule out a later error
                    // item. Original success collectors propagate every item
                    // error through EOF; Collect publishes those errors even
                    // without event recording, after the fold's first outcome.
                    assert!(
                        self.app
                            .world_mut()
                            .query::<&rig_ecs::bus::Streamed>()
                            .iter(self.app.world())
                            .all(|stream| stream.errors.is_empty()),
                        "successful prompt must not contain stream error items"
                    );
                    return Ok(self
                        .app
                        .world()
                        .get::<RunResult>(run)
                        .expect("settled run has a result")
                        .0
                        .clone());
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("native ECS run exceeded its deadline")
    }
}

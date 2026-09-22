//! Native ECS execution for provider cassette comparisons.
//!
//! This support owns a World and drives its public plugins. It does not use
//! rig-agent's builder, runner, policy, or recorded effect answers.

#[cfg(test)]
#[path = "ecs_agent/tests.rs"]
mod tests;

use std::{
    sync::{Arc, LazyLock},
    time::Duration,
};

use bevy_app::App;
use bevy_ecs::prelude::*;
use rig_cassette::effect_log::EffectLogRecorder;
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
    agent::{
        DefaultMaxTurns, Failed, Failure, Grant, MaxTurns, Owner, Preamble, RunResult, Settled,
        UsesModel,
    },
    bus::{Handlers, Recording},
    systems::RunCommands,
};

/// Transport runtime driven independently of the test's `app.update()` loop.
///
/// The static owns the runtime for the process lifetime, so handles remain valid
/// across tests. Building it requires neither entering nor blocking a runtime,
/// including when first called from a current-thread Tokio test.
/// The position of `entity` among its parent's children: the sibling
/// order every ordered read of the graph uses.
pub fn sibling_index(world: &World, entity: Entity) -> Option<usize> {
    let parent = world.get::<ChildOf>(entity)?.parent();
    world
        .get::<Children>(parent)?
        .iter()
        .position(|child| child == entity)
}

pub fn io_runtime() -> tokio::runtime::Handle {
    static IO: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("ecs-harness-io")
            .enable_all()
            .build()
            .expect("the harness IO runtime starts")
    });
    IO.handle().clone()
}

/// Enter the supplied runtime ([`io_runtime`]) on each future and stream poll.
/// ECS retains ownership of both: cancelling the effect drops its request or
/// stream instead of leaving a detached request task on the transport runtime.
pub struct RuntimeHandler<S> {
    /// Service whose future is polled with the transport runtime entered.
    pub inner: Arc<S>,
    /// Tokio runtime handle used while polling the service future and stream.
    pub runtime: tokio::runtime::Handle,
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
pub struct EcsAgent {
    /// Bevy application containing the agent, handlers, and run entities.
    pub app: App,
    /// Agent entity configured by this harness.
    pub agent: Entity,
    tool_count: u64,
    recorder: EffectLogRecorder,
    golden_identity: bool,
}

impl EcsAgent {
    /// Create a native agent with the supplied model, preamble, and turn budget.
    pub fn new(model: impl CompletionModel + 'static, preamble: &str, turns: usize) -> Self {
        Self::configured(model, preamble, turns, false, true, |_| {})
    }

    /// Create a one-turn parity agent, optionally retaining recorded stream events.
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
        app.add_plugins(rig_ecs::RigPlugin::with_policy(ServingPolicy::default()));
        app.finish();
        app.cleanup();
        let recorder = if keep_events {
            EffectLogRecorder::keeping_stream_events()
        } else {
            EffectLogRecorder::new()
        };
        Recording::install(app.world_mut(), recorder.clone());
        crate::goldens::attach_world_recorder(&recorder);
        setup(app.world_mut());
        let model = Handlers::with(app.world_mut(), |handlers| {
            handlers.register(
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
                    runtime: io_runtime(),
                },
            )
        })
        .expect("bus installed")
        .expect("fresh model key");
        let agent = app
            .world_mut()
            .spawn((
                Owner(if golden_identity { "golden" } else { "parity" }.into()),
                rig_ecs::agent::PolicyVersion("ecs-native/v1".into()),
                Preamble(Some(preamble.into())),
                DefaultMaxTurns(if golden_identity { None } else { Some(turns) }),
                MaxTurns(turns),
                UsesModel(model),
            ))
            .id();
        Self {
            app,
            agent,
            tool_count: 0,
            recorder,
            golden_identity,
        }
    }

    /// Register a tool in declaration order and attach it to the agent.
    pub fn tool<T: Tool + 'static>(&mut self, tool: T) {
        let order = self.tool_count;
        let handler = Handlers::with(self.app.world_mut(), |handlers| {
            handlers.register(
                if self.golden_identity {
                    format!("golden/tool:{}#{order}", T::NAME)
                } else {
                    format!("parity/tool#{order}")
                },
                RuntimeHandler {
                    inner: Arc::new(ToolAdapter::new(tool)),
                    runtime: io_runtime(),
                },
            )
        })
        .expect("bus installed")
        .expect("fresh tool key");
        self.app
            .world_mut()
            .spawn((Grant(handler), ChildOf(self.agent)));
        self.tool_count += 1;
    }

    /// Execute with ordinary plugin defaults. Transport IO progresses independently
    /// of updates; the deadline is a failing test guard, never a successful ending.
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
        let run = self
            .app
            .world_mut()
            .spawn_run(self.agent, &[], prompt, streamed, max_turns);
        self.wait_for_success(run).await
    }

    /// Drive a caller-configured native run to the same success boundary.
    pub async fn wait_for_success(&mut self, run: Entity) -> String {
        self.wait_for_outcome(run)
            .await
            .expect("native ECS run must succeed")
    }

    /// Return the recorder's current effect-log snapshot.
    pub fn effect_log(&self) -> rig_cassette::effect_log::EffectLog {
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

    /// Drive a run until settlement or failure, panicking after the 30-second deadline.
    pub async fn wait_for_outcome(&mut self, run: Entity) -> Result<String, Failure> {
        rig_cassette::ecs::identity::stamp_run(self.app.world_mut(), run, &self.recorder)
            .expect("the run stamps its program identity");
        crate::goldens::capture_world_program(self.app.world_mut(), run, &self.recorder.log());
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

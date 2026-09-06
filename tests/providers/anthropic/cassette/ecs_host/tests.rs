//! Synthetic host acknowledgements held open independently of provider fixtures.
use super::*;
use rig::{
    effect::{EffectKind, HandlerDescriptor},
    serve::{OutcomeSink, Serve},
    test_utils::{MockCompletionModel, MockTurn},
};
use std::sync::atomic::{AtomicUsize, Ordering};
struct GatedNote {
    started: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Semaphore>,
}
impl Serve for GatedNote {
    type Family = <NoteTaker as Serve>::Family;
    fn descriptor(&self) -> HandlerDescriptor {
        NoteTaker.descriptor()
    }
    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        self.started.notify_one();
        self.release.acquire().await.expect("gate open").forget();
        NoteTaker.serve(kind, sink).await;
    }
}
#[derive(Resource)]
struct ModelDispatches(Arc<AtomicUsize>);
fn model_issued(
    event: On<Add, rig_ecs::bus::Issued>,
    effects: Query<&PendingEffect>,
    seen: Res<ModelDispatches>,
) {
    if matches!(
        effects
            .get(event.event().entity)
            .expect("issued effect")
            .kind,
        EffectKind::Completion { .. }
    ) {
        seen.0.fetch_add(1, Ordering::SeqCst);
    }
}
#[tokio::test]
async fn host_acknowledgements_precede_continuation() {
    for hooks in [
        Hooks::AtStart,
        Hooks::AtCompletionCall,
        Hooks::AtOutcome,
        Hooks::AtSettled,
        Hooks::StartAndSettled,
        Hooks::Twice,
    ] {
        let host = Host {
            with_tool: matches!(hooks, Hooks::AtOutcome),
            ..PLAIN
        };
        let turns = if host.with_tool {
            vec![
                MockTurn::tool_call("add-call", "add", serde_json::json!({"x":17,"y":25})),
                MockTurn::text("42"),
            ]
        } else {
            vec![MockTurn::text("ready")]
        };
        let mut ecs = agent(MockCompletionModel::from_turns(turns), &host, hooks);
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let count = Arc::new(AtomicUsize::new(0));
        ecs.app
            .insert_resource(ModelDispatches(count.clone()))
            .add_observer(model_issued);
        Handlers::with(ecs.app.world_mut(), |handlers| {
            handlers.register(
                NOTE_KEY,
                RuntimeHandler {
                    inner: Arc::new(GatedNote {
                        started: started.clone(),
                        release: release.clone(),
                    }),
                    runtime: tokio::runtime::Handle::current(),
                },
            )
        })
        .expect("bus")
        .expect("replace same note family");
        tokio::time::timeout(std::time::Duration::from_secs(5),async{
 let mut response=Box::pin(run_prompt(&mut ecs,&host));
 tokio::select!{output=&mut response=>panic!("{hooks:?} returned before note acknowledgement: {output}"),()=started.notified()=>{}}
 for _ in 0..64 {assert!(futures::poll!(&mut response).is_pending(),"{hooks:?} returned before note acknowledgement");tokio::task::yield_now().await;}
 let expected=usize::from(matches!(hooks,Hooks::AtOutcome|Hooks::AtSettled));assert_eq!(count.load(Ordering::SeqCst),expected,"{hooks:?} cannot issue the next model while acknowledgement is held");
 release.add_permits(2);let output=response.await;assert_eq!(output,if host.with_tool{"42"}else{"ready"});
 }).await.expect("gated host completes");
    }
}

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering::SeqCst};

use tokio::sync::Notify;

use crate::agent::AgentBuilder;
use crate::test_utils::{MockCompletionModel, MockFrame, MockRuntime, MockScript, MockTurn};
use rig_core::driver::{Exchange, Opening, Transport};

/// Holds the first request open until the test releases it.
#[derive(Clone)]
struct HoldFirst {
    inner: MockRuntime,
    requests: Arc<AtomicU32>,
    held: Arc<Notify>,
    release: Arc<Notify>,
}

impl Transport<MockScript> for HoldFirst {
    fn send(
        &self,
        payload: crate::completion::CompletionRequest,
        exchange: Exchange,
    ) -> Opening<MockFrame> {
        let this = self.clone();
        Opening::new(async move {
            if this.requests.fetch_add(1, SeqCst) == 0 {
                this.held.notify_one();
                this.release.notified().await;
            }
            Transport::<MockScript>::send(&this.inner, payload, exchange).await
        })
    }
}

/// A run that ends while another run of the same agent is still waiting on
/// its model settles only its own work: it returns once the in-flight count
/// stops falling, without waiting for the other run's live dispatch.
#[tokio::test]
async fn a_finished_run_does_not_wait_for_another_runs_live_dispatch() {
    let mock = MockCompletionModel::from_turns([MockTurn::text("one"), MockTurn::text("two")]);
    let held = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let model = rig_core::Model::new(
        mock.wire,
        HoldFirst {
            inner: mock.transport,
            requests: Arc::new(AtomicU32::new(0)),
            held: held.clone(),
            release: release.clone(),
        },
    );
    let agent = AgentBuilder::new(model).build();

    let mut waiting = std::pin::pin!(agent.prompt("waits").run());
    tokio::select! {
        biased;
        _ = &mut waiting => panic!("the held run cannot finish before its release"),
        () = held.notified() => {}
    }

    let finished = agent.prompt("finishes").run().await.expect("the free run");
    release.notify_one();
    let waited = waiting.await.expect("the held run");

    let outputs = std::collections::HashSet::from([finished.output(), waited.output()]);
    assert_eq!(
        outputs,
        std::collections::HashSet::from(["one".to_string(), "two".to_string()])
    );
}

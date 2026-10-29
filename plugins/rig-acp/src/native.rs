use rig_harness::prelude::*;

mod mcp;
mod server;
mod worker;

/// Offers `rig acp`, negotiating ACP v2 or v1 over stdin and stdout.
#[derive(Default)]
pub struct AcpPlugin;

impl Plugin for AcpPlugin {
    fn build(&self, app: &mut App) {
        app.add_front(Front::new("acp").sessionless().on_command(
            "acp",
            "",
            "Serve ACP over stdio",
        ))
        .add_front(Front::new("acp-worker").on_command(
            "acp-worker",
            "",
            "Run an ACP session worker",
        ));
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front("acp-worker") {
            worker::install(app);
        } else if app.runs_front("acp") {
            app.add_systems(Startup, start)
                .add_systems(First, finished)
                .add_systems(
                    Last,
                    shutdown
                        .in_set(bevy_app::OnAppExitSystems)
                        .run_if(on_message::<bevy_app::AppExit>),
                );
        }
    }
}

#[derive(Resource)]
struct Server {
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    task: Option<tokio::task::JoinHandle<()>>,
    done: async_channel::Receiver<bool>,
}

fn start(runtime: Res<TokioRuntime>, home: Res<RigHome>, wake: Res<Wake>, mut commands: Commands) {
    let home = home.0.clone();
    let wake = wake.clone();
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let (done, receive) = async_channel::bounded(1);
    let task = runtime.0.spawn(async move {
        let result = server::serve(home, stopped).await;
        if let Err(error) = &result {
            eprintln!("ACP: {error}");
        }
        let _ = done.send(result.is_ok()).await;
        wake.wake();
    });
    commands.insert_resource(Server {
        stop: Some(stop),
        task: Some(task),
        done: receive,
    });
}

fn finished(server: Option<Res<Server>>, mut commands: Commands) {
    if let Some(server) = server
        && let Ok(success) = server.done.try_recv()
    {
        commands.trigger(Exit(if success {
            ExitReason::Quit
        } else {
            ExitReason::Failed(1)
        }));
    }
}

fn shutdown(server: Option<ResMut<Server>>, runtime: Res<TokioRuntime>) {
    let Some(mut server) = server else { return };
    if let Some(stop) = server.stop.take() {
        let _ = stop.send(());
    }
    if let Some(task) = server.task.take() {
        let _ = runtime.0.block_on(task);
    }
}

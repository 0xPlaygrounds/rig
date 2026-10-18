//! The worker's one connection to `rigd`: a [`rig_client::Client`] with the
//! worker's token, no autostart and no reconnect. Its `worker/command`
//! notifications are queued for the app, which the client wakes; what the
//! worker tells `rigd` without waiting (events, applied, quiescent, plugin
//! replies) goes through one sender thread, in order.

use std::path::Path;
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender, unbounded};
use rig_client::{Autostart, Client, ClientError, Options, Target};
use rig_ecs::tasks::Wake;
use rig_protocol::methods::{WORKER_COMMAND, WorkerCommand};
use rig_protocol::{Endpoint, Method, RpcError, codes};

/// What `rigd` sent unasked, in the order it came.
pub(crate) enum Inbound {
    Command(WorkerCommand),
    /// The connection is gone: `rigd` is.
    Closed,
}

/// The connection, shared by the app and the threads tools call from.
pub(crate) struct Link {
    client: Arc<Client>,
    outgoing: Sender<(&'static str, serde_json::Value)>,
}

impl Link {
    /// Connects to the `rigd` at `endpoint` with `token`; what it sends
    /// unasked goes to the returned receiver, each followed by a `wake`.
    pub(crate) fn connect(
        endpoint: &Endpoint,
        token: &str,
        wake: Wake,
    ) -> Result<(Arc<Link>, Receiver<Inbound>), ClientError> {
        let (inbound, received) = unbounded();
        let commands = inbound.clone();
        let woken = wake.clone();
        let options = Options {
            token: Some(token.to_owned()),
            autostart: Autostart::Off,
            reconnect: false,
            on_notification: Some(Arc::new(move |note: rig_protocol::Notification| {
                if note.method != WORKER_COMMAND {
                    return;
                }
                match serde_json::from_value(note.params) {
                    Ok(command) => drop(commands.send(Inbound::Command(command))),
                    Err(e) => bevy_log::warn!("rigd sent a command the worker cannot read: {e}"),
                }
            })),
            wake: Some(Arc::new(move |()| woken.wake())),
            ..Options::default()
        };
        // `run/rig.sock` of the rig home.
        let rig_home = endpoint
            .path
            .parent()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let client = Arc::new(Client::connect_with(
            &Target::Local { rig_home },
            "worker",
            options,
        )?);
        let (outgoing, queue) = unbounded::<(&'static str, serde_json::Value)>();
        let sender = client.clone();
        std::thread::Builder::new()
            .name("rig worker channel".into())
            .spawn(move || {
                for (method, params) in queue {
                    match sender.call_raw(method, params) {
                        Ok(_) => {}
                        Err(ClientError::Rpc(e)) => bevy_log::warn!("rigd refused {method}: {e}"),
                        Err(e) => {
                            bevy_log::error!("rigd is gone: {e}");
                            inbound.send(Inbound::Closed).ok();
                            wake.wake();
                            return;
                        }
                    }
                }
            })?;
        Ok((Arc::new(Link { client, outgoing }), received))
    }

    /// The client itself, for a plugin that calls `rigd` as this session.
    pub(crate) fn client(&self) -> &Arc<Client> {
        &self.client
    }

    /// Calls `M` and waits for its answer.
    pub(crate) fn call<M: Method>(&self, params: &M::Params) -> Result<M::Result, RpcError> {
        self.client.call::<M>(params).map_err(|e| match e {
            ClientError::Rpc(e) => e,
            other => RpcError::new(codes::INTERNAL, other.to_string()),
        })
    }

    /// Calls `M` without waiting, after what was sent before it: a failure
    /// is logged, and one that means `rigd` is gone closes the channel.
    pub(crate) fn send<M: Method>(&self, params: &M::Params) {
        match serde_json::to_value(params) {
            Ok(params) => drop(self.outgoing.send((M::NAME, params))),
            Err(e) => bevy_log::warn!("could not send {}: {e}", M::NAME),
        }
    }
}

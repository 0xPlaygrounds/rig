use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

use rig_harness::harness_protocol::Home;
use rig_harness::rig_core::catalog::ModelStatus;
use rig_harness::services::RigHome;

use std::future::Future;
use std::sync::{Arc, Mutex};

use rig_harness::rig_core::http_client::{
    self, Bytes, DynHttpClient, HeaderValue, HttpClientExt, LazyBody, MultipartForm, Request,
    Response, StreamingResponse,
};
use rig_harness::rig_core::wasm_compat::WasmCompatSend;

use super::*;

#[test]
fn plugin_registers_separate_vendor_without_changing_anthropic() {
    let mut app = App::new();
    app.init_resource::<Models>();
    let original: Vec<_> = app
        .world()
        .resource::<Models>()
        .0
        .catalog()
        .iter()
        .filter(|spec| spec.provider.vendor() == "anthropic")
        .map(|spec| spec.reference())
        .collect();
    app.add_plugins(ClaudeCloakPlugin);
    let models = app.world().resource::<Models>();
    let after: Vec<_> = models
        .0
        .catalog()
        .iter()
        .filter(|spec| spec.provider.vendor() == "anthropic")
        .map(|spec| spec.reference())
        .collect();
    assert!(!original.is_empty());
    assert_eq!(original, after);
    for reference in original {
        let separate = reference.replacen("anthropic/", "claude-cloak/", 1);
        assert!(models.0.catalog().resolve(&separate).is_ok());
        let retired = reference.replacen("anthropic/", "anthropic-oauth/", 1);
        assert!(models.0.catalog().resolve(&retired).is_err());
    }
}

#[test]
fn the_models_are_usable_only_once_logged_in() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let mut app = App::new();
    app.insert_resource(RigHome(Home::at(root.path())))
        .init_resource::<Models>()
        .add_plugins(ClaudeCloakPlugin);
    let usable = |app: &App| {
        let models = &app.world().resource::<Models>().0;
        let statuses = models.statuses().into_iter();
        let cloaked = statuses.filter(|(spec, _)| spec.provider.vendor() == VENDOR);
        cloaked
            .map(|(_, status)| status == ModelStatus::Usable)
            .collect::<Vec<_>>()
    };
    let before = usable(&app);
    anyhow::ensure!(!before.is_empty() && before.iter().all(|usable| !usable));
    let stored = root.path().join("anthropic-oauth");
    std::fs::DirBuilder::new().mode(0o700).create(&stored)?;
    let credentials = stored.join("credentials.json");
    std::fs::write(&credentials, "{}")?;
    std::fs::set_permissions(&credentials, std::fs::Permissions::from_mode(0o644))?;
    // A file others may read is not one a request reads.
    anyhow::ensure!(usable(&app).iter().all(|usable| !usable));
    std::fs::set_permissions(&credentials, std::fs::Permissions::from_mode(0o600))?;
    let after = usable(&app);
    anyhow::ensure!(after.iter().any(|usable| *usable), "{after:?}");
    Ok(())
}

/// A wrapper closest to the network that keeps each request and sends none.
#[derive(Clone, Default)]
struct Stopped(Arc<Mutex<Vec<Request<Bytes>>>>);

impl HttpClientExt for Stopped {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let kept = self.keep(req.map(Into::into));
        async move { Err(kept) }
    }

    fn send_multipart<U>(
        &self,
        _: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Err(http_client::Error::permanent(std::io::Error::other(
            "multipart",
        ))))
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let kept = self.keep(req.map(Into::into));
        async move { Err(kept) }
    }
}

impl Stopped {
    /// Keeps `request`; the error is what the caller gets in place of a reply.
    fn keep(&self, request: Request<Bytes>) -> http_client::Error {
        if let Ok(mut kept) = self.0.lock() {
            kept.push(request);
        }
        http_client::Error::permanent(std::io::Error::other("stopped before the network"))
    }
}

/// The bytes a [`TrafficLog`] wrote, shared with the test.
#[derive(Clone, Default)]
struct Written(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Written {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .map_err(|_| std::io::Error::other("poisoned"))?
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A model call on a signed-in home, through the connector's wrappers (the
/// real traffic log, then a stand-in for the network inside it): both see
/// the request the plugin sends, and the log writes no sign-in token.
#[tokio::test]
async fn wrappers_see_the_request_the_plugin_sends() -> anyhow::Result<()> {
    use rig_harness::rig_core::{
        completion::CompletionRequest,
        effect::{EffectId, EffectKind},
        serve::Dispatch,
    };

    const TOKEN: &str = "sk-ant-oat01-offline-sign-in-token-7f3a9c";
    let root = tempfile::tempdir()?;
    let stored = root.path().join("anthropic-oauth");
    std::fs::DirBuilder::new().mode(0o700).create(&stored)?;
    let credentials = stored.join("credentials.json");
    std::fs::write(
        &credentials,
        serde_json::to_vec(&serde_json::json!({
            "access_token": TOKEN,
            "refresh_token": "offline-refresh-token",
            "expires_at": 4_102_444_800_u64,
            "account_id": "offline-account",
            "org_id": null,
        }))?,
    )?;
    std::fs::set_permissions(&credentials, std::fs::Permissions::from_mode(0o600))?;

    let network = Stopped::default();
    let written = Written::default();
    let log = rig_cassette::traffic::TrafficLog::new(written.clone())?;
    let mut app = App::new();
    app.insert_resource(RigHome(Home::at(root.path())))
        .init_resource::<Models>()
        .add_plugins(ClaudeCloakPlugin);
    let mut models = app.world_mut().resource_mut::<Models>();
    let stand_in = network.clone();
    models
        .0
        .wrap_http(move |_| DynHttpClient::new(stand_in.clone()));
    let logged = log.clone();
    models.0.wrap_http(move |inner| logged.wrap_http(inner));
    let models = &app.world().resource::<Models>().0;
    let spec = models.catalog().resolve("claude-cloak/claude-sonnet-4-6")?;
    let handler = connect(root.path(), spec.spec, models.catalog(), models.transport())?;
    let outcome = handler
        .handle(
            EffectKind::Completion {
                request: Arc::new(CompletionRequest::new("hello").max_tokens(32)),
                stream: false,
            },
            Dispatch::new(EffectId::from_raw(0), false),
        )
        .await
        .into_outcome()
        .await;
    anyhow::ensure!(outcome.is_err(), "the stand-in sends nothing");

    let sent = network
        .0
        .lock()
        .map_err(|_| anyhow::anyhow!("poisoned"))?
        .drain(..)
        .collect::<Vec<_>>();
    let [request] = sent.as_slice() else {
        anyhow::bail!("{} requests reached the network", sent.len());
    };
    let uri = request.uri().to_string();
    anyhow::ensure!(uri.ends_with("/v1/messages?beta=true"), "{uri}");
    let headers = request.headers();
    anyhow::ensure!(!headers.contains_key("x-api-key"));
    anyhow::ensure!(
        headers.get("authorization").map(HeaderValue::as_bytes)
            == Some(format!("Bearer {TOKEN}").as_bytes())
    );
    let body: serde_json::Value = serde_json::from_slice(request.body())?;
    anyhow::ensure!(body.get("_omp").is_none(), "{body}");

    log.flush();
    let text = String::from_utf8(
        written
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned"))?
            .clone(),
    )?;
    let line: serde_json::Value = serde_json::from_str(text.trim())?;
    anyhow::ensure!(!text.contains(TOKEN), "{text}");
    anyhow::ensure!(!text.contains("_omp"), "{text}");
    anyhow::ensure!(
        line.pointer("/request/url")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|url| url.ends_with("/v1/messages?beta=true")),
        "{line}"
    );
    anyhow::ensure!(
        line.pointer("/request/headers/authorization").is_some(),
        "{line}"
    );
    anyhow::ensure!(line.pointer("/request/body/messages").is_some(), "{line}");
    Ok(())
}

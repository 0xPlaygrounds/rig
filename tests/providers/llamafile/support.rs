//! Shared helpers for Llamafile live tests.

use rig::provider::ProviderConfig;
use rig::providers::llamafile;
use url::Url;

const DEFAULT_API_BASE_URL: &str = "http://localhost:8080";
const DEFAULT_MODEL: &str = llamafile::LLAMA_CPP;

pub(super) fn api_base_url() -> String {
    std::env::var("LLAMAFILE_API_BASE_URL").unwrap_or_else(|_| DEFAULT_API_BASE_URL.to_string())
}

pub(super) fn model_name() -> String {
    std::env::var("LLAMAFILE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

/// `LLAMAFILE_API_BASE_URL` is a bare host URL; the functions path builds
/// wire URLs verbatim, so the `/v1` the deleted client appended lives here.
fn versioned_base_url() -> String {
    format!("{}/v1", api_base_url().trim_end_matches('/'))
}

pub(super) fn config(model: impl Into<String>) -> llamafile::functions::Config {
    llamafile::functions::Config::new(model).with_base_url(versioned_base_url())
}

pub(super) fn provider(model: impl Into<String>) -> ProviderConfig {
    ProviderConfig::Llamafile(config(model))
}

pub(super) fn embedding_config(model: impl Into<String>) -> llamafile::functions::EmbeddingConfig {
    llamafile::functions::EmbeddingConfig::new(model).with_base_url(versioned_base_url())
}

fn server_addr() -> Option<String> {
    let url = Url::parse(&api_base_url()).ok()?;
    let host = url.host_str()?;
    let port = url.port_or_known_default()?;

    Some(format!("{host}:{port}"))
}

pub(super) fn skip_if_server_unavailable() -> bool {
    let Some(addr) = server_addr() else {
        eprintln!(
            "skipping llamafile live test: could not derive a socket address from {:?}",
            api_base_url()
        );
        return true;
    };

    if std::net::TcpStream::connect(&addr).is_err() {
        eprintln!("skipping llamafile live test: no server listening on {addr}");
        return true;
    }

    false
}

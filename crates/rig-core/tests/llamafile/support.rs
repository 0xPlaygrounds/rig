//! Shared helpers for Llamafile live tests.

use rig_core::providers::llamafile;
use url::Url;

const DEFAULT_API_BASE_URL: &str = "http://localhost:8080";
const DEFAULT_MODEL: &str = llamafile::LLAMA_CPP;

pub(super) fn api_base_url() -> String {
    std::env::var("LLAMAFILE_API_BASE_URL").unwrap_or_else(|_| DEFAULT_API_BASE_URL.to_string())
}

pub(super) fn model_name() -> String {
    std::env::var("LLAMAFILE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

pub(super) fn client() -> llamafile::Client {
    llamafile::Client::from_url(&api_base_url()).expect("client should build")
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

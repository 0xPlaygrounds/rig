//! Browser sign-in pieces: PKCE codes, the authorize URL, the callback
//! listener's place, and launching the default browser.

use super::{CHATGPT_AUTH_BASE, CHATGPT_CLIENT_ID};
use crate::providers::internal::loopback::{Loopback, LoopbackHost};
use base64::Engine;
use base64::prelude::BASE64_URL_SAFE_NO_PAD;
use sha2::{Digest, Sha256};
use std::process::{Command, Stdio};

/// The registered callback ports, tried in order.
const CALLBACK_PORTS: [u16; 2] = [1455, 1457];
const CALLBACK_PATH: &str = "/auth/callback";
/// Enough for the token to work and to refresh it.
const SCOPE: &str = "openid profile email offline_access";

/// A PKCE verifier and its S256 challenge.
pub(super) struct Pkce {
    pub(super) verifier: String,
    pub(super) challenge: String,
}

impl Pkce {
    pub(super) fn generate() -> Self {
        // Four v4 UUIDs: 64 bytes from the OS generator.
        let verifier = random_token(4);
        Self {
            challenge: pkce_challenge(&verifier),
            verifier,
        }
    }
}

/// The S256 challenge of `verifier`.
pub(super) fn pkce_challenge(verifier: &str) -> String {
    BASE64_URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

/// `uuids` × 16 random bytes, URL-safe base64.
pub(super) fn random_token(uuids: usize) -> String {
    let bytes: Vec<u8> = (0..uuids)
        .flat_map(|_| uuid::Uuid::new_v4().into_bytes())
        .collect();
    BASE64_URL_SAFE_NO_PAD.encode(bytes)
}

/// The registered callback, on `127.0.0.1`, whose pages tell the user to
/// return to `app`.
pub(super) fn loopback(app: &str) -> Loopback {
    Loopback::new("ChatGPT", LoopbackHost::Ipv4, CALLBACK_PORTS, CALLBACK_PATH).with_app_name(app)
}

/// The page that asks the user to sign in and sends them back to
/// `redirect_uri` with a code.
pub(super) fn authorize_url(
    redirect_uri: &str,
    code_challenge: &str,
    state: &str,
    originator: &str,
) -> String {
    let query = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs([
            ("response_type", "code"),
            ("client_id", CHATGPT_CLIENT_ID),
            ("redirect_uri", redirect_uri),
            ("code_challenge", code_challenge),
            ("code_challenge_method", "S256"),
            ("state", state),
            ("scope", SCOPE),
            ("id_token_add_organizations", "true"),
            ("codex_cli_simplified_flow", "true"),
            ("originator", originator),
        ])
        .finish();
    format!("{CHATGPT_AUTH_BASE}/oauth/authorize?{query}")
}

/// Asks the desktop to open `url` in the default browser through
/// [`browser_command`], without waiting. Returns whether it started.
pub(super) fn open_browser(url: &str) -> bool {
    let Ok(mut child) = browser_command(url).spawn() else {
        return false;
    };
    // Reap the opener when it exits.
    std::thread::Builder::new()
        .name("browser-opener".into())
        .spawn(move || {
            child.wait().ok();
        })
        .ok();
    true
}

/// The command that asks the desktop to open `url` in the default browser:
/// `open` on macOS, `start` through `cmd` on Windows, `xdg-open` elsewhere.
/// It has no standard input, and its output is discarded so it does not
/// write to the terminal. Native only.
pub fn browser_command(url: &str) -> Command {
    let mut command = platform_command(url);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command
}

#[cfg(target_os = "macos")]
fn platform_command(url: &str) -> Command {
    let mut command = Command::new("open");
    command.arg(url);
    command
}

#[cfg(windows)]
fn platform_command(url: &str) -> Command {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let mut command = Command::new("cmd");
    // Quoted so that `cmd` keeps the query's `&`s; the URL holds no `"`.
    command
        .raw_arg(format!("/c start \"\" \"{url}\""))
        .creation_flags(CREATE_NO_WINDOW);
    command
}

#[cfg(not(any(target_os = "macos", windows)))]
fn platform_command(url: &str) -> Command {
    let mut command = Command::new("xdg-open");
    command.arg(url);
    command
}

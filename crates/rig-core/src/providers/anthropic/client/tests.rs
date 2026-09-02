use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvVarGuard {
    key: &'static str,
    original: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let original = std::env::var(key).ok();
        // SAFETY: Tests in this module hold ENV_LOCK while mutating process
        // environment and restore the original value before releasing it.
        unsafe { std::env::set_var(key, value) };

        Self { key, original }
    }

    fn remove(key: &'static str) -> Self {
        let original = std::env::var(key).ok();
        // SAFETY: Tests in this module hold ENV_LOCK while mutating process
        // environment and restore the original value before releasing it.
        unsafe { std::env::remove_var(key) };

        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: Tests in this module hold ENV_LOCK while mutating process
        // environment and restore the original value before releasing it.
        unsafe {
            match &self.original {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

#[test]
fn test_client_initialization() {
    let _client = crate::providers::anthropic::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");
    let _client_from_builder = crate::providers::anthropic::Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
}

#[test]
fn from_env_uses_anthropic_base_url() {
    let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
    let _api_key = EnvVarGuard::set("ANTHROPIC_API_KEY", "dummy-key");
    let _base_url = EnvVarGuard::set(
        "ANTHROPIC_BASE_URL",
        "https://anthropic-compatible.example/v1/messages",
    );

    let client =
        <crate::providers::anthropic::client::Anthropic as crate::client::Provider>::from_env(
            crate::test_utils::RecordingHttpClient::new(""),
        )
        .expect("Client::from_env should build with ANTHROPIC_BASE_URL");

    assert_eq!(
        client.base_url(),
        "https://anthropic-compatible.example",
        "from_env should apply ANTHROPIC_BASE_URL and the existing Anthropic base URL normalization"
    );
}

#[test]
fn from_env_uses_default_base_url_when_anthropic_base_url_is_unset() {
    let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
    let _api_key = EnvVarGuard::set("ANTHROPIC_API_KEY", "dummy-key");
    let _base_url = EnvVarGuard::remove("ANTHROPIC_BASE_URL");

    let client =
        <crate::providers::anthropic::client::Anthropic as crate::client::Provider>::from_env(
            crate::test_utils::RecordingHttpClient::new(""),
        )
        .expect("Client::from_env should build without ANTHROPIC_BASE_URL");

    assert_eq!(client.base_url(), "https://api.anthropic.com");
}

//! The request identity and attestation for the observed Claude Code release.

pub(crate) mod checksum;

use rig_harness::rig_core::completion::CacheRetention;

pub(crate) const VERSION: &str = "2.1.307";
pub(crate) const SDK: &str = "0.128.0";
pub(crate) const RUNTIME: &str = "v26.3.0";
pub(crate) const OAUTH_BETA: &str = "oauth-2025-04-20";
pub(crate) const UTILITY_BETAS: &str = "oauth-2025-04-20,interleaved-thinking-2025-05-14,redact-thinking-2026-02-12,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,per-turn-control-2026-07-01,mid-conversation-tool-changes-2026-07-01,mid-conversation-system-clear-at-2026-08-21,structured-outputs-2025-12-15,cache-diagnosis-2026-04-07";
pub(crate) const USER_AGENT: &str = "claude-cli/2.1.307 (external, cli)";
pub(crate) const IDENTITY: &str = "You are Claude Code, Anthropic's official CLI for Claude.";
pub(crate) const ENTRYPOINT: &str = "cli";
pub(crate) const BETAS: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,per-turn-control-2026-07-01,mid-conversation-tool-changes-2026-07-01,mid-conversation-system-clear-at-2026-08-21,extended-cache-ttl-2025-04-11,cache-diagnosis-2026-04-07";
pub(crate) const CACHE: CacheRetention = CacheRetention::Long;
pub(crate) const GZIP_THRESHOLD: usize = 4096;
pub(crate) const SUFFIX_SALT: &str = "59cf53e54c78";
pub(crate) const CCH_SEED: u64 = 0x4d659218e32a3268;

pub(crate) fn seed(text: &str) -> [u16; 3] {
    let units: Vec<_> = text.encode_utf16().collect();
    [4, 7, 20].map(|i| units.get(i).copied().unwrap_or(u16::from(b'0')))
}

pub(crate) fn suffix_from_seed(selected: &[u16; 3]) -> String {
    use sha2::{Digest, Sha256};
    let hash = format!(
        "{:x}",
        Sha256::digest(format!(
            "{SUFFIX_SALT}{}{VERSION}",
            String::from_utf16_lossy(selected)
        ))
    );
    hash.chars().take(3).collect()
}

pub(crate) fn suffix(text: &str) -> String {
    suffix_from_seed(&seed(text))
}

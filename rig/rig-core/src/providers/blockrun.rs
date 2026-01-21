//! BlockRun API client and Rig integration
//!
//! BlockRun provides pay-per-request access to 30+ AI models via x402 micropayments.
//! Users pay with USDC on Base - no API keys required, just a funded wallet.
//!
//! # Example
//! ```ignore
//! use rig::providers::blockrun;
//!
//! // Create client with wallet private key (for signing payments)
//! let client = blockrun::Client::from_env(); // reads BLOCKRUN_WALLET_KEY
//!
//! // Or create with explicit private key
//! let client = blockrun::Client::from_private_key("0x...your_private_key")?;
//!
//! // Use any supported model - no API keys needed, just USDC in your wallet
//! let claude = client.completion_model(blockrun::CLAUDE_SONNET_4);
//! let gpt = client.completion_model(blockrun::GPT_4O);
//! let deepseek = client.completion_model(blockrun::DEEPSEEK_CHAT);
//! ```
//!
//! # Supported Models
//!
//! BlockRun provides access to models from multiple providers:
//! - **Anthropic**: Claude Sonnet 4, Claude Opus 4, Claude Haiku 3.5
//! - **OpenAI**: GPT-4o, GPT-4o-mini, o1, o3-mini
//! - **Google**: Gemini 2.0 Flash, Gemini 2.5 Pro
//! - **DeepSeek**: DeepSeek Chat, DeepSeek Reasoner
//! - **xAI**: Grok 2, Grok 3 (with Live Search)
//!
//! # Payment Flow
//!
//! BlockRun uses the x402 protocol for payments:
//! 1. Client makes request without payment
//! 2. Server returns 402 with payment requirements
//! 3. Client signs EIP-712 authorization (USDC on Base)
//! 4. Client retries with signed payment header
//! 5. Server verifies, processes request, settles payment
//!
//! The private key never leaves your machine - it's only used for local signing.

use crate::client::{
    self, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder, ProviderClient,
};
use crate::completion::{self, CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::sse::{Event, GenericEventSource};
use crate::http_client::{self, HttpClientExt};
use crate::message::{Document, DocumentSourceKind};
use crate::{json_utils, message, OneOrMany};
use async_stream::stream;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use futures::StreamExt;
use http::{Request, StatusCode};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{enabled, info_span, Instrument, Level};

// ================================================================
// Constants
// ================================================================
const BLOCKRUN_API_BASE_URL: &str = "https://blockrun.ai/api";

// Base Mainnet
const BASE_CHAIN_ID: u64 = 8453;
const USDC_BASE: &str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913";

// ================================================================
// Available Models
// ================================================================

// Anthropic
pub const CLAUDE_SONNET_4: &str = "anthropic/claude-sonnet-4";
pub const CLAUDE_OPUS_4: &str = "anthropic/claude-opus-4";
pub const CLAUDE_HAIKU_35: &str = "anthropic/claude-3-5-haiku";

// OpenAI
pub const GPT_4O: &str = "openai/gpt-4o";
pub const GPT_4O_MINI: &str = "openai/gpt-4o-mini";
pub const GPT_O1: &str = "openai/o1";
pub const GPT_O3_MINI: &str = "openai/o3-mini";

// Google
pub const GEMINI_20_FLASH: &str = "google/gemini-2.0-flash";
pub const GEMINI_25_PRO: &str = "google/gemini-2.5-pro";

// DeepSeek
pub const DEEPSEEK_CHAT: &str = "deepseek/deepseek-chat";
pub const DEEPSEEK_REASONER: &str = "deepseek/deepseek-reasoner";

// xAI
pub const GROK_2: &str = "xai/grok-2";
pub const GROK_3: &str = "xai/grok-3";

// ================================================================
// x402 Payment Types
// ================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PaymentAccept {
    scheme: String,
    network: String,
    amount: String,
    asset: String,
    #[serde(rename = "payTo")]
    pay_to: String,
    #[serde(rename = "maxTimeoutSeconds")]
    max_timeout_seconds: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    extra: Option<PaymentExtra>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PaymentExtra {
    name: String,
    version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PaymentRequired {
    #[serde(rename = "x402Version")]
    x402_version: u32,
    accepts: Vec<PaymentAccept>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resource: Option<ResourceInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResourceInfo {
    url: String,
    description: String,
    #[serde(rename = "mimeType")]
    mime_type: String,
}

#[derive(Debug, Serialize)]
struct PaymentPayload {
    #[serde(rename = "x402Version")]
    x402_version: u32,
    resource: ResourceInfo,
    accepted: PaymentAccepted,
    payload: SignaturePayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    extensions: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct PaymentAccepted {
    scheme: String,
    network: String,
    amount: String,
    asset: String,
    #[serde(rename = "payTo")]
    pay_to: String,
    #[serde(rename = "maxTimeoutSeconds")]
    max_timeout_seconds: u64,
    extra: PaymentExtra,
}

#[derive(Debug, Serialize)]
struct SignaturePayload {
    signature: String,
    authorization: Authorization,
}

#[derive(Debug, Serialize)]
struct Authorization {
    from: String,
    to: String,
    value: String,
    #[serde(rename = "validAfter")]
    valid_after: String,
    #[serde(rename = "validBefore")]
    valid_before: String,
    nonce: String,
}

// ================================================================
// EIP-712 Signing
// ================================================================

/// EIP-712 domain separator for USDC on Base
fn eip712_domain_separator() -> [u8; 32] {
    use sha3::{Digest, Keccak256};

    // EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)
    let type_hash = Keccak256::digest(
        b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)",
    );

    let name_hash = Keccak256::digest(b"USD Coin");
    let version_hash = Keccak256::digest(b"2");

    let mut chain_id_bytes = [0u8; 32];
    chain_id_bytes[24..].copy_from_slice(&BASE_CHAIN_ID.to_be_bytes());

    let contract_bytes = hex::decode(&USDC_BASE[2..]).expect("Invalid USDC address");
    let mut contract_padded = [0u8; 32];
    contract_padded[12..].copy_from_slice(&contract_bytes);

    // Hash: keccak256(typeHash || nameHash || versionHash || chainId || verifyingContract)
    let mut data = Vec::with_capacity(160);
    data.extend_from_slice(&type_hash);
    data.extend_from_slice(&name_hash);
    data.extend_from_slice(&version_hash);
    data.extend_from_slice(&chain_id_bytes);
    data.extend_from_slice(&contract_padded);

    Keccak256::digest(&data).into()
}

/// Hash for TransferWithAuthorization struct
fn transfer_struct_hash(
    from: &str,
    to: &str,
    value: &str,
    valid_after: u64,
    valid_before: u64,
    nonce: &[u8; 32],
) -> [u8; 32] {
    use sha3::{Digest, Keccak256};

    // TransferWithAuthorization(address from,address to,uint256 value,uint256 validAfter,uint256 validBefore,bytes32 nonce)
    let type_hash = Keccak256::digest(
        b"TransferWithAuthorization(address from,address to,uint256 value,uint256 validAfter,uint256 validBefore,bytes32 nonce)",
    );

    // Pad addresses to 32 bytes
    let from_bytes = hex::decode(&from[2..]).expect("Invalid from address");
    let mut from_padded = [0u8; 32];
    from_padded[12..].copy_from_slice(&from_bytes);

    let to_bytes = hex::decode(&to[2..]).expect("Invalid to address");
    let mut to_padded = [0u8; 32];
    to_padded[12..].copy_from_slice(&to_bytes);

    // Value as uint256
    let value_num: u128 = value.parse().expect("Invalid value");
    let mut value_bytes = [0u8; 32];
    value_bytes[16..].copy_from_slice(&value_num.to_be_bytes());

    // Timestamps as uint256
    let mut valid_after_bytes = [0u8; 32];
    valid_after_bytes[24..].copy_from_slice(&valid_after.to_be_bytes());

    let mut valid_before_bytes = [0u8; 32];
    valid_before_bytes[24..].copy_from_slice(&valid_before.to_be_bytes());

    // Hash the struct
    let mut data = Vec::with_capacity(224);
    data.extend_from_slice(&type_hash);
    data.extend_from_slice(&from_padded);
    data.extend_from_slice(&to_padded);
    data.extend_from_slice(&value_bytes);
    data.extend_from_slice(&valid_after_bytes);
    data.extend_from_slice(&valid_before_bytes);
    data.extend_from_slice(nonce);

    Keccak256::digest(&data).into()
}

/// Create EIP-712 typed data hash
fn eip712_hash(struct_hash: [u8; 32]) -> [u8; 32] {
    use sha3::{Digest, Keccak256};

    let domain_separator = eip712_domain_separator();

    let mut data = Vec::with_capacity(66);
    data.extend_from_slice(&[0x19, 0x01]);
    data.extend_from_slice(&domain_separator);
    data.extend_from_slice(&struct_hash);

    Keccak256::digest(&data).into()
}

/// Sign EIP-712 typed data with secp256k1 private key
fn sign_eip712(private_key: &[u8; 32], message_hash: [u8; 32]) -> Result<String, CompletionError> {
    use k256::ecdsa::{RecoveryId, Signature, SigningKey};

    let signing_key = SigningKey::from_bytes(private_key.into())
        .map_err(|e| CompletionError::ProviderError(format!("Invalid private key: {}", e)))?;

    let (signature, recovery_id): (Signature, RecoveryId) = signing_key
        .sign_prehash_recoverable(&message_hash)
        .map_err(|e| CompletionError::ProviderError(format!("Signing failed: {}", e)))?;

    // Encode as 65-byte signature (r || s || v)
    let mut sig_bytes = [0u8; 65];
    sig_bytes[..64].copy_from_slice(&signature.to_bytes());
    sig_bytes[64] = recovery_id.to_byte() + 27; // Ethereum uses 27/28 for v

    Ok(format!("0x{}", hex::encode(sig_bytes)))
}

/// Get wallet address from private key
fn get_address_from_private_key(private_key: &[u8; 32]) -> Result<String, CompletionError> {
    use k256::ecdsa::SigningKey;
    use sha3::{Digest, Keccak256};

    let signing_key = SigningKey::from_bytes(private_key.into())
        .map_err(|e| CompletionError::ProviderError(format!("Invalid private key: {}", e)))?;

    let public_key = signing_key.verifying_key();
    let public_key_bytes = public_key.to_encoded_point(false);
    let public_key_uncompressed = &public_key_bytes.as_bytes()[1..]; // Skip the 0x04 prefix

    let hash = Keccak256::digest(public_key_uncompressed);
    let address_bytes = &hash[12..]; // Last 20 bytes

    Ok(format!("0x{}", hex::encode(address_bytes)))
}

// ================================================================
// BlockRun API Key (Private Key for signing)
// ================================================================

#[derive(Clone)]
pub struct BlockRunAuth {
    private_key: [u8; 32],
    address: String,
}

impl std::fmt::Debug for BlockRunAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockRunAuth")
            .field("address", &self.address)
            .field("private_key", &"[REDACTED]")
            .finish()
    }
}

impl crate::client::ApiKey for BlockRunAuth {
    fn into_header(self) -> Option<crate::http_client::Result<(http::HeaderName, http::HeaderValue)>> {
        // BlockRun uses x402 payment headers instead of bearer auth
        // The payment is added per-request, not as a default header
        None
    }
}

impl BlockRunAuth {
    pub fn new(private_key_hex: &str) -> Result<Self, CompletionError> {
        let key_hex = private_key_hex.strip_prefix("0x").unwrap_or(private_key_hex);
        let key_bytes = hex::decode(key_hex)
            .map_err(|e| CompletionError::ProviderError(format!("Invalid hex key: {}", e)))?;

        if key_bytes.len() != 32 {
            return Err(CompletionError::ProviderError(
                "Private key must be 32 bytes".to_string(),
            ));
        }

        let mut private_key = [0u8; 32];
        private_key.copy_from_slice(&key_bytes);

        let address = get_address_from_private_key(&private_key)?;

        Ok(Self {
            private_key,
            address,
        })
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    /// Create a signed x402 payment payload
    fn create_payment(
        &self,
        payment_required: &PaymentRequired,
    ) -> Result<String, CompletionError> {
        let accept = payment_required
            .accepts
            .first()
            .ok_or_else(|| CompletionError::ProviderError("No payment options".to_string()))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| CompletionError::ProviderError(format!("Time error: {}", e)))?
            .as_secs();

        let valid_after = now.saturating_sub(600); // 10 minutes before (clock skew)
        let valid_before = now + accept.max_timeout_seconds;

        // Generate random nonce
        let mut nonce = [0u8; 32];
        rand::rng().fill_bytes(&mut nonce);
        let nonce_hex = format!("0x{}", hex::encode(nonce));

        // Create struct hash
        let struct_hash = transfer_struct_hash(
            &self.address,
            &accept.pay_to,
            &accept.amount,
            valid_after,
            valid_before,
            &nonce,
        );

        // Create final hash and sign
        let message_hash = eip712_hash(struct_hash);
        let signature = sign_eip712(&self.private_key, message_hash)?;

        // Build payment payload
        let payload = PaymentPayload {
            x402_version: 2,
            resource: payment_required
                .resource
                .clone()
                .unwrap_or_else(|| ResourceInfo {
                    url: "https://blockrun.ai/api/v1/chat/completions".to_string(),
                    description: "BlockRun AI API call".to_string(),
                    mime_type: "application/json".to_string(),
                }),
            accepted: PaymentAccepted {
                scheme: accept.scheme.clone(),
                network: accept.network.clone(),
                amount: accept.amount.clone(),
                asset: accept.asset.clone(),
                pay_to: accept.pay_to.clone(),
                max_timeout_seconds: accept.max_timeout_seconds,
                extra: PaymentExtra {
                    name: "USD Coin".to_string(),
                    version: "2".to_string(),
                },
            },
            payload: SignaturePayload {
                signature,
                authorization: Authorization {
                    from: self.address.clone(),
                    to: accept.pay_to.clone(),
                    value: accept.amount.clone(),
                    valid_after: valid_after.to_string(),
                    valid_before: valid_before.to_string(),
                    nonce: nonce_hex,
                },
            },
            extensions: None,
        };

        let json = serde_json::to_string(&payload)
            .map_err(|e| CompletionError::ProviderError(format!("JSON error: {}", e)))?;

        Ok(BASE64.encode(json.as_bytes()))
    }
}

// ================================================================
// BlockRun Provider
// ================================================================

#[derive(Debug, Clone)]
pub struct BlockRunExt {
    auth: BlockRunAuth,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BlockRunExtBuilder;

impl Provider for BlockRunExt {
    type Builder = BlockRunExtBuilder;

    const VERIFY_PATH: &'static str = "/v1/models";

    fn build<H>(
        builder: &crate::client::ClientBuilder<
            Self::Builder,
            <Self::Builder as ProviderBuilder>::ApiKey,
            H,
        >,
    ) -> http_client::Result<Self> {
        Ok(Self {
            auth: builder.get_api_key().clone(),
        })
    }
}

impl<H> Capabilities<H> for BlockRunExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing; // TODO: Add image generation support
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for BlockRunExt {}

impl ProviderBuilder for BlockRunExtBuilder {
    type Output = BlockRunExt;
    type ApiKey = BlockRunAuth;

    const BASE_URL: &'static str = BLOCKRUN_API_BASE_URL;
}

pub type Client<H = reqwest::Client> = client::Client<BlockRunExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<BlockRunExtBuilder, BlockRunAuth, H>;

impl ProviderClient for Client {
    type Input = BlockRunAuth;

    fn from_env() -> Self {
        let private_key =
            std::env::var("BLOCKRUN_WALLET_KEY").expect("BLOCKRUN_WALLET_KEY not set");
        let auth = BlockRunAuth::new(&private_key).expect("Invalid private key");

        let mut client_builder = Self::builder();
        client_builder.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        let client_builder = client_builder.api_key(auth);
        client_builder.build().unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        let mut client_builder = Self::builder();
        client_builder.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        let client_builder = client_builder.api_key(input);
        client_builder.build().unwrap()
    }
}

impl Client {
    /// Create a new BlockRun client with the given wallet private key.
    ///
    /// BlockRun uses x402 micropayments instead of API keys.
    /// The private key is used to sign payment authorizations locally.
    /// It never leaves your machine - only the signature is sent.
    ///
    /// # Example
    /// ```ignore
    /// let client = blockrun::Client::from_private_key("0x...")?;
    /// ```
    pub fn from_private_key(private_key: &str) -> Result<Self, CompletionError> {
        let auth = BlockRunAuth::new(private_key)?;
        Ok(Self::from_val(auth))
    }

    /// Get the wallet address associated with this client
    pub fn address(&self) -> &str {
        self.ext().auth.address()
    }
}

// ================================================================
// API Response Types
// ================================================================

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Usage {
    pub completion_tokens: u32,
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

impl Usage {
    fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_vec",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message::System {
            content: content.to_owned(),
            name: None,
        }
    }
}

impl From<message::ToolResult> for Message {
    fn from(tool_result: message::ToolResult) -> Self {
        let content = match tool_result.content.first() {
            message::ToolResultContent::Text(text) => text.text,
            message::ToolResultContent::Image(_) => String::from("[Image]"),
        };

        Message::ToolResult {
            tool_call_id: tool_result.id,
            content,
        }
    }
}

impl From<message::ToolCall> for ToolCall {
    fn from(tool_call: message::ToolCall) -> Self {
        Self {
            id: tool_call.id,
            index: 0,
            r#type: ToolType::Function,
            function: Function {
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            },
        }
    }
}

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::User { content } => {
                let mut messages = vec![];

                let tool_results = content
                    .clone()
                    .into_iter()
                    .filter_map(|content| match content {
                        message::UserContent::ToolResult(tool_result) => {
                            Some(Message::from(tool_result))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();

                messages.extend(tool_results);

                let text_messages = content
                    .into_iter()
                    .filter_map(|content| match content {
                        message::UserContent::Text(text) => Some(Message::User {
                            content: text.text,
                            name: None,
                        }),
                        message::UserContent::Document(Document {
                            data:
                                DocumentSourceKind::Base64(content)
                                | DocumentSourceKind::String(content),
                            ..
                        }) => Some(Message::User {
                            content,
                            name: None,
                        }),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                messages.extend(text_messages);

                Ok(messages)
            }
            message::Message::Assistant { content, .. } => {
                let mut messages: Vec<Message> = vec![];
                let mut text_content = String::new();

                content.iter().for_each(|content| {
                    if let message::AssistantContent::Text(text) = content {
                        text_content.push_str(text.text());
                    }
                });

                messages.push(Message::Assistant {
                    content: text_content,
                    name: None,
                    tool_calls: vec![],
                });

                let tool_calls = content
                    .clone()
                    .into_iter()
                    .filter_map(|content| match content {
                        message::AssistantContent::ToolCall(tool_call) => {
                            Some(ToolCall::from(tool_call))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();

                if !tool_calls.is_empty() {
                    messages.push(Message::Assistant {
                        content: "".to_string(),
                        name: None,
                        tool_calls,
                    });
                }

                Ok(messages)
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    pub index: usize,
    #[serde(default)]
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: completion::ToolDefinition,
}

impl From<crate::completion::ToolDefinition> for ToolDefinition {
    fn from(tool: crate::completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: tool,
        }
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let content = match &choice.message {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut content = if content.trim().is_empty() {
                    vec![]
                } else {
                    vec![completion::AssistantContent::text(content)]
                };

                content.extend(
                    tool_calls
                        .iter()
                        .map(|call| {
                            completion::AssistantContent::tool_call(
                                &call.id,
                                &call.function.name,
                                call.function.arguments.clone(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                Ok(content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }?;

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = completion::Usage {
            input_tokens: response.usage.prompt_tokens as u64,
            output_tokens: response.usage.completion_tokens as u64,
            total_tokens: response.usage.total_tokens as u64,
        };

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
        })
    }
}

// ================================================================
// Completion Request
// ================================================================

#[derive(Debug, Serialize, Deserialize)]
struct BlockRunCompletionRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for BlockRunCompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        let mut full_history: Vec<Message> = match &req.preamble {
            Some(preamble) => vec![Message::system(preamble)],
            None => vec![],
        };

        if let Some(docs) = req.normalized_documents() {
            let docs: Vec<Message> = docs.try_into()?;
            full_history.extend(docs);
        }

        let chat_history: Vec<Message> = req
            .chat_history
            .clone()
            .into_iter()
            .map(|message| message.try_into())
            .collect::<Result<Vec<Vec<Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        full_history.extend(chat_history);

        Ok(Self {
            model: model.to_string(),
            messages: full_history,
            temperature: req.temperature,
            tools: req
                .tools
                .clone()
                .into_iter()
                .map(ToolDefinition::from)
                .collect::<Vec<_>>(),
            tool_choice: None, // TODO: Handle tool_choice properly
            additional_params: req.additional_params,
        })
    }
}

// ================================================================
// Completion Model
// ================================================================

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub client: Client<T>,
    pub model: String,
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self {
            client: client.clone(),
            model: model.into(),
        }
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "blockrun",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        span.record("gen_ai.system_instructions", &completion_request.preamble);

        let request =
            BlockRunCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        if enabled!(Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "BlockRun completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let url = format!("{}/v1/chat/completions", BLOCKRUN_API_BASE_URL);

        async move {
            // Use reqwest directly for the initial request to handle 402 properly
            // Rig's HTTP client abstraction treats 402 as an error before we can process it
            let http_client = reqwest::Client::new();

            // First request - will return 402 with payment requirements
            let initial_response = http_client
                .post(&url)
                .header("Content-Type", "application/json")
                .body(body.clone())
                .send()
                .await
                .map_err(|e| CompletionError::ProviderError(format!("Request failed: {}", e)))?;

            let status = initial_response.status();

            // Handle 402 Payment Required
            if status == StatusCode::PAYMENT_REQUIRED {
                // Extract payment requirements from header
                let payment_header = initial_response
                    .headers()
                    .get("x-payment-required")
                    .or_else(|| initial_response.headers().get("payment-required"))
                    .ok_or_else(|| {
                        CompletionError::ProviderError(
                            "402 response missing payment header".to_string(),
                        )
                    })?
                    .to_str()
                    .map_err(|_| {
                        CompletionError::ProviderError("Invalid payment header".to_string())
                    })?
                    .to_string();

                // Parse payment requirements
                let payment_required_json = BASE64.decode(&payment_header).map_err(|e| {
                    CompletionError::ProviderError(format!(
                        "Failed to decode payment header: {}",
                        e
                    ))
                })?;

                let payment_required: PaymentRequired =
                    serde_json::from_slice(&payment_required_json).map_err(|e| {
                        CompletionError::ProviderError(format!(
                            "Failed to parse payment requirements: {}",
                            e
                        ))
                    })?;

                // Create signed payment
                let payment_payload = self.client.ext().auth.create_payment(&payment_required)?;

                // Retry with payment header
                let paid_response = http_client
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .header("payment", &payment_payload)
                    .header("x-payment", &payment_payload)
                    .body(body)
                    .send()
                    .await
                    .map_err(|e| CompletionError::ProviderError(format!("Paid request failed: {}", e)))?;

                let paid_status = paid_response.status();
                let response_body = paid_response
                    .bytes()
                    .await
                    .map_err(|e| CompletionError::ProviderError(format!("Failed to read response: {}", e)))?;

                if paid_status.is_success() {
                    match serde_json::from_slice::<ApiResponse<CompletionResponse>>(&response_body)?
                    {
                        ApiResponse::Ok(response) => {
                            let span = tracing::Span::current();
                            span.record("gen_ai.usage.input_tokens", response.usage.prompt_tokens);
                            span.record(
                                "gen_ai.usage.output_tokens",
                                response.usage.completion_tokens,
                            );
                            if enabled!(Level::TRACE) {
                                tracing::trace!(target: "rig::completions",
                                    "BlockRun completion response: {}",
                                    serde_json::to_string_pretty(&response)?
                                );
                            }
                            response.try_into()
                        }
                        ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
                    }
                } else {
                    Err(CompletionError::ProviderError(
                        String::from_utf8_lossy(&response_body).to_string(),
                    ))
                }
            } else if status.is_success() {
                // Unexpected success without payment (shouldn't happen)
                let response_body = initial_response
                    .bytes()
                    .await
                    .map_err(|e| CompletionError::ProviderError(format!("Failed to read response: {}", e)))?;
                match serde_json::from_slice::<ApiResponse<CompletionResponse>>(&response_body)? {
                    ApiResponse::Ok(response) => response.try_into(),
                    ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
                }
            } else {
                let response_body = initial_response
                    .bytes()
                    .await
                    .map_err(|e| CompletionError::ProviderError(format!("Failed to read response: {}", e)))?;
                Err(CompletionError::ProviderError(
                    String::from_utf8_lossy(&response_body).to_string(),
                ))
            }
        }
        .instrument(span)
        .await
    }

    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>
    {
        let preamble = completion_request.preamble.clone();
        let mut request =
            BlockRunCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        let params = json_utils::merge(
            request.additional_params.unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream": true, "stream_options": {"include_usage": true}}),
        );
        request.additional_params = Some(params);

        if enabled!(Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "BlockRun streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let url = format!("{}/v1/chat/completions", BLOCKRUN_API_BASE_URL);

        // Use reqwest directly for the initial 402 request
        let http_client = reqwest::Client::new();

        // First request - will return 402 with payment requirements
        let initial_response = http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .body(body.clone())
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(format!("Request failed: {}", e)))?;

        let status = initial_response.status();

        // Handle 402 Payment Required
        let payment_payload = if status == StatusCode::PAYMENT_REQUIRED {
            let payment_header = initial_response
                .headers()
                .get("x-payment-required")
                .or_else(|| initial_response.headers().get("payment-required"))
                .ok_or_else(|| {
                    CompletionError::ProviderError("402 response missing payment header".to_string())
                })?
                .to_str()
                .map_err(|_| CompletionError::ProviderError("Invalid payment header".to_string()))?
                .to_string();

            let payment_required_json = BASE64.decode(&payment_header).map_err(|e| {
                CompletionError::ProviderError(format!("Failed to decode payment header: {}", e))
            })?;

            let payment_required: PaymentRequired =
                serde_json::from_slice(&payment_required_json).map_err(|e| {
                    CompletionError::ProviderError(format!(
                        "Failed to parse payment requirements: {}",
                        e
                    ))
                })?;

            self.client.ext().auth.create_payment(&payment_required)?
        } else {
            return Err(CompletionError::ProviderError(
                "Expected 402 response for payment".to_string(),
            ));
        };

        // Build the paid request using Rig's HTTP client for streaming support
        let final_req = self
            .client
            .post("/v1/chat/completions")?
            .header("payment", &payment_payload)
            .header("x-payment", &payment_payload)
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "blockrun",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        tracing::Instrument::instrument(
            send_streaming_request(self.client.clone(), final_req),
            span,
        )
        .await
    }
}

// ================================================================
// Streaming
// ================================================================

#[derive(Deserialize, Debug)]
struct StreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    tool_calls: Vec<StreamingToolCall>,
}

#[derive(Deserialize, Debug)]
struct StreamingToolCall {
    #[serde(default)]
    id: Option<String>,
    index: usize,
    function: StreamingFunction,
}

#[derive(Deserialize, Debug)]
struct StreamingFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
struct StreamingChoice {
    delta: StreamingDelta,
}

#[derive(Deserialize, Debug)]
struct StreamingCompletionChunk {
    choices: Vec<StreamingChoice>,
    usage: Option<Usage>,
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct StreamingCompletionResponse {
    pub usage: Usage,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.usage.prompt_tokens as u64;
        usage.output_tokens = self.usage.completion_tokens as u64;
        usage.total_tokens = self.usage.total_tokens as u64;
        Some(usage)
    }
}

async fn send_streaming_request<T>(
    http_client: T,
    req: Request<Vec<u8>>,
) -> Result<crate::streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    let mut event_source = GenericEventSource::new(http_client, req);

    let stream = stream! {
        let mut final_usage = Usage::new();
        let mut calls: HashMap<usize, (String, String, String)> = HashMap::new();

        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => {
                    tracing::trace!("SSE connection opened");
                    continue;
                }
                Ok(Event::Message(message)) => {
                    if message.data.trim().is_empty() || message.data == "[DONE]" {
                        continue;
                    }

                    let parsed = serde_json::from_str::<StreamingCompletionChunk>(&message.data);
                    let Ok(data) = parsed else {
                        let err = parsed.unwrap_err();
                        tracing::debug!("Couldn't parse SSE payload: {:?}", err);
                        continue;
                    };

                    if let Some(choice) = data.choices.first() {
                        let delta = &choice.delta;

                        // Handle tool calls
                        for tool_call in &delta.tool_calls {
                            let function = &tool_call.function;

                            if function.name.as_ref().map(|s| !s.is_empty()).unwrap_or(false) {
                                let id = tool_call.id.clone().unwrap_or_default();
                                let name = function.name.clone().unwrap();
                                calls.insert(tool_call.index, (id, name, String::new()));
                            } else if let Some(arguments) = &function.arguments {
                                if let Some((id, name, existing_args)) = calls.get(&tool_call.index) {
                                    let combined = format!("{}{}", existing_args, arguments);
                                    calls.insert(tool_call.index, (id.clone(), name.clone(), combined));
                                }
                            }
                        }

                        if let Some(content) = &delta.content {
                            yield Ok(crate::streaming::RawStreamingChoice::Message(content.clone()));
                        }
                    }

                    if let Some(usage) = data.usage {
                        final_usage = usage;
                    }
                }
                Err(crate::http_client::Error::StreamEnded) => {
                    break;
                }
                Err(err) => {
                    tracing::error!(?err, "SSE error");
                    yield Err(CompletionError::ResponseError(err.to_string()));
                    break;
                }
            }
        }

        event_source.close();

        // Flush accumulated tool calls
        for (_index, (id, name, arguments)) in calls {
            if let Ok(arguments_json) = serde_json::from_str::<serde_json::Value>(&arguments) {
                yield Ok(crate::streaming::RawStreamingChoice::ToolCall(
                    crate::streaming::RawStreamingToolCall::new(id, name, arguments_json)
                ));
            }
        }

        yield Ok(crate::streaming::RawStreamingChoice::FinalResponse(
            StreamingCompletionResponse { usage: final_usage }
        ));
    };

    Ok(crate::streaming::StreamingCompletionResponse::stream(
        Box::pin(stream),
    ))
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_address_derivation() {
        // Test vector: well-known test private key
        let private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
        let auth = BlockRunAuth::new(private_key).unwrap();
        // This is the first Hardhat/Anvil test account address
        assert_eq!(
            auth.address().to_lowercase(),
            "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266"
        );
    }

    #[test]
    fn test_payment_required_parsing() {
        let json = r#"{
            "x402Version": 2,
            "accepts": [{
                "scheme": "exact",
                "network": "eip155:8453",
                "amount": "1000",
                "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                "payTo": "0x1234567890123456789012345678901234567890",
                "maxTimeoutSeconds": 300,
                "extra": {"name": "USD Coin", "version": "2"}
            }],
            "resource": {
                "url": "https://blockrun.ai/api/v1/chat/completions",
                "description": "AI inference",
                "mimeType": "application/json"
            }
        }"#;

        let payment_required: PaymentRequired = serde_json::from_str(json).unwrap();
        assert_eq!(payment_required.x402_version, 2);
        assert_eq!(payment_required.accepts.len(), 1);
        assert_eq!(payment_required.accepts[0].amount, "1000");
    }

    #[test]
    fn test_completion_response_parsing() {
        let json = r#"{
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, world!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let response: CompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.choices.len(), 1);
        match &response.choices[0].message {
            Message::Assistant { content, .. } => assert_eq!(content, "Hello, world!"),
            _ => panic!("Expected assistant message"),
        }
    }
}

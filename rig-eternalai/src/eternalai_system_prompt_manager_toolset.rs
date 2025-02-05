use ethers::prelude::*;
use reqwest::get;
use std::ffi::c_uint;
use std::sync::Arc;

const IPFS: &str = "ipfs://";
const LIGHTHOUSE_IPFS: &str = "https://gateway.lighthouse.storage/ipfs/";
const GCS_ETERNAL_AI_BASE_URL: &str = "https://cdn.eternalai.org/upload/";

pub async fn fetch_system_prompt_raw_or_ipfs(content: &str) -> Option<String> {
    if content.contains(IPFS) {
        let light_house = content.replace(IPFS, LIGHTHOUSE_IPFS);
        tracing::debug!("light_house : {}", light_house);
        let mut response = get(light_house).await.unwrap();
        if response.status().is_success() {
            let body = response.text().await.unwrap();
            tracing::debug!("light_house body: {}", body);
            return Some(body);
        } else {
            let gcs = content.replace(IPFS, GCS_ETERNAL_AI_BASE_URL);
            tracing::debug!("gcs: {}", gcs);
            response = get(gcs).await.unwrap();
            if response.status().is_success() {
                let body = response.text().await.unwrap();
                tracing::debug!("gcs body: {}", body);
                return Some(body);
            } else {
                return None;
            }
        }
    }
    Some(content.to_string())
}

pub async fn get_on_chain_system_prompt(
    rpc_url: &str,
    contract_addr: &str,
    agent_id: c_uint,
) -> Result<Option<String>, String> {
    abigen!(
        SystemPromptManagementContract,
        r#"
        [{"inputs": [{"internalType": "uint256", "name": "_agentId", "type": "uint256"}], "name": "getAgentSystemPrompt", "outputs": [{"internalType": "bytes[]", "name": "","type": "bytes[]"}], "stateMutability": "view", "type": "function"}]
        "#
    );
    let provider =
        Provider::<Http>::try_from(rpc_url).map_err(|e| format!("Failed to parse url: {}", e))?;
    let client = Arc::new(provider);
    let contract_address: Address = contract_addr
        .parse()
        .map_err(|e| format!("invalid contract address: {}", e))?;
    let contract = SystemPromptManagementContract::new(contract_address, client);
    let system_prompts: Vec<Bytes> = contract
        .get_agent_system_prompt(U256::from(agent_id))
        .call()
        .await
        .map_err(|e| format!("invalid agent system prompt: {}", e))?;

    let decoded_strings: Vec<String> = system_prompts
        .iter()
        .map(|bytes| {
            String::from_utf8(bytes.to_vec()).unwrap_or_else(|_| "[Invalid UTF-8]".to_string())
        })
        .collect();

    if !decoded_strings.is_empty() {
        let prompt = decoded_strings[0].clone();
        tracing::debug!("system prompt : {}", prompt);
        return Ok(fetch_system_prompt_raw_or_ipfs(&prompt).await);
    }
    Ok(None)
}

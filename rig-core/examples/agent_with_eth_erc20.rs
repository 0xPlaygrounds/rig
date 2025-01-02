use std::{collections::HashMap, str::FromStr, sync::Arc};

use alloy::{
    network::EthereumWallet,
    primitives::{Address, TxHash, B256, U256},
    providers::{ProviderBuilder, RootProvider},
    signers::local::PrivateKeySigner,
    sol,
    transports::http::{Client, Http},
};
use anyhow::{anyhow, Result};
use rig::{completion::ToolDefinition, tool::Tool};
use serde::{Deserialize, Serialize};
use serde_json::json;

// Chain infos for context, in your project you should put it in a separate configuration file or database,
// refer to https://github.com/anylots/rig-eth/tree/dev/configs.
pub const CHAIN_INFOS: &str = r#"[
    {  
        "chain": "ethereum",  
        "provider_url": "https://eth-mainnet.g.alchemy.com/v2/YOUR-API-KEY",  
        "tokens": {
            "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"  
        }
    },  
    {
        "chain": "arbitrum",  
        "provider_url": "https://arb-mainnet.g.alchemy.com/v2/YOUR-API-KEY",  
        "tokens": {  
            "USDC": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
            "WBTC": "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f"  
        }  
    },
    {
        "chain": "base",  
        "provider_url": "https://base-mainnet.g.alchemy.com/v2/YOUR-API-KEY",
        "tokens": {  
            "USDC": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "WBTC": "0x0555E30da8f98308EdB960aa94C0Db47230d2B9c"
        }  
    },
    {
        "chain": "local",  
        "provider_url": "http://localhost:8545",
        "tokens": {  
            "USDC": "0x5FbDB2315678afecb367f032d93F642f64180aa3"
        }  
    }
]"#;

#[derive(Debug, Serialize, Deserialize)]
pub struct ChainInfo {
    chain: String,
    #[serde(skip_serializing)]
    provider_url: String,
    tokens: HashMap<String, String>, // token_symbol => token_address
}

#[derive(Deserialize)]
pub struct TransferArgs {
    chain: String,
    token_address: String,
    to_address: String,
    amount: String,
}

#[derive(Debug, thiserror::Error)]
#[error("ERC20 error")]
pub struct ERC20Error {
    message: String,
}

sol! {
    #[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
    #[sol(rpc)]
    interface IERC20 {
        function transfer(address to, uint256 amount) public returns (bool);
        function decimals() public view returns (uint8);
    }
}

#[derive(Deserialize, Serialize)]
pub struct ERC20Transfer;
impl Tool for ERC20Transfer {
    const NAME: &'static str = "erc20_transfer";

    type Error = ERC20Error;
    type Args = TransferArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "erc20_transfer".to_string(),
            description: "Transfer ERC20 tokens to a specific address".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "token_address": {
                        "type": "string",
                        "description": "The address of the ERC20 token contract"
                    },
                    "chain": {
                        "type": "string",
                        "description": "The chain name, such as arbitrum"
                    },
                    "to_address": {
                        "type": "string",
                        "description": "The receiving address"
                    },
                    "amount": {
                        "type": "string",
                        "description": "The amount of tokens to transfer"
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let chain_name = args.chain;
        let token_address =
            Address::from_str(&args.token_address).expect("parse token_address error");
        let to_address = Address::from_str(&args.to_address).expect("parse to_address error");
        let amount = u128::from_str(&args.amount).expect("parse amount error");
        println!(
            "call erc20_transfer tool, chain_name: {}, token_address: {}, to_address: {}, amount: {}",
            chain_name, token_address, to_address, amount
        );

        let chain_infos: Vec<ChainInfo> = serde_json::from_str(CHAIN_INFOS).unwrap();
        let provider_url = chain_infos
            .iter()
            .find(|c| c.chain == chain_name)
            .unwrap()
            .provider_url
            .clone();

        let result = transfer_erc20(to_address, amount, token_address, provider_url).await;
        match result {
            Ok(h) => Ok(h.to_string()),
            Err(e) => Err(ERC20Error {
                message: format!("transfer_erc20 error: {}", e),
            }),
        }
    }
}

async fn transfer_erc20(
    to_address: Address,
    amount: u128,
    token_address: Address,
    provider_url: String,
) -> std::result::Result<B256, anyhow::Error> {
    // Read the private key from the environment variable
    // let private_key = env::var("PRIVATE_KEY").expect("ETH wallet PRIVATE_KEY not set");

    // [RISK WARNING! Writing a private key in the code file is insecure behavior.]
    // The following code is for testing only. Set up signer from private key, be aware of danger.
    let private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
    let priv_signer: PrivateKeySigner = private_key.parse().expect("parse PrivateKeySigner");

    // Create a http client to the EVM chain network.
    let provider: RootProvider<Http<Client>> =
        ProviderBuilder::new().on_http(provider_url.parse().expect("parse l1_rpc to Url"));

    // Create eth signer with provider.
    let signer = Arc::new(
        ProviderBuilder::new()
            .with_recommended_fillers()
            .wallet(EthereumWallet::from(priv_signer.clone()))
            .on_provider(provider.clone()),
    );

    // Create contract instance.
    let erc20 = IERC20::IERC20Instance::new(token_address, signer);

    // Sync send transfer call.
    let tx_hash: std::result::Result<TxHash, anyhow::Error> = async move {
        let handle = tokio::task::spawn_blocking(move || {
            let result = tokio::runtime::Handle::current().block_on(async {
                let decimal = erc20.decimals().call().await.unwrap()._0;
                erc20
                    .transfer(to_address, U256::from(amount * 10u128.pow(decimal.into())))
                    .send()
                    .await
            });
            result
        });
        match handle.await {
            Ok(Ok(tx)) => Ok(tx.tx_hash().clone()),
            Ok(Err(e)) => Err(anyhow!(format!("alloy rpc error: {}", e))), // sign_transaction
            Err(e) => Err(anyhow!(format!("tokio exec error: {}", e))),    // spawn_blocking
        }
    }
    .await;
    tx_hash
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    use rig::completion::Prompt;
    use rig::providers::openai;

    // Create OpenAI client and model
    let openai_client = openai::Client::from_url("sk-xxxxx", "https://api.xxxxx.xx/");

    // load chain configs
    let chain_infos: Vec<ChainInfo> = serde_json::from_str(CHAIN_INFOS).unwrap();

    let transfer_agent = openai_client
        .agent("Qwen/Qwen2.5-32B-Instruct")
        .preamble("You are a transfer agent here to help the user perform ERC20 token transfers.")
        .context(&serde_json::to_string(&chain_infos).unwrap())
        .max_tokens(2048)
        .tool(ERC20Transfer)
        .build();

    // Prompt the agent and print the response
    println!("Transfer ERC20 tokens");
    println!(
        "Transfer Agent: {}",
        transfer_agent
            .prompt("Transfer 10 USDC to 0x1CBd0109c7452926fC7cCf06e73aCC505A296cc7 on local")
            .await?
    );
    Ok(())
}

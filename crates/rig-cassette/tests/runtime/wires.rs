//! The matrix's wires over a bank transport: the models and parameters each
//! provider's `corpus_matrix` file names, bound to a client that serves bank
//! replies instead of a cassette. The key and the base URL reach no service.

use rig::http_client::DynHttpClient;
use rig::providers::anthropic::wire::AnthropicConfig;
use rig::providers::doubleword::{QWEN3_5_9B, QWEN3_5_397B_A17B};
use rig::providers::gemini::GeminiConfig;
use rig::providers::gemini::completion::{
    GEMINI_3_1_FLASH_LITE_PREVIEW, GEMINI_3_FLASH_PREVIEW, GenerateContent,
};
use rig::providers::openai::wire::{Chat, DEEPSEEK, DOUBLEWORD, OpenAIConfig, OpenAiWire, VENICE};
use rig::providers::openai::{GPT_5_MINI, GPT_5_NANO};
use rig::providers::venice::MISTRAL_SMALL_3_2_24B;
use rig_test_support::cassette_models::{AnthropicModels, GeminiModels, OpenAiModels};

use crate::ecs_matrix::{Wire, cells, cells::ThinkingWire};

const KEY: &str = "bank";
const BASE_URL: &str = "http://bank.invalid";

pub(crate) type OpenAi = Wire<rig::Model<OpenAiWire>>;

pub(crate) fn deepseek(http: DynHttpClient) -> OpenAi {
    let client = OpenAiModels::new(
        OpenAIConfig::with_key(&DEEPSEEK, KEY).with_base_url(BASE_URL),
        http,
    );
    Wire {
        thinking: ThinkingWire::DeepSeek,
        model: client.completion("deepseek-chat"),
        route: Some(client.completion("deepseek-reasoner")),
        temperature: Some(0.0),
        additional_params: None,
    }
}

pub(crate) fn doubleword(http: DynHttpClient) -> OpenAi {
    let client = OpenAiModels::new(
        OpenAIConfig::with_key(&DOUBLEWORD, KEY).with_base_url(BASE_URL),
        http,
    );
    Wire {
        thinking: ThinkingWire::Doubleword,
        model: client.completion(QWEN3_5_397B_A17B),
        route: Some(client.completion(QWEN3_5_9B)),
        temperature: Some(0.0),
        additional_params: Some(cells::reasoning_off),
    }
}

pub(crate) fn venice(http: DynHttpClient) -> OpenAi {
    let client = OpenAiModels::new(
        OpenAIConfig::with_key(&VENICE, KEY).with_base_url(BASE_URL),
        http,
    );
    Wire {
        thinking: ThinkingWire::Venice,
        model: client.completion(MISTRAL_SMALL_3_2_24B),
        route: Some(client.completion(MISTRAL_SMALL_3_2_24B)),
        temperature: Some(0.0),
        additional_params: None,
    }
}

pub(crate) fn openai_chat(http: DynHttpClient) -> Wire<rig::Model<Chat>> {
    let client = OpenAiModels::new(OpenAIConfig::new(KEY).with_base_url(BASE_URL), http);
    Wire {
        thinking: ThinkingWire::OpenAiChat,
        model: client.chat(GPT_5_MINI),
        route: Some(client.chat(GPT_5_NANO)),
        temperature: None,
        additional_params: None,
    }
}

pub(crate) fn openai_responses(http: DynHttpClient) -> OpenAi {
    let client = OpenAiModels::new(OpenAIConfig::new(KEY).with_base_url(BASE_URL), http);
    Wire {
        thinking: ThinkingWire::OpenAiResponses,
        model: client.completion(GPT_5_MINI),
        route: Some(client.completion(GPT_5_NANO)),
        temperature: None,
        additional_params: Some(cells::openai_responses_stateless),
    }
}

pub(crate) fn gemini(http: DynHttpClient) -> Wire<rig::Model<GenerateContent, DynHttpClient>> {
    let client = GeminiModels::new(GeminiConfig::new(KEY).with_base_url(BASE_URL), http);
    Wire {
        thinking: ThinkingWire::Gemini,
        model: client.completion(GEMINI_3_FLASH_PREVIEW),
        route: Some(client.completion(GEMINI_3_1_FLASH_LITE_PREVIEW)),
        temperature: Some(0.0),
        additional_params: None,
    }
}

pub(crate) fn anthropic(
    http: DynHttpClient,
) -> Wire<rig::Model<rig::providers::anthropic::wire::Messages>> {
    let client = AnthropicModels::new(AnthropicConfig::new(KEY).with_base_url(BASE_URL), http);
    Wire {
        thinking: ThinkingWire::Anthropic,
        model: client.completion("claude-haiku-4-5-20251001"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

// The focused families' wires: the models their `corpus_matrix_*` and
// `ecs_matrix_*` files name, without a route.

fn thinking_disabled() -> serde_json::Value {
    serde_json::json!({"thinking": {"type": "disabled"}})
}

pub(crate) fn deepseek_flash(http: DynHttpClient) -> OpenAi {
    let client = OpenAiModels::new(
        OpenAIConfig::with_key(&DEEPSEEK, KEY).with_base_url(BASE_URL),
        http,
    );
    Wire {
        thinking: ThinkingWire::DeepSeek,
        model: client.completion("deepseek-flash"),
        route: None,
        temperature: Some(0.0),
        additional_params: Some(thinking_disabled),
    }
}

fn gemini_model(
    http: DynHttpClient,
    model: &str,
) -> Wire<rig::Model<GenerateContent, DynHttpClient>> {
    let client = GeminiModels::new(GeminiConfig::new(KEY).with_base_url(BASE_URL), http);
    Wire {
        thinking: ThinkingWire::Gemini,
        model: client.completion(model),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

pub(crate) fn gemini_flash_lite(
    http: DynHttpClient,
) -> Wire<rig::Model<GenerateContent, DynHttpClient>> {
    gemini_model(http, "gemini-2.5-flash-lite")
}

pub(crate) fn gemini_flash(
    http: DynHttpClient,
) -> Wire<rig::Model<GenerateContent, DynHttpClient>> {
    gemini_model(http, "gemini-2.5-flash")
}

pub(crate) fn gemini_task(http: DynHttpClient) -> Wire<rig::Model<GenerateContent, DynHttpClient>> {
    Wire {
        additional_params: Some(
            || serde_json::json!({"generationConfig":{"thinkingConfig":{"thinkingLevel":"low"}}}),
        ),
        ..gemini_model(http, "gemini-3.8-flash")
    }
}

pub(crate) fn gemini_preview(
    http: DynHttpClient,
) -> Wire<rig::Model<GenerateContent, DynHttpClient>> {
    gemini_model(http, GEMINI_3_FLASH_PREVIEW)
}

pub(crate) fn openai_chat_mini(http: DynHttpClient) -> Wire<rig::Model<Chat>> {
    let client = OpenAiModels::new(OpenAIConfig::new(KEY).with_base_url(BASE_URL), http);
    Wire {
        thinking: ThinkingWire::OpenAiChat,
        model: client.chat("gpt-4.1-mini"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

pub(crate) fn openai_chat_task(http: DynHttpClient) -> Wire<rig::Model<Chat>> {
    Wire {
        additional_params: Some(
            || serde_json::json!({"prompt_cache_key": "rig-native-long-tasks"}),
        ),
        ..openai_chat_mini(http)
    }
}

pub(crate) fn openai_responses_mini(http: DynHttpClient) -> OpenAi {
    let client = OpenAiModels::new(OpenAIConfig::new(KEY).with_base_url(BASE_URL), http);
    Wire {
        thinking: ThinkingWire::OpenAiResponses,
        model: client.completion("gpt-4.1-mini"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

pub(crate) fn openai_responses_task(http: DynHttpClient) -> OpenAi {
    Wire {
        additional_params: Some(
            || serde_json::json!({"prompt_cache_key": "rig-native-long-tasks", "store": false}),
        ),
        ..openai_responses_mini(http)
    }
}

pub(crate) fn openai_chat_image(http: DynHttpClient) -> Wire<rig::Model<Chat>> {
    Wire {
        route: None,
        ..openai_chat(http)
    }
}

pub(crate) fn openai_responses_image(http: DynHttpClient) -> OpenAi {
    Wire {
        route: None,
        ..openai_responses(http)
    }
}

pub(crate) fn anthropic_sonnet(
    http: DynHttpClient,
) -> Wire<rig::Model<rig::providers::anthropic::wire::Messages>> {
    let client = AnthropicModels::new(AnthropicConfig::new(KEY).with_base_url(BASE_URL), http);
    Wire {
        thinking: ThinkingWire::Anthropic,
        model: client.completion(rig::providers::anthropic::completion::CLAUDE_SONNET_4_6),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

pub(crate) fn deepseek_scripted(http: DynHttpClient) -> OpenAi {
    Wire {
        route: None,
        ..deepseek(http)
    }
}

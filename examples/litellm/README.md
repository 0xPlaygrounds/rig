# Rig with LiteLLM

Run a Rig agent through the [LiteLLM Proxy](https://docs.litellm.ai/docs/proxy/quick_start).
Rig sends OpenAI-compatible requests to the proxy, which routes them to the
configured provider. No LiteLLM Rust dependency is needed.

## Start a local proxy

Install the proxy with `uv`:

```sh
uv tool install 'litellm[proxy]'
```

From the repository root, in a shell with `OPENAI_API_KEY` exported, run:

```sh
litellm --config examples/litellm/config.yaml --host 127.0.0.1 --port 4000
```

The supplied configuration maps the public alias `rig-chat` to
`openai/gpt-4o-mini`. The provider key stays in the proxy's environment.
To use another provider, change `litellm_params` in `config.yaml` and supply
that provider's credentials to the proxy.

## Run the agent

In another shell, from the repository root:

```sh
LITELLM_API_KEY=unused cargo run -p litellm
```

The local configuration has no proxy authentication, so `unused` is a dummy
key. The example sends one prompt and prints the response.

For an existing authenticated proxy, export its key as `LITELLM_API_KEY` and
set the URL and model alias to match your deployment:

```sh
export LITELLM_BASE_URL=https://your-litellm-host/v1
export LITELLM_MODEL=your-model-alias
cargo run -p litellm
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `LITELLM_API_KEY` | Proxy key, or `unused` for the local unauthenticated configuration | Required |
| `LITELLM_BASE_URL` | Proxy API base URL, including `/v1` | `http://localhost:4000/v1` |
| `LITELLM_MODEL` | A `model_name` alias from the proxy configuration | `rig-chat` |

The Rust process needs the proxy key, not the upstream provider key.
`client.chat(model)` explicitly selects `/chat/completions`, producing
`/v1/chat/completions` with the default base URL. Rig's OpenAI client otherwise
defaults to the Responses API. The model value is the proxy alias, not the
provider-qualified `litellm_params.model` value.

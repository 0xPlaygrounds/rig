# rig-vectorize

Vector store integration for [Cloudflare Vectorize](https://developers.cloudflare.com/vectorize/). This integration supports vector similarity search and document insertion using Rig's embedding providers.

You can find end-to-end examples [here](https://github.com/0xPlaygrounds/rig/tree/main/rig-integrations/rig-vectorize/examples).

For Vectorize-specific questions, ask in the [Cloudflare Developers Discord](https://discord.com/channels/595317990191398933/1152193114522525726).

## Running Integration Tests

Integration tests require a real Cloudflare Vectorize index.

### 1. Create a Vectorize index

```bash
npx wrangler vectorize create rig-integration-test --dimensions=1536 --metric=cosine
```

### 2. Set environment variables and run tests

```bash
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
export CLOUDFLARE_API_TOKEN="your-api-token"
export VECTORIZE_INDEX_NAME="rig-integration-test"
cargo test --package rig-vectorize --test integration_tests -- --test-threads=1
```

Tests run sequentially (`--test-threads=1`) to avoid conflicts since they clear the index before each test.

**Note:** Vectorize has eventual consistency. Tests wait 5 seconds after inserting documents before querying (configured via `EVENTUAL_CONSISTENCY_DELAY` constant).

### 3. (Optional) Enable filter tests

Filter tests require metadata indexes. Without them, filter tests will be skipped:

```bash
npx wrangler vectorize create-metadata-index rig-integration-test --property-name=category --type=string
npx wrangler vectorize create-metadata-index rig-integration-test --property-name=id --type=string
```

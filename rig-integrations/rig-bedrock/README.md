## Rig-Bedrock
This companion crate integrates AWS Bedrock as model provider with Rig.

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-bedrock = "0.1.0"
rig-core = "0.9.1"
```

You can also run `cargo add rig-bedrock rig-core` to add the most recent versions of the dependencies to your project.

See the [`/examples`](./examples) folder for usage examples.

Make sure to have AWS credentials env vars loaded before starting client such as:
```shell
export AWS_DEFAULT_REGION=us-east-1
export AWS_SECRET_ACCESS_KEY=.......
export AWS_ACCESS_KEY_ID=......
```

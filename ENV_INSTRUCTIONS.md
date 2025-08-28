# Environment Variables

**Web Search**

- `TAVILY_API_KEY` - Used to power your agent's web searches. You can get a free key [here](https://app.tavily.com).

**Benchmarking**

- `OPENAI_API_KEY` - Used for benchmarking OpenAI models. You can get a key [here](https://platform.openai.com/api-keys).
- `ANTHROPIC_API_KEY` - Used for benchmarking Anthropic models. You can get a key [here](https://console.anthropic.com/settings/keys).
- `GOOGLE_API_KEY` - Used for benchmarking Google models. You can get a key [here](https://aistudio.google.com/app/apikey).

**Observability**

- `WANDB_API_KEY` - Enables metric logging to Weights & Biases.
- `LANGSMITH_API_KEY` - Enables tracing to LangSmith.
- `LANGSMITH_PROJECT` - The project name to use for tracing.
- `LANGSMITH_TRACING` - Whether to enable tracing.

**AWS**

To enable backup to S3 and weight transfer between the SFT and GRPO runs, you'll also need to provide AWS credentials. If you don't already have AWS credentials with create/read/write permissions for s3 buckets, follow the instructions [here](CONFIGURING_AWS.md).

- `AWS_ACCESS_KEY_ID` - Your AWS access key ID, which should have create/read/write permissions for s3 buckets.
- `AWS_SECRET_ACCESS_KEY` - Your matching secret access key.
- `AWS_REGION` - The region of the S3 bucket.
- `BACKUP_BUCKET` - The name of the S3 bucket in which to store model checkpoints and logging data. Can be a new bucket or an existing one.

**Open Agent Platform (Optional)**

- `SUPABASE_KEY` - The key to use for the Open Agent Platform.
- `SUPABASE_URL` - The URL to use for the Open Agent Platform.
- `GET_API_KEYS_FROM_CONFIG` - Whether to get API keys from the config. Should be set to `true` for a production deployment on Open Agent Platform. Should be set to `false` otherwise, such as for local development.

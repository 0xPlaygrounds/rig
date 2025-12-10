use aws_config::meta::region::RegionProviderChain;
use aws_sdk_s3vectors::Client;
use aws_sdk_s3vectors::config::Credentials;
use rig::Embed;
use rig::client::{EmbeddingsClient, ProviderClient};
use rig::embeddings::EmbeddingsBuilder;
use rig::providers::openai::{self, Client as OpenAIClient};
use rig::vector_store::request::VectorSearchRequest;
use rig::vector_store::{InsertDocuments, VectorStoreIndex};
use std::env;

const BUCKET_NAME: &str = "foo_bucket";
const INDEX_NAME: &str = "foo_index";

#[derive(Embed, serde::Deserialize, serde::Serialize, Debug)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let access_key_id = env::var("AWS_ACCESS_KEY_ID")
        .expect("AWS_ACCESS_KEY_ID does not exist as an environment variable");
    let secret_access_key = env::var("AWS_SECRET_ACCESS_KEY")
        .expect("AWS_ACCESS_KEY_ID does not exist as an environment variable");

    let credentials = Credentials::new(access_key_id, secret_access_key, None, None, "test");
    let region_provider = RegionProviderChain::default_provider().or_else("us-east-1");

    let config = aws_config::from_env()
        .credentials_provider(credentials)
        .region(region_provider)
        .load()
        .await;

    let s3vectors_client = Client::new(&config);

    // set up infra idempotently - see individual functions
    create_vector_bucket(&s3vectors_client).await?;
    create_index(&s3vectors_client).await?;

    // Initialize OpenAI client.
    // Get your API key from https://platform.openai.com/api-keys
    let openai_client = OpenAIClient::from_env();

    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    let documents = EmbeddingsBuilder::new(model.clone())
        .document(Word {
            id: "0981d983-a5f8-49eb-89ea-f7d3b2196d2e".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        })?
        .document(Word {
            id: "62a36d43-80b6-4fd6-990c-f75bb02287d1".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        })?
        .document(Word {
            id: "f9e17d59-32e5-440c-be02-b2759a654824".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        })?
        .build()
        .await?;

    let store =
        rig_s3vectors::S3VectorsVectorStore::new(model, s3vectors_client, BUCKET_NAME, INDEX_NAME);

    store.insert_documents(documents).await?;
    let query = "What is a linglingdong?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(2)
        .build()?;

    let results = store.top_n::<Word>(req).await?;

    println!("#{} results for query: {}", results.len(), query);
    for (distance, _id, doc) in results.iter() {
        println!("Result distance {distance} for word: {doc:?}");

        // expected output
        // Result distance 0.693218142100547 for word: glarb-glarb
        // Result distance 0.2529120980283861 for word: linglingdong
    }

    Ok(())
}

pub async fn create_index(client: &aws_sdk_s3vectors::Client) -> Result<(), anyhow::Error> {
    if check_vector_index_exists(client).await? {
        return Ok(());
    };

    client
        .create_index()
        .index_name(INDEX_NAME)
        .vector_bucket_name(BUCKET_NAME)
        .send()
        .await
        .map_err(|x| anyhow::anyhow!("Error while creating index: {x}"))?;

    Ok(())
}

pub async fn check_vector_index_exists(
    client: &aws_sdk_s3vectors::Client,
) -> Result<bool, anyhow::Error> {
    match client
        .get_index()
        .vector_bucket_name(BUCKET_NAME)
        .index_name(INDEX_NAME)
        .send()
        .await
    {
        Ok(_) => Ok(true),
        Err(e) => {
            let aws_sdk_s3vectors::error::SdkError::ServiceError(err) = e else {
                return Err(anyhow::anyhow!(
                    "Error while checking vector index exists: {e}"
                ));
            };

            let err = err.into_err();

            if let aws_sdk_s3vectors::operation::get_index::GetIndexError::NotFoundException(_) =
                err
            {
                Ok(false)
            } else {
                Err(anyhow::anyhow!(
                    "Error while checking vector index exists: {err}"
                ))
            }
        }
    }
}

pub async fn create_vector_bucket(client: &aws_sdk_s3vectors::Client) -> Result<(), anyhow::Error> {
    if check_vector_bucket_exists(client).await? {
        return Ok(());
    };

    client
        .create_vector_bucket()
        .vector_bucket_name(BUCKET_NAME)
        .send()
        .await
        .map_err(|x| anyhow::anyhow!("Error while creating bucket: {x}"))?;

    Ok(())
}

pub async fn check_vector_bucket_exists(
    client: &aws_sdk_s3vectors::Client,
) -> Result<bool, anyhow::Error> {
    match client
        .get_vector_bucket()
        .vector_bucket_name(BUCKET_NAME)
        .send()
        .await
    {
        Ok(_) => Ok(true),
        Err(e) => {
            let aws_sdk_s3vectors::error::SdkError::ServiceError(err) = e else {
                return Err(anyhow::anyhow!(
                    "Error while checking vector bucket exists: {e}"
                ));
            };
            let err = err.into_err();

            if let aws_sdk_s3vectors::operation::get_vector_bucket::GetVectorBucketError::NotFoundException(_) = err {
                Ok(false)
            } else {
               Err(anyhow::anyhow!("Error while checking vector bucket exists: {err}"))
            }
        }
    }
}

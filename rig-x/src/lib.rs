use reqwest::header::{ACCEPT, CONTENT_TYPE};
use rig::completion::Chat;

const BASE_URL_2: &str = "https://api.x.com/2";

pub struct XArgs {
    pub token: String,
    pub client_id: String,
    pub client_secret: String,
    pub refresh_token: String,
    pub scopes: Vec<String>,
}

#[derive(serde::Deserialize)]
pub struct TokenResponse {
    pub refresh_token: String,
    pub access_token: String,
}

#[derive(serde::Deserialize)]
pub struct TweetResponse {
    pub edit_history_tweet_ids: Vec<String>,
    pub text: String,
    pub id: String,
}

pub struct XBot {
    client: reqwest::Client,
    x_args: XArgs,
}

impl XBot {
    pub fn new(x_args: XArgs) -> Self {
        Self {
            x_args,
            client: reqwest::Client::new(),
        }
    }

    pub async fn refresh_connection(mut self) -> reqwest::Result<Self> {
        let XArgs {
            client_id,
            client_secret,
            refresh_token,
            scopes,
            ..
        } = &self.x_args;

        let TokenResponse {
            access_token,
            refresh_token,
        } = self
            .client
            .post(format!("{BASE_URL_2}/oauth2/token"))
            .header(ACCEPT, "application/json")
            .header(
                CONTENT_TYPE,
                "application/x-www-form-urlencoded;charset=UTF-8",
            )
            .basic_auth(client_id, Some(client_secret))
            .query(&[
                ("grant_type", "refresh_token"),
                ("client_id", client_id),
                ("refresh_token", refresh_token),
                ("scope", &scopes.join(" ")),
            ])
            .send()
            .await?
            .json::<TokenResponse>()
            .await?;

        self.x_args.refresh_token = refresh_token;
        self.x_args.token = access_token;

        Ok(self)
    }

    pub async fn tweet(&self, tweet: String) -> reqwest::Result<TweetResponse> {
        self.client
            .post(format!("{BASE_URL_2}/tweets"))
            .bearer_auth(self.x_args.token.clone())
            .body(tweet)
            .send()
            .await?
            .json::<TweetResponse>()
            .await
    }
}

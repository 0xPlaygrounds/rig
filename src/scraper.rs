use crate::api::client::TwitterApiClient;
use crate::auth::user_auth::TwitterUserAuth;
use crate::auth::TwitterAuth;
use crate::constants::BEARER_TOKEN;
use crate::error::Result;
use crate::error::TwitterError;
use crate::models::{Profile, Tweet};
use crate::search::{fetch_search_tweets, SearchMode};
use crate::timeline::v1::{QueryProfilesResponse, QueryTweetsResponse};
use crate::timeline::v2::QueryTweetsResponse as V2QueryTweetsResponse;
use serde_json::Value;

pub struct Scraper {
    client: TwitterApiClient,
    auth: Box<dyn TwitterAuth + Send + Sync>,
}

impl Scraper {
    pub async fn new() -> Result<Self> {
        let auth = Box::new(TwitterUserAuth::new(BEARER_TOKEN.to_string()).await?);
        let client = TwitterApiClient::new(auth.clone())?;
        Ok(Self { client, auth })
    }

    pub async fn login(
        &mut self,
        username: String,
        password: String,
        email: Option<String>,
        two_factor_secret: Option<String>,
    ) -> Result<()> {
        if let Some(user_auth) = self.auth.as_any().downcast_ref::<TwitterUserAuth>() {
            let mut auth = user_auth.clone();
            auth.login(
                &username,
                &password,
                email.as_deref(),
                two_factor_secret.as_deref(),
            )
            .await?;

            self.auth = Box::new(auth.clone());
            self.client = TwitterApiClient::new(Box::new(auth))?;
            Ok(())
        } else {
            Err(TwitterError::Auth("Invalid auth type".into()))
        }
    }

    pub async fn get_profile(&self, username: &str) -> Result<crate::models::Profile> {
        crate::profile::get_profile(username, &*self.auth).await
    }
    pub async fn send_tweet(
        &self,
        text: &str,
        reply_to: Option<&str>,
        media_data: Option<Vec<(Vec<u8>, String)>>,
    ) -> Result<Value> {
        crate::tweets::create_tweet_request(text, &*self.auth, reply_to, media_data).await
    }

    pub fn get_auth(&self) -> &Box<dyn TwitterAuth + Send + Sync> {
        self.client.get_auth()
    }

    pub async fn get_home_timeline(
        &self,
        count: i32,
        seen_tweet_ids: Vec<String>,
    ) -> Result<Vec<Value>> {
        crate::timeline::home::fetch_home_timeline(count, seen_tweet_ids, &*self.auth).await
    }

    pub async fn save_cookies(&self, cookie_file: &str) -> Result<()> {
        if let Some(user_auth) = self.auth.as_any().downcast_ref::<TwitterUserAuth>() {
            user_auth.save_cookies_to_file(cookie_file).await
        } else {
            Err(TwitterError::Auth("Invalid auth type".into()))
        }
    }

    pub async fn get_cookie_string(&self) -> Result<String> {
        if let Some(user_auth) = self.auth.as_any().downcast_ref::<TwitterUserAuth>() {
            user_auth.get_cookie_string().await
        } else {
            Err(TwitterError::Auth("Invalid auth type".into()))
        }
    }

    pub async fn set_cookies(&mut self, json_str: &str) -> Result<()> {
        if let Some(user_auth) = self.auth.as_any().downcast_ref::<TwitterUserAuth>() {
            let mut auth = user_auth.clone();
            auth.set_cookies(json_str).await?;

            self.auth = Box::new(auth.clone());
            self.client = TwitterApiClient::new(Box::new(auth))?;
            Ok(())
        } else {
            Err(TwitterError::Auth("Invalid auth type".into()))
        }
    }

    pub async fn set_from_cookie_string(&mut self, cookie_string: &str) -> Result<()> {
        if let Some(user_auth) = self.auth.as_any().downcast_ref::<TwitterUserAuth>() {
            let mut auth = user_auth.clone();
            auth.set_from_cookie_string(cookie_string).await?;

            self.auth = Box::new(auth.clone());
            self.client = TwitterApiClient::new(Box::new(auth))?;
            Ok(())
        } else {
            Err(TwitterError::Auth("Invalid auth type".into()))
        }
    }

    pub async fn is_logged_in(&self) -> Result<bool> {
        self.auth.is_logged_in().await
    }

    pub async fn get_followers(
        &self,
        user_id: &str,
        count: i32,
        cursor: Option<String>,
    ) -> Result<(Vec<Profile>, Option<String>)> {
        crate::relationships::get_followers(user_id, count, cursor, &*self.auth).await
    }

    pub async fn get_following(
        &self,
        user_id: &str,
        count: i32,
        cursor: Option<String>,
    ) -> Result<(Vec<Profile>, Option<String>)> {
        crate::relationships::get_following(user_id, count, cursor, &*self.auth).await
    }

    pub async fn follow_user(&self, username: &str) -> Result<()> {
        crate::relationships::follow_user(username, &*self.auth).await
    }

    pub async fn unfollow_user(&self, username: &str) -> Result<()> {
        crate::relationships::unfollow_user(username, &*self.auth).await
    }

    pub async fn send_quote_tweet(
        &self,
        text: &str,
        quoted_tweet_id: &str,
        media_data: Option<Vec<(Vec<u8>, String)>>,
    ) -> Result<Value> {
        crate::tweets::create_quote_tweet(text, quoted_tweet_id, media_data, &*self.auth).await
    }

    pub async fn fetch_tweets_and_replies(
        &self,
        username: &str,
        max_tweets: i32,
        cursor: Option<&str>,
    ) -> Result<V2QueryTweetsResponse> {
        crate::tweets::fetch_tweets_and_replies(username, max_tweets, cursor, &*self.auth).await
    }
    pub async fn fetch_tweets_and_replies_by_user_id(
        &self,
        user_id: &str,
        max_tweets: i32,
        cursor: Option<&str>,
    ) -> Result<V2QueryTweetsResponse> {
        crate::tweets::fetch_tweets_and_replies_by_user_id(user_id, max_tweets, cursor, &*self.auth)
            .await
    }
    pub async fn fetch_list_tweets(
        &self,
        list_id: &str,
        max_tweets: i32,
        cursor: Option<&str>,
    ) -> Result<Value> {
        crate::tweets::fetch_list_tweets(list_id, max_tweets, cursor, &*self.auth).await
    }

    pub async fn like_tweet(&self, tweet_id: &str) -> Result<Value> {
        crate::tweets::like_tweet(tweet_id, &*self.auth).await
    }

    pub async fn retweet(&self, tweet_id: &str) -> Result<Value> {
        crate::tweets::retweet(tweet_id, &*self.auth).await
    }

    pub async fn create_long_tweet(
        &self,
        text: &str,
        reply_to: Option<&str>,
        media_ids: Option<Vec<String>>,
    ) -> Result<Value> {
        crate::tweets::create_long_tweet(text, reply_to, media_ids, &*self.auth).await
    }

    pub async fn get_tweet(&self, id: &str) -> Result<Tweet> {
        crate::tweets::get_tweet(id, &*self.auth).await
    }

    pub async fn search_tweets(
        &self,
        query: &str,
        max_tweets: i32,
        search_mode: SearchMode,
        cursor: Option<String>,
    ) -> Result<QueryTweetsResponse> {
        let auth = self.get_auth();

        fetch_search_tweets(query, max_tweets, search_mode, auth.as_ref(), cursor).await
    }

    pub async fn search_profiles(
        &self,
        query: &str,
        max_profiles: i32,
        cursor: Option<String>,
    ) -> Result<QueryProfilesResponse> {
        let auth = self.get_auth();

        crate::search::search_profiles(query, max_profiles, auth.as_ref(), cursor).await
    }

    pub async fn get_user_tweets(
        &self,
        user_id: &str,
        count: i32,
        cursor: Option<String>,
    ) -> Result<V2QueryTweetsResponse> {
        crate::tweets::fetch_user_tweets(user_id, count, cursor.as_deref(), &*self.auth).await
    }
}

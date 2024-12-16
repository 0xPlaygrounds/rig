use crate::error::Result;
use async_trait::async_trait;
use reqwest::header::HeaderMap;
use std::any::Any;
pub mod user_auth;

#[async_trait]
pub trait TwitterAuth: Any + Send + Sync {
    async fn is_logged_in(&self) -> Result<bool>;
    async fn install_headers(&self, headers: &mut HeaderMap) -> Result<()>;
    async fn get_cookies(&self) -> Result<Vec<cookie::Cookie<'static>>>;
    fn delete_token(&mut self);
    fn as_any(&self) -> &dyn Any;
}

pub struct AuthConfig {
    pub username: Option<String>,
    pub password: Option<String>,
    pub email: Option<String>,
    pub bearer_token: String,
    pub two_factor_secret: Option<String>,
}

impl AuthConfig {
    pub fn new(bearer_token: String) -> Self {
        Self {
            username: None,
            password: None,
            email: None,
            bearer_token,
            two_factor_secret: None,
        }
    }

    pub fn with_credentials(
        mut self,
        username: String,
        password: String,
        email: Option<String>,
    ) -> Self {
        self.username = Some(username);
        self.password = Some(password);
        self.email = email;
        self
    }
}

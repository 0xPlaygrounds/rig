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
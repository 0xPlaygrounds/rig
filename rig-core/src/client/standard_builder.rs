use crate::http_client::HttpClientExt;
use reqwest::header::HeaderMap;

/// Standard builder for creating client instances
///
/// Supports optional extension data for providers with custom fields.
/// For standard providers, `Ext` defaults to `()`.
pub struct Builder<'a, Client, T, Ext = ()> {
    api_key: &'a str,
    base_url: Option<&'a str>,
    http_client: Option<T>,
    custom_headers: Option<HeaderMap>,
    extension: Ext,
    _phantom: std::marker::PhantomData<Client>,
}

impl<'a, Client, T> Builder<'a, Client, T, ()>
where
    T: HttpClientExt + Default,
    Client: StandardClientBuilder<T>,
{
    fn new(api_key: &'a str) -> Self {
        Self {
            api_key,
            base_url: None,
            http_client: None,
            custom_headers: None,
            extension: (),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, Client, T, Ext> Builder<'a, Client, T, Ext>
where
    T: HttpClientExt + Default,
    Client: StandardClientBuilder<T>,
{
    /// Set the base URL for the API endpoint
    pub fn base_url(mut self, base_url: &'a str) -> Self {
        self.base_url = Some(base_url);
        self
    }

    /// Set the HTTP client for the API endpoint
    pub fn http_client(mut self, http_client: T) -> Self {
        self.http_client = Some(http_client);
        self
    }

    /// Set custom headers for the API endpoint
    pub fn custom_headers(mut self, headers: HeaderMap) -> Self {
        self.custom_headers = Some(headers);
        self
    }

    /// Add extension data to the builder
    pub fn with_extension<NewExt>(self, extension: NewExt) -> Builder<'a, Client, T, NewExt> {
        Builder {
            api_key: self.api_key,
            base_url: self.base_url,
            http_client: self.http_client,
            custom_headers: self.custom_headers,
            extension,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Update the extension data with a function
    pub fn update_extension<F>(mut self, f: F) -> Self
    where
        F: FnOnce(Ext) -> Ext,
    {
        self.extension = f(self.extension);
        self
    }

    /// Get the base URL, using the default if not set
    pub fn get_base_url<'b>(&self, default_base_url: &'b str) -> &'b str
    where
        'a: 'b,
    {
        self.base_url.unwrap_or(default_base_url)
    }

    /// Get the HTTP client, using the default if not set
    pub fn get_http_client(&self) -> T
    where
        T: Default + Clone,
    {
        self.http_client.clone().unwrap_or_default()
    }

    /// Get the API key
    pub fn get_api_key(&self) -> &str {
        self.api_key
    }

    /// Get the custom headers, with content-type header if `has_default_headers` is true
    pub fn get_headers(&self, has_default_headers: bool) -> HeaderMap {
        if has_default_headers {
            let mut headers = default_headers();
            headers.extend(self.custom_headers.clone().unwrap_or_default());
            headers
        } else {
            self.custom_headers.clone().unwrap_or_default()
        }
    }

    /// Try to get the extension as a specific type
    ///
    /// Returns `Some(T)` if the extension type matches, `None` otherwise.
    pub fn try_get_extension<ExtType>(&self) -> Option<ExtType>
    where
        ExtType: Clone + 'static,
        Ext: 'static,
    {
        use std::any::Any;
        (&self.extension as &dyn Any)
            .downcast_ref::<ExtType>()
            .cloned()
    }

    /// Build the client from the builder
    pub fn build(self) -> Result<Client, crate::client::ClientBuilderError>
    where
        T: Default + Clone,
        Client: StandardClientBuilder<T>,
        Ext: Default + 'static,
    {
        Client::build_from_builder(self)
    }
}

/// Trait for standardizing client builder implementations across providers
///
/// Implement this trait to get a standardized builder API. The `build_from_builder`
/// method handles both standard and extended cases through the `extension` parameter.
pub trait StandardClientBuilder<T>: Sized
where
    T: HttpClientExt,
{
    /// Build client from builder, including optional extension data
    ///
    /// For standard builders (Ext = ()), the extension field will be `()`.
    /// For extended builders, you can use `try_get_extension` to safely extract the extension.
    ///
    /// Returns `Result` to allow error handling.
    fn build_from_builder<Ext>(
        builder: Builder<'_, Self, T, Ext>,
    ) -> Result<Self, crate::client::ClientBuilderError>
    where
        Ext: Default + 'static,
        T: Default + Clone;

    fn builder(api_key: &str) -> Builder<'_, Self, T>
    where
        T: Default,
    {
        Builder::new(api_key)
    }
}

/// Default headers for the API endpoint
///
/// This includes the Content-Type header for JSON requests.
pub(crate) fn default_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );
    headers
}

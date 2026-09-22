use rig_reqwest::ReqwestMiddlewareClient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let inner = reqwest_middleware::ClientBuilder::new(reqwest::Client::builder().build()?).build();
    let client = ReqwestMiddlewareClient::new(inner);
    let inner = client.clone().into_inner();
    let _boxed = ReqwestMiddlewareClient::from(inner).boxed();
    Ok(())
}

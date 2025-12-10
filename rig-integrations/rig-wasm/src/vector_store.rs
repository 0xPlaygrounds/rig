use rig::vector_store::{VectorSearchRequest, VectorStoreError};
use send_wrapper::SendWrapper;
use serde::Deserialize;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{
    JsFuture,
    js_sys::{self, Array},
};

use crate::{JsResult, ensure_type_implements_functions};

#[wasm_bindgen]
pub struct JsVectorStore {
    inner: SendWrapper<JsValue>,
}

#[wasm_bindgen]
impl JsVectorStore {
    #[wasm_bindgen(constructor)]
    pub fn new(shim: JsValue) -> JsResult<Self> {
        let required_fns = vec!["topN", "topNIds"];
        ensure_type_implements_functions(&shim, required_fns)?;
        let inner = SendWrapper::new(shim);
        Ok(Self { inner })
    }
}

impl rig::vector_store::VectorStoreIndex for JsVectorStore {
    fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest,
    ) -> impl std::future::Future<
        Output = Result<Vec<(f64, String, T)>, rig::vector_store::VectorStoreError>,
    > + Send {
        let (result_tx, result_rx) = futures::channel::oneshot::channel();
        let (error_tx, error_rx) = futures::channel::oneshot::channel();
        let inner = self.inner.clone();

        wasm_bindgen_futures::spawn_local(async move {
            let req = match serde_wasm_bindgen::to_value(&req)
                .map_err(|x| VectorStoreError::DatastoreError(x.to_string().into()))
            {
                Ok(req) => req,
                Err(e) => {
                    error_tx
                        .send(e)
                        .expect("sending a message to a oneshot channel shouldn't fail");

                    return;
                }
            };
            let call_fn = match js_sys::Reflect::get(&inner, &JsValue::from_str("topN"))
                .map_err(|_| VectorStoreError::DatastoreError("vector_store.topN missing".into()))
            {
                Ok(res) => res,
                Err(e) => {
                    error_tx
                        .send(e)
                        .expect("sending a message to a oneshot channel shouldn't fail");

                    return;
                }
            };
            let call_fn = call_fn.unchecked_into::<js_sys::Function>();

            if !call_fn.is_function() {
                error_tx
                    .send(VectorStoreError::DatastoreError(
                        "vector_store.topN is not a function".into(),
                    ))
                    .expect("sending a message to a oneshot channel shouldn't fail");

                return;
            }

            let promise = match call_fn
                .call1(&inner, &req)
                .map_err(|_| VectorStoreError::DatastoreError("vector_store.topN failed".into()))
            {
                Ok(res) => res,
                Err(e) => {
                    error_tx
                        .send(e)
                        .expect("sending a message to a oneshot channel shouldn't fail");

                    return;
                }
            };
            let promise = match promise.dyn_into::<js_sys::Promise>().map_err(|_| {
                VectorStoreError::DatastoreError(
                    "vector_store.top_n did not return a Promise".into(),
                )
            }) {
                Ok(res) => res,
                Err(e) => {
                    error_tx
                        .send(e)
                        .expect("sending a message to a oneshot channel shouldn't fail");

                    return;
                }
            };

            let js_result = match JsFuture::from(promise).await.map_err(|x| {
                VectorStoreError::DatastoreError(
                    format!("promise did not return a JS future: {x:?}").into(),
                )
            }) {
                Ok(res) => res,
                Err(e) => {
                    error_tx
                        .send(e)
                        .expect("sending a message to a oneshot channel shouldn't fail");

                    return;
                }
            };

            let mut out: Vec<(f64, String, serde_json::Value)> = Vec::new();

            let arr = Array::from(&js_result).into_iter();
            for tuple in arr {
                let tuple = Array::from(&tuple);

                if tuple.length() != 3 {
                    let err =
                        VectorStoreError::DatastoreError("expected [number, string, obj]".into());
                    error_tx
                        .send(err)
                        .expect("sending a message to a oneshot channel shouldn't fail");

                    return;
                }

                let score = match tuple.get(0).as_f64() {
                    Some(res) => res,
                    None => {
                        let err = VectorStoreError::DatastoreError(
                            "expected score to be a number".into(),
                        );
                        error_tx
                            .send(err)
                            .expect("sending a message to a oneshot channel shouldn't fail");

                        return;
                    }
                };
                let id = match tuple.get(1).as_string() {
                    Some(res) => res,
                    None => {
                        let err = VectorStoreError::DatastoreError(
                            "expected document ID to be a string".into(),
                        );
                        error_tx
                            .send(err)
                            .expect("sending a message to a oneshot channel shouldn't fail");

                        return;
                    }
                };

                let obj: serde_json::Value = match serde_wasm_bindgen::from_value(tuple.get(2)) {
                    Ok(res) => res,
                    Err(e) => {
                        let err = VectorStoreError::DatastoreError(
                            format!(
                                "expected document to be a JSON compatible object or string: {e}"
                            )
                            .into(),
                        );
                        error_tx
                            .send(err)
                            .expect("sending a message to a oneshot channel shouldn't fail");

                        return;
                    }
                };

                out.push((score, id, obj));
            }

            let _ = result_tx.send(out);
        });

        async {
            tokio::select! {
                res = result_rx => {
                    {
                        let res = res.unwrap();
                        let res: Vec<(f64, String, T)> = res
                            .into_iter()
                            .map(|(score, id, payload)| {
                                let payload: T = serde_json::from_value(payload).unwrap();
                                (score, id, payload)
                            })
                            .collect();
                        Ok(res)
                    }
                },
                err = error_rx => {
                    Err(VectorStoreError::DatastoreError(err.inspect_err(|x| println!("Future was cancelled: {x}")).unwrap().to_string().into()))
                }
            }
        }
    }

    fn top_n_ids(
        &self,
        req: VectorSearchRequest,
    ) -> impl std::future::Future<
        Output = Result<Vec<(f64, String)>, rig::vector_store::VectorStoreError>,
    > + Send {
        let (result_tx, result_rx) = futures::channel::oneshot::channel();
        let (error_tx, error_rx) = futures::channel::oneshot::channel();
        let inner = self.inner.clone();

        wasm_bindgen_futures::spawn_local(async move {
            let req = match serde_wasm_bindgen::to_value(&req)
                .map_err(|x| VectorStoreError::DatastoreError(x.to_string().into()))
            {
                Ok(req) => req,
                Err(e) => {
                    error_tx
                        .send(e)
                        .expect("sending a message to a oneshot channel shouldn't fail");

                    return;
                }
            };
            let call_fn = js_sys::Reflect::get(&inner, &JsValue::from_str("topNIds"))
                .map_err(|_| {
                    VectorStoreError::DatastoreError("vector_store.topNIds missing".into())
                })
                .expect("Call function doesn't exist!")
                .unchecked_into::<js_sys::Function>();

            let promise = match call_fn
                .call1(&inner, &req)
                .map_err(|_| VectorStoreError::DatastoreError("vector_store.topNIds failed".into()))
            {
                Ok(res) => res,
                Err(e) => {
                    error_tx
                        .send(e)
                        .expect("sending a message to a oneshot channel shouldn't fail");

                    return;
                }
            };
            let promise = match promise.dyn_into::<js_sys::Promise>().map_err(|_| {
                VectorStoreError::DatastoreError(
                    "vector_store.top_n_ids did not return a Promise".into(),
                )
            }) {
                Ok(res) => res,
                Err(e) => {
                    error_tx
                        .send(e)
                        .expect("sending a message to a oneshot channel shouldn't fail");

                    return;
                }
            };

            let js_result = match JsFuture::from(promise).await.map_err(|x| {
                VectorStoreError::DatastoreError(
                    format!("promise did not return a JS future: {x:?}").into(),
                )
            }) {
                Ok(res) => res,
                Err(e) => {
                    error_tx
                        .send(e)
                        .expect("sending a message to a oneshot channel shouldn't fail");

                    return;
                }
            };

            let mut out: Vec<(f64, String)> = Vec::new();

            let arr = Array::from(&js_result).into_iter();
            for tuple in arr {
                let tuple = Array::from(&tuple);

                if tuple.length() != 2 {
                    panic!("expected [number, string]");
                }

                let score = match tuple.get(0).as_f64() {
                    Some(res) => res,
                    None => {
                        let err = VectorStoreError::DatastoreError(
                            "expected score to be a number".into(),
                        );
                        error_tx
                            .send(err)
                            .expect("sending a message to a oneshot channel shouldn't fail");

                        return;
                    }
                };
                let id = match tuple.get(1).as_string() {
                    Some(res) => res,
                    None => {
                        let err = VectorStoreError::DatastoreError(
                            "expected document ID to be a string".into(),
                        );
                        error_tx
                            .send(err)
                            .expect("sending a message to a oneshot channel shouldn't fail");

                        return;
                    }
                };

                out.push((score, id));
            }

            let _ = result_tx.send(out);
        });

        async {
            tokio::select! {
                res = result_rx => {
                        let res = res.unwrap();
                        Ok(res)
                },
                err = error_rx => {
                    Err(VectorStoreError::DatastoreError(err.inspect_err(|x| println!("Future was cancelled: {x}")).unwrap().to_string().into()))
                }
            }
        }
    }
}

use send_wrapper::SendWrapper;
use serde::Deserialize;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{
    JsFuture,
    js_sys::{self, Array},
};

use crate::JsVectorStoreShim;

#[wasm_bindgen]
pub struct JsVectorStore {
    inner: SendWrapper<JsVectorStoreShim>,
}

#[wasm_bindgen]
impl JsVectorStore {
    #[wasm_bindgen(constructor)]
    pub fn new(shim: JsVectorStoreShim) -> Self {
        let inner = SendWrapper::new(shim);
        Self { inner }
    }
}

impl rig::vector_store::VectorStoreIndex for JsVectorStore {
    fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<
        Output = Result<Vec<(f64, String, T)>, rig::vector_store::VectorStoreError>,
    > + Send {
        let (tx, rx) = futures::channel::oneshot::channel();
        let query = query.to_string();
        let inner = self.inner.clone().unchecked_into::<JsVectorStoreShim>();

        wasm_bindgen_futures::spawn_local(async move {
            let promise = inner
                .top_n(&query, n as u32)
                .unchecked_into::<js_sys::Promise>();

            let js_result = JsFuture::from(promise).await.unwrap();

            let mut out: Vec<(f64, String, serde_json::Value)> = Vec::new();

            Array::from(&js_result).into_iter().for_each(|tuple| {
                let tuple = Array::from(&tuple);

                if tuple.length() != 3 {
                    panic!("expected [number, string, obj]");
                }

                let score = tuple.get(0).as_f64().unwrap();
                let id = tuple.get(1).as_string().unwrap();

                let obj: serde_json::Value = serde_wasm_bindgen::from_value(tuple.get(2)).unwrap();

                out.push((score, id, obj));
            });

            let _ = tx.send(out);
        });

        async {
            let js_result = rx.await.unwrap();

            let res: Vec<(f64, String, T)> = js_result
                .into_iter()
                .map(|(score, id, payload)| {
                    let payload: T = serde_json::from_value(payload).unwrap();
                    (score, id, payload)
                })
                .collect();
            Ok(res)
        }
    }

    fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<
        Output = Result<Vec<(f64, String)>, rig::vector_store::VectorStoreError>,
    > + Send {
        let (tx, rx) = futures::channel::oneshot::channel();
        let query = query.to_string();
        let inner = self.inner.clone().unchecked_into::<JsVectorStoreShim>();

        wasm_bindgen_futures::spawn_local(async move {
            let promise = inner
                .top_n(&query, n as u32)
                .unchecked_into::<js_sys::Promise>();

            let js_result = JsFuture::from(promise).await.unwrap();

            let mut out: Vec<(f64, String)> = Vec::new();

            Array::from(&js_result).into_iter().for_each(|tuple| {
                let tuple = Array::from(&tuple);

                if tuple.length() != 3 {
                    panic!("expected [number, string, obj]");
                }

                let score = tuple.get(0).as_f64().unwrap();
                let id = tuple.get(1).as_string().unwrap();

                out.push((score, id));
            });

            let _ = tx.send(out);
        });

        async {
            let js_result = rx.await.unwrap();

            Ok(js_result)
        }
    }
}

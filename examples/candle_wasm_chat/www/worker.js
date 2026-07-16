import init, {
    chat,
    clear_history,
    embedded_model_size,
    initialize,
} from "./pkg/candle_wasm_chat.js";
import { createWorkerRuntime } from "./worker-runtime.js";

const runtime = createWorkerRuntime(
    { init, initialize, embeddedModelSize: embedded_model_size, chat, clearHistory: clear_history },
    (message) => self.postMessage(message),
);

self.onmessage = ({ data }) => {
    void runtime.handle(data);
};

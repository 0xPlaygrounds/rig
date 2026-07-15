import init, {
    chat,
    clear_history,
    embedded_model_size,
    initialize,
} from "./pkg/candle_wasm_chat.js";

const ready = (async () => {
    const started = performance.now();
    await init();
    initialize();
    self.postMessage({
        type: "ready",
        modelBytes: embedded_model_size(),
        initializationMs: Math.round(performance.now() - started),
    });
})().catch((error) => {
    self.postMessage({ type: "error", message: String(error) });
    throw error;
});

self.onmessage = async ({ data }) => {
    try {
        await ready;
        if (data.type === "clear") {
            clear_history();
            self.postMessage({ type: "cleared" });
            return;
        }
        if (data.type === "chat") {
            const output = await chat(data.message);
            self.postMessage({ type: "response", output });
        }
    } catch (error) {
        self.postMessage({ type: "error", message: String(error) });
    }
};

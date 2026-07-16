export function createWorkerRuntime(api, postMessage, now = () => performance.now()) {
    const ready = (async () => {
        const started = now();
        await api.init();
        api.initialize();
        postMessage({
            type: "ready",
            modelBytes: api.embeddedModelSize(),
            initializationMs: Math.round(now() - started),
        });
    })().catch((error) => {
        postMessage({ type: "error", message: String(error) });
        throw error;
    });

    return {
        ready,
        async handle(data) {
            try {
                await ready;
                if (data.type === "clear") {
                    api.clearHistory();
                    postMessage({ type: "cleared" });
                    return;
                }
                if (data.type === "chat") {
                    const output = await api.chat(data.message);
                    postMessage({ type: "response", output });
                }
            } catch (error) {
                postMessage({ type: "error", message: String(error) });
            }
        },
    };
}

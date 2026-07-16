import assert from "node:assert/strict";
import test from "node:test";

import { createWorkerRuntime } from "./worker-runtime.js";

test("worker runtime initializes, chats, and clears", async () => {
    const events = [];
    const calls = [];
    const runtime = createWorkerRuntime(
        {
            async init() {
                calls.push("init");
            },
            initialize() {
                calls.push("initialize");
            },
            embeddedModelSize() {
                return 272_696_282;
            },
            async chat(message) {
                calls.push(["chat", message]);
                return `local:${message}`;
            },
            clearHistory() {
                calls.push("clear");
            },
        },
        (event) => events.push(event),
        (() => {
            let time = 100;
            return () => (time += 25);
        })(),
    );

    await runtime.ready;
    await runtime.handle({ type: "chat", message: "hello" });
    await runtime.handle({ type: "clear" });

    assert.deepEqual(calls, ["init", "initialize", ["chat", "hello"], "clear"]);
    assert.deepEqual(events, [
        { type: "ready", modelBytes: 272_696_282, initializationMs: 25 },
        { type: "response", output: "local:hello" },
        { type: "cleared" },
    ]);
});

test("worker runtime reports chat failures", async () => {
    const events = [];
    const runtime = createWorkerRuntime(
        {
            async init() {},
            initialize() {},
            embeddedModelSize: () => 1,
            async chat() {
                throw new Error("generation failed");
            },
            clearHistory() {},
        },
        (event) => events.push(event),
        () => 0,
    );

    await runtime.ready;
    await runtime.handle({ type: "chat", message: "hello" });
    assert.equal(events.at(-1)?.type, "error");
    assert.match(events.at(-1)?.message, /generation failed/);
});

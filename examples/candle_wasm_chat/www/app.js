const transcript = document.querySelector("#transcript");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const send = document.querySelector("#send");
const clear = document.querySelector("#clear");
const status = document.querySelector("#status");
const worker = new Worker("./worker.js", { type: "module" });

let ready = false;

function setBusy(busy) {
    send.disabled = busy || !ready;
    input.disabled = busy || !ready;
    clear.disabled = busy || !ready;
}

function addMessage(role, text) {
    const article = document.createElement("article");
    article.className = `message ${role}`;
    const label = document.createElement("strong");
    label.textContent = role === "user" ? "You" : "Local model";
    const body = document.createElement("p");
    body.textContent = text;
    article.append(label, body);
    transcript.append(article);
    article.scrollIntoView({ behavior: "smooth", block: "end" });
}

worker.onmessage = ({ data }) => {
    if (data.type === "ready") {
        ready = true;
        const megabytes = (data.modelBytes / 1_000_000).toFixed(1);
        status.textContent = `Ready · ${megabytes} MB embedded · initialized in ${data.initializationMs} ms`;
        setBusy(false);
        input.focus();
        return;
    }
    if (data.type === "response") {
        addMessage("assistant", data.output || "(No text generated)");
        status.textContent = "Ready";
        setBusy(false);
        input.focus();
        return;
    }
    if (data.type === "cleared") {
        transcript.replaceChildren();
        addMessage("assistant", "Conversation cleared. Send another message.");
        status.textContent = "Ready";
        setBusy(false);
        return;
    }
    if (data.type === "error") {
        addMessage("assistant", `Error: ${data.message}`);
        status.textContent = "Error";
        setBusy(false);
    }
};

worker.onerror = ({ message }) => {
    addMessage("assistant", `Worker error: ${message}`);
    status.textContent = "Error";
    setBusy(false);
};

form.addEventListener("submit", (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message || !ready) {
        return;
    }
    addMessage("user", message);
    input.value = "";
    status.textContent = "Generating locally…";
    setBusy(true);
    worker.postMessage({ type: "chat", message });
});

clear.addEventListener("click", () => {
    status.textContent = "Clearing…";
    setBusy(true);
    worker.postMessage({ type: "clear" });
});

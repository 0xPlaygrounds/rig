
let imports = {};
imports['__wbindgen_placeholder__'] = module.exports;
let wasm;
const { TextEncoder, TextDecoder } = require(`util`);

let WASM_VECTOR_LEN = 0;

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextEncoder = new TextEncoder('utf-8');

const encodeString = (typeof cachedTextEncoder.encodeInto === 'function'
    ? function (arg, view) {
    return cachedTextEncoder.encodeInto(arg, view);
}
    : function (arg, view) {
    const buf = cachedTextEncoder.encode(arg);
    view.set(buf);
    return {
        read: arg.length,
        written: buf.length
    };
});

function passStringToWasm0(arg, malloc, realloc) {

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = encodeString(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedDataViewMemory0 = null;

function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_export_4.set(idx, obj);
    return idx;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => {
    wasm.__wbindgen_export_6.get(state.dtor)(state.a, state.b)
});

function makeMutClosure(arg0, arg1, dtor, f) {
    const state = { a: arg0, b: arg1, cnt: 1, dtor };
    const real = (...args) => {
        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            if (--state.cnt === 0) {
                wasm.__wbindgen_export_6.get(state.dtor)(a, state.b);
                CLOSURE_DTORS.unregister(state);
            } else {
                state.a = a;
            }
        }
    };
    real.original = state;
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_export_4.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

module.exports.initPanicHook = function() {
    wasm.initPanicHook();
};

function passArrayJsValueToWasm0(array, malloc) {
    const ptr = malloc(array.length * 4, 4) >>> 0;
    for (let i = 0; i < array.length; i++) {
        const add = addToExternrefTable0(array[i]);
        getDataViewMemory0().setUint32(ptr + 4 * i, add, true);
    }
    WASM_VECTOR_LEN = array.length;
    return ptr;
}
function __wbg_adapter_54(arg0, arg1) {
    wasm._dyn_core__ops__function__FnMut_____Output___R_as_wasm_bindgen__closure__WasmClosure___describe__invoke__h5e2bc3d42a24471a(arg0, arg1);
}

function __wbg_adapter_57(arg0, arg1, arg2) {
    wasm.closure383_externref_shim(arg0, arg1, arg2);
}

function __wbg_adapter_242(arg0, arg1, arg2, arg3) {
    wasm.closure405_externref_shim(arg0, arg1, arg2, arg3);
}

/**
 * Configuration options for Cloudflare's image optimization feature:
 * <https://blog.cloudflare.com/introducing-polish-automatic-image-optimizati/>
 * @enum {0 | 1 | 2}
 */
module.exports.PolishConfig = Object.freeze({
    Off: 0, "0": "Off",
    Lossy: 1, "1": "Lossy",
    Lossless: 2, "2": "Lossless",
});
/**
 * @enum {0 | 1 | 2}
 */
module.exports.RequestRedirect = Object.freeze({
    Error: 0, "0": "Error",
    Follow: 1, "1": "Follow",
    Manual: 2, "2": "Manual",
});

const __wbindgen_enum_ReadableStreamType = ["bytes"];

const __wbindgen_enum_RequestCredentials = ["omit", "same-origin", "include"];

const __wbindgen_enum_RequestMode = ["same-origin", "no-cors", "cors", "navigate"];

const AssistantContentFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_assistantcontent_free(ptr >>> 0, 1));

class AssistantContent {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(AssistantContent.prototype);
        obj.__wbg_ptr = ptr;
        AssistantContentFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        AssistantContentFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_assistantcontent_free(ptr, 0);
    }
    /**
     * @param {string} text
     * @returns {AssistantContent}
     */
    static text(text) {
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.assistantcontent_text(ptr0, len0);
        return AssistantContent.__wrap(ret);
    }
    /**
     * @param {string} id
     * @param {ToolFunction} _function
     * @returns {AssistantContent}
     */
    static tool_call(id, _function) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        _assertClass(_function, ToolFunction);
        var ptr1 = _function.__destroy_into_raw();
        const ret = wasm.assistantcontent_tool_call(ptr0, len0, ptr1);
        return AssistantContent.__wrap(ret);
    }
    /**
     * @param {string} id
     * @param {string} call_id
     * @param {ToolFunction} _function
     * @returns {AssistantContent}
     */
    static tool_call_with_call_id(id, call_id, _function) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(call_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        _assertClass(_function, ToolFunction);
        var ptr2 = _function.__destroy_into_raw();
        const ret = wasm.assistantcontent_tool_call_with_call_id(ptr0, len0, ptr1, len1, ptr2);
        return AssistantContent.__wrap(ret);
    }
}
module.exports.AssistantContent = AssistantContent;

const DocumentFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_document_free(ptr >>> 0, 1));

class Document {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Document.prototype);
        obj.__wbg_ptr = ptr;
        DocumentFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    static __unwrap(jsValue) {
        if (!(jsValue instanceof Document)) {
            return 0;
        }
        return jsValue.__destroy_into_raw();
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DocumentFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_document_free(ptr, 0);
    }
    /**
     * @param {string} id
     * @param {string} text
     */
    constructor(id, text) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.document_new(ptr0, len0, ptr1, len1);
        this.__wbg_ptr = ret >>> 0;
        DocumentFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} additional_props
     * @returns {Document}
     */
    setAdditionalProps(additional_props) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.document_setAdditionalProps(ptr, additional_props);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return Document.__wrap(ret[0]);
    }
}
module.exports.Document = Document;

const IntoUnderlyingByteSourceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_intounderlyingbytesource_free(ptr >>> 0, 1));

class IntoUnderlyingByteSource {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IntoUnderlyingByteSourceFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_intounderlyingbytesource_free(ptr, 0);
    }
    /**
     * @returns {ReadableStreamType}
     */
    get type() {
        const ret = wasm.intounderlyingbytesource_type(this.__wbg_ptr);
        return __wbindgen_enum_ReadableStreamType[ret];
    }
    /**
     * @returns {number}
     */
    get autoAllocateChunkSize() {
        const ret = wasm.intounderlyingbytesource_autoAllocateChunkSize(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {ReadableByteStreamController} controller
     */
    start(controller) {
        wasm.intounderlyingbytesource_start(this.__wbg_ptr, controller);
    }
    /**
     * @param {ReadableByteStreamController} controller
     * @returns {Promise<any>}
     */
    pull(controller) {
        const ret = wasm.intounderlyingbytesource_pull(this.__wbg_ptr, controller);
        return ret;
    }
    cancel() {
        const ptr = this.__destroy_into_raw();
        wasm.intounderlyingbytesource_cancel(ptr);
    }
}
module.exports.IntoUnderlyingByteSource = IntoUnderlyingByteSource;

const IntoUnderlyingSinkFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_intounderlyingsink_free(ptr >>> 0, 1));

class IntoUnderlyingSink {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IntoUnderlyingSinkFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_intounderlyingsink_free(ptr, 0);
    }
    /**
     * @param {any} chunk
     * @returns {Promise<any>}
     */
    write(chunk) {
        const ret = wasm.intounderlyingsink_write(this.__wbg_ptr, chunk);
        return ret;
    }
    /**
     * @returns {Promise<any>}
     */
    close() {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.intounderlyingsink_close(ptr);
        return ret;
    }
    /**
     * @param {any} reason
     * @returns {Promise<any>}
     */
    abort(reason) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.intounderlyingsink_abort(ptr, reason);
        return ret;
    }
}
module.exports.IntoUnderlyingSink = IntoUnderlyingSink;

const IntoUnderlyingSourceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_intounderlyingsource_free(ptr >>> 0, 1));

class IntoUnderlyingSource {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IntoUnderlyingSourceFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_intounderlyingsource_free(ptr, 0);
    }
    /**
     * @param {ReadableStreamDefaultController} controller
     * @returns {Promise<any>}
     */
    pull(controller) {
        const ret = wasm.intounderlyingsource_pull(this.__wbg_ptr, controller);
        return ret;
    }
    cancel() {
        const ptr = this.__destroy_into_raw();
        wasm.intounderlyingsource_cancel(ptr);
    }
}
module.exports.IntoUnderlyingSource = IntoUnderlyingSource;

const JsToolFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jstool_free(ptr >>> 0, 1));
/**
 * A tool that uses JavaScript.
 * Unfortunately, JavaScript functions are *mut u8 at their core (when it comes to how they're typed in Rust).
 * This means that we need to use `send_wrapper::SendWrapper` which automatically makes it Send.
 * However, if it gets dropped from outside of the thread where it was created, it will panic.
 */
class JsTool {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(JsTool.prototype);
        obj.__wbg_ptr = ptr;
        JsToolFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JsToolFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jstool_free(ptr, 0);
    }
    /**
     * @param {JsToolObject} tool
     * @returns {JsTool}
     */
    static new(tool) {
        const ret = wasm.jstool_new(tool);
        return JsTool.__wrap(ret);
    }
}
module.exports.JsTool = JsTool;

const MessageFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_message_free(ptr >>> 0, 1));

class Message {

    static __unwrap(jsValue) {
        if (!(jsValue instanceof Message)) {
            return 0;
        }
        return jsValue.__destroy_into_raw();
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MessageFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_message_free(ptr, 0);
    }
}
module.exports.Message = Message;

const MinifyConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_minifyconfig_free(ptr >>> 0, 1));
/**
 * Configuration options for Cloudflare's minification features:
 * <https://www.cloudflare.com/website-optimization/>
 */
class MinifyConfig {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MinifyConfigFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_minifyconfig_free(ptr, 0);
    }
    /**
     * @returns {boolean}
     */
    get js() {
        const ret = wasm.__wbg_get_minifyconfig_js(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {boolean} arg0
     */
    set js(arg0) {
        wasm.__wbg_set_minifyconfig_js(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {boolean}
     */
    get html() {
        const ret = wasm.__wbg_get_minifyconfig_html(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {boolean} arg0
     */
    set html(arg0) {
        wasm.__wbg_set_minifyconfig_html(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {boolean}
     */
    get css() {
        const ret = wasm.__wbg_get_minifyconfig_css(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {boolean} arg0
     */
    set css(arg0) {
        wasm.__wbg_set_minifyconfig_css(this.__wbg_ptr, arg0);
    }
}
module.exports.MinifyConfig = MinifyConfig;

const OpenAIAgentFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_openaiagent_free(ptr >>> 0, 1));

class OpenAIAgent {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(OpenAIAgent.prototype);
        obj.__wbg_ptr = ptr;
        OpenAIAgentFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        OpenAIAgentFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_openaiagent_free(ptr, 0);
    }
    /**
     * @param {string} prompt
     * @returns {Promise<string>}
     */
    prompt(prompt) {
        const ptr0 = passStringToWasm0(prompt, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.openaiagent_prompt(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * @param {string} prompt
     * @param {number} turns
     * @returns {Promise<string>}
     */
    prompt_multi_turn(prompt, turns) {
        const ptr0 = passStringToWasm0(prompt, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.openaiagent_prompt_multi_turn(this.__wbg_ptr, ptr0, len0, turns);
        return ret;
    }
    /**
     * @param {string} prompt
     * @param {Message[]} messages
     * @returns {Promise<string>}
     */
    chat(prompt, messages) {
        const ptr0 = passStringToWasm0(prompt, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayJsValueToWasm0(messages, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.openaiagent_chat(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        return ret;
    }
}
module.exports.OpenAIAgent = OpenAIAgent;

const OpenAIAgentBuilderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_openaiagentbuilder_free(ptr >>> 0, 1));

class OpenAIAgentBuilder {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(OpenAIAgentBuilder.prototype);
        obj.__wbg_ptr = ptr;
        OpenAIAgentBuilderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        OpenAIAgentBuilderFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_openaiagentbuilder_free(ptr, 0);
    }
    /**
     * @param {OpenAIClient} client
     * @param {string} model_name
     */
    constructor(client, model_name) {
        _assertClass(client, OpenAIClient);
        const ptr0 = passStringToWasm0(model_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.openaiagentbuilder_new(client.__wbg_ptr, ptr0, len0);
        this.__wbg_ptr = ret >>> 0;
        OpenAIAgentBuilderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {string} preamble
     * @returns {OpenAIAgentBuilder}
     */
    setPreamble(preamble) {
        const ptr = this.__destroy_into_raw();
        const ptr0 = passStringToWasm0(preamble, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.openaiagentbuilder_setPreamble(ptr, ptr0, len0);
        return OpenAIAgentBuilder.__wrap(ret);
    }
    /**
     * @param {JsToolObject} tool
     * @returns {OpenAIAgentBuilder}
     */
    addTool(tool) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.openaiagentbuilder_addTool(ptr, tool);
        return OpenAIAgentBuilder.__wrap(ret);
    }
    /**
     * @returns {OpenAIAgent}
     */
    build() {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.openaiagentbuilder_build(ptr);
        return OpenAIAgent.__wrap(ret);
    }
}
module.exports.OpenAIAgentBuilder = OpenAIAgentBuilder;

const OpenAIClientFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_openaiclient_free(ptr >>> 0, 1));

class OpenAIClient {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(OpenAIClient.prototype);
        obj.__wbg_ptr = ptr;
        OpenAIClientFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        OpenAIClientFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_openaiclient_free(ptr, 0);
    }
    /**
     * @param {string} api_key
     */
    constructor(api_key) {
        const ptr0 = passStringToWasm0(api_key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.openaiclient_new(ptr0, len0);
        this.__wbg_ptr = ret >>> 0;
        OpenAIClientFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {string} api_key
     * @param {string} base_url
     * @returns {OpenAIClient}
     */
    static from_url(api_key, base_url) {
        const ptr0 = passStringToWasm0(api_key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(base_url, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.openaiclient_from_url(ptr0, len0, ptr1, len1);
        return OpenAIClient.__wrap(ret);
    }
    /**
     * @param {string} model_name
     * @returns {OpenAICompletionModel}
     */
    completion_model(model_name) {
        const ptr0 = passStringToWasm0(model_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.openaiclient_completion_model(this.__wbg_ptr, ptr0, len0);
        return OpenAICompletionModel.__wrap(ret);
    }
    /**
     * @param {string} model_name
     * @returns {OpenAIAgentBuilder}
     */
    agent(model_name) {
        const ptr0 = passStringToWasm0(model_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.openaiagentbuilder_new(this.__wbg_ptr, ptr0, len0);
        return OpenAIAgentBuilder.__wrap(ret);
    }
}
module.exports.OpenAIClient = OpenAIClient;

const OpenAICompletionModelFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_openaicompletionmodel_free(ptr >>> 0, 1));

class OpenAICompletionModel {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(OpenAICompletionModel.prototype);
        obj.__wbg_ptr = ptr;
        OpenAICompletionModelFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        OpenAICompletionModelFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_openaicompletionmodel_free(ptr, 0);
    }
    /**
     * @param {OpenAIClient} client
     * @param {string} model_name
     */
    constructor(client, model_name) {
        _assertClass(client, OpenAIClient);
        const ptr0 = passStringToWasm0(model_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.openaiclient_completion_model(client.__wbg_ptr, ptr0, len0);
        this.__wbg_ptr = ret >>> 0;
        OpenAICompletionModelFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
module.exports.OpenAICompletionModel = OpenAICompletionModel;

const OpenAICompletionRequestFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_openaicompletionrequest_free(ptr >>> 0, 1));

class OpenAICompletionRequest {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(OpenAICompletionRequest.prototype);
        obj.__wbg_ptr = ptr;
        OpenAICompletionRequestFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        OpenAICompletionRequestFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_openaicompletionrequest_free(ptr, 0);
    }
    /**
     * @param {OpenAICompletionModel} model
     * @param {Message} prompt
     */
    constructor(model, prompt) {
        _assertClass(model, OpenAICompletionModel);
        var ptr0 = model.__destroy_into_raw();
        _assertClass(prompt, Message);
        var ptr1 = prompt.__destroy_into_raw();
        const ret = wasm.openaicompletionrequest_new(ptr0, ptr1);
        this.__wbg_ptr = ret >>> 0;
        OpenAICompletionRequestFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {string} preamble
     * @returns {OpenAICompletionRequest}
     */
    setPreamble(preamble) {
        const ptr = this.__destroy_into_raw();
        const ptr0 = passStringToWasm0(preamble, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.openaicompletionrequest_setPreamble(ptr, ptr0, len0);
        return OpenAICompletionRequest.__wrap(ret);
    }
    /**
     * @param {Message[]} chat_history
     * @returns {OpenAICompletionRequest}
     */
    setChatHistory(chat_history) {
        const ptr = this.__destroy_into_raw();
        const ptr0 = passArrayJsValueToWasm0(chat_history, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.openaicompletionrequest_setChatHistory(ptr, ptr0, len0);
        return OpenAICompletionRequest.__wrap(ret);
    }
    /**
     * @param {Document[]} documents
     * @returns {OpenAICompletionRequest}
     */
    setDocuments(documents) {
        const ptr = this.__destroy_into_raw();
        const ptr0 = passArrayJsValueToWasm0(documents, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.openaicompletionrequest_setDocuments(ptr, ptr0, len0);
        return OpenAICompletionRequest.__wrap(ret);
    }
    /**
     * @param {ToolDefinition[]} tools
     * @returns {OpenAICompletionRequest}
     */
    setTools(tools) {
        const ptr = this.__destroy_into_raw();
        const ptr0 = passArrayJsValueToWasm0(tools, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.openaicompletionrequest_setTools(ptr, ptr0, len0);
        return OpenAICompletionRequest.__wrap(ret);
    }
    /**
     * @param {number} temperature
     * @returns {OpenAICompletionRequest}
     */
    setTemperature(temperature) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.openaicompletionrequest_setTemperature(ptr, temperature);
        return OpenAICompletionRequest.__wrap(ret);
    }
    /**
     * @param {bigint} max_tokens
     * @returns {OpenAICompletionRequest}
     */
    setMaxTokens(max_tokens) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.openaicompletionrequest_setMaxTokens(ptr, max_tokens);
        return OpenAICompletionRequest.__wrap(ret);
    }
    /**
     * @param {any} obj
     * @returns {OpenAICompletionRequest}
     */
    setAdditionalParams(obj) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.openaicompletionrequest_setAdditionalParams(ptr, obj);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return OpenAICompletionRequest.__wrap(ret[0]);
    }
    /**
     * @returns {Promise<AssistantContent[]>}
     */
    send() {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.openaicompletionrequest_send(ptr);
        return ret;
    }
}
module.exports.OpenAICompletionRequest = OpenAICompletionRequest;

const R2RangeFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_r2range_free(ptr >>> 0, 1));

class R2Range {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        R2RangeFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_r2range_free(ptr, 0);
    }
    /**
     * @returns {number | undefined}
     */
    get offset() {
        const ret = wasm.__wbg_get_r2range_offset(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @param {number | null} [arg0]
     */
    set offset(arg0) {
        wasm.__wbg_set_r2range_offset(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? 0 : arg0);
    }
    /**
     * @returns {number | undefined}
     */
    get length() {
        const ret = wasm.__wbg_get_r2range_length(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @param {number | null} [arg0]
     */
    set length(arg0) {
        wasm.__wbg_set_r2range_length(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? 0 : arg0);
    }
    /**
     * @returns {number | undefined}
     */
    get suffix() {
        const ret = wasm.__wbg_get_r2range_suffix(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @param {number | null} [arg0]
     */
    set suffix(arg0) {
        wasm.__wbg_set_r2range_suffix(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? 0 : arg0);
    }
}
module.exports.R2Range = R2Range;

const ToolDefinitionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_tooldefinition_free(ptr >>> 0, 1));

class ToolDefinition {

    static __unwrap(jsValue) {
        if (!(jsValue instanceof ToolDefinition)) {
            return 0;
        }
        return jsValue.__destroy_into_raw();
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ToolDefinitionFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_tooldefinition_free(ptr, 0);
    }
}
module.exports.ToolDefinition = ToolDefinition;

const ToolFunctionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_toolfunction_free(ptr >>> 0, 1));

class ToolFunction {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ToolFunctionFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_toolfunction_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.toolfunction_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {any}
     */
    args() {
        const ret = wasm.toolfunction_args(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
module.exports.ToolFunction = ToolFunction;

module.exports.__wbg_String_8f0eb39a4a4c2f66 = function(arg0, arg1) {
    const ret = String(arg1);
    const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
};

module.exports.__wbg_abort_410ec47a64ac6117 = function(arg0, arg1) {
    arg0.abort(arg1);
};

module.exports.__wbg_abort_775ef1d17fc65868 = function(arg0) {
    arg0.abort();
};

module.exports.__wbg_append_299d5d48292c0495 = function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
    arg0.append(getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
}, arguments) };

module.exports.__wbg_append_8c7dd8d641a5f01b = function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
    arg0.append(getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
}, arguments) };

module.exports.__wbg_append_b2d1fc16de2a0e81 = function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5) {
    arg0.append(getStringFromWasm0(arg1, arg2), arg3, getStringFromWasm0(arg4, arg5));
}, arguments) };

module.exports.__wbg_append_b44785ebeb668479 = function() { return handleError(function (arg0, arg1, arg2, arg3) {
    arg0.append(getStringFromWasm0(arg1, arg2), arg3);
}, arguments) };

module.exports.__wbg_assistantcontent_new = function(arg0) {
    const ret = AssistantContent.__wrap(arg0);
    return ret;
};

module.exports.__wbg_buffer_09165b52af8c5237 = function(arg0) {
    const ret = arg0.buffer;
    return ret;
};

module.exports.__wbg_buffer_609cc3eee51ed158 = function(arg0) {
    const ret = arg0.buffer;
    return ret;
};

module.exports.__wbg_byobRequest_77d9adf63337edfb = function(arg0) {
    const ret = arg0.byobRequest;
    return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
};

module.exports.__wbg_byteLength_e674b853d9c77e1d = function(arg0) {
    const ret = arg0.byteLength;
    return ret;
};

module.exports.__wbg_byteOffset_fd862df290ef848d = function(arg0) {
    const ret = arg0.byteOffset;
    return ret;
};

module.exports.__wbg_call_1d435e50dda5a7d0 = function(arg0, arg1) {
    const ret = arg0.call(arg1);
    return ret;
};

module.exports.__wbg_call_672a4d21634d4a24 = function() { return handleError(function (arg0, arg1) {
    const ret = arg0.call(arg1);
    return ret;
}, arguments) };

module.exports.__wbg_call_7cccdd69e0791ae2 = function() { return handleError(function (arg0, arg1, arg2) {
    const ret = arg0.call(arg1, arg2);
    return ret;
}, arguments) };

module.exports.__wbg_clearTimeout_b1115618e821c3b2 = function(arg0) {
    const ret = clearTimeout(arg0);
    return ret;
};

module.exports.__wbg_close_304cc1fef3466669 = function() { return handleError(function (arg0) {
    arg0.close();
}, arguments) };

module.exports.__wbg_close_5ce03e29be453811 = function() { return handleError(function (arg0) {
    arg0.close();
}, arguments) };

module.exports.__wbg_definition_fa528d6711afb910 = function(arg0, arg1, arg2) {
    let deferred0_0;
    let deferred0_1;
    try {
        deferred0_0 = arg1;
        deferred0_1 = arg2;
        const ret = arg0.definition(getStringFromWasm0(arg1, arg2));
        return ret;
    } finally {
        wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
    }
};

module.exports.__wbg_document_unwrap = function(arg0) {
    const ret = Document.__unwrap(arg0);
    return ret;
};

module.exports.__wbg_done_769e5ede4b31c67b = function(arg0) {
    const ret = arg0.done;
    return ret;
};

module.exports.__wbg_enqueue_bb16ba72f537dc9e = function() { return handleError(function (arg0, arg1) {
    arg0.enqueue(arg1);
}, arguments) };

module.exports.__wbg_entries_3265d4158b33e5dc = function(arg0) {
    const ret = Object.entries(arg0);
    return ret;
};

module.exports.__wbg_error_7534b8e9a36f1ab4 = function(arg0, arg1) {
    let deferred0_0;
    let deferred0_1;
    try {
        deferred0_0 = arg0;
        deferred0_1 = arg1;
        console.error(getStringFromWasm0(arg0, arg1));
    } finally {
        wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
    }
};

module.exports.__wbg_fetch_3afbdcc7ddbf16fe = function(arg0) {
    const ret = fetch(arg0);
    return ret;
};

module.exports.__wbg_fetch_509096533071c657 = function(arg0, arg1) {
    const ret = arg0.fetch(arg1);
    return ret;
};

module.exports.__wbg_get_67b2ba62fc30de12 = function() { return handleError(function (arg0, arg1) {
    const ret = Reflect.get(arg0, arg1);
    return ret;
}, arguments) };

module.exports.__wbg_get_b9b93047fe3cf45b = function(arg0, arg1) {
    const ret = arg0[arg1 >>> 0];
    return ret;
};

module.exports.__wbg_getwithrefkey_1dc361bd10053bfe = function(arg0, arg1) {
    const ret = arg0[arg1];
    return ret;
};

module.exports.__wbg_has_a5ea9117f258a0ec = function() { return handleError(function (arg0, arg1) {
    const ret = Reflect.has(arg0, arg1);
    return ret;
}, arguments) };

module.exports.__wbg_headers_9cb51cfd2ac780a4 = function(arg0) {
    const ret = arg0.headers;
    return ret;
};

module.exports.__wbg_instanceof_ArrayBuffer_e14585432e3737fc = function(arg0) {
    let result;
    try {
        result = arg0 instanceof ArrayBuffer;
    } catch (_) {
        result = false;
    }
    const ret = result;
    return ret;
};

module.exports.__wbg_instanceof_Map_f3469ce2244d2430 = function(arg0) {
    let result;
    try {
        result = arg0 instanceof Map;
    } catch (_) {
        result = false;
    }
    const ret = result;
    return ret;
};

module.exports.__wbg_instanceof_Promise_935168b8f4b49db3 = function(arg0) {
    let result;
    try {
        result = arg0 instanceof Promise;
    } catch (_) {
        result = false;
    }
    const ret = result;
    return ret;
};

module.exports.__wbg_instanceof_Response_f2cc20d9f7dfd644 = function(arg0) {
    let result;
    try {
        result = arg0 instanceof Response;
    } catch (_) {
        result = false;
    }
    const ret = result;
    return ret;
};

module.exports.__wbg_instanceof_Uint8Array_17156bcf118086a9 = function(arg0) {
    let result;
    try {
        result = arg0 instanceof Uint8Array;
    } catch (_) {
        result = false;
    }
    const ret = result;
    return ret;
};

module.exports.__wbg_isArray_a1eab7e0d067391b = function(arg0) {
    const ret = Array.isArray(arg0);
    return ret;
};

module.exports.__wbg_isSafeInteger_343e2beeeece1bb0 = function(arg0) {
    const ret = Number.isSafeInteger(arg0);
    return ret;
};

module.exports.__wbg_iterator_9a24c88df860dc65 = function() {
    const ret = Symbol.iterator;
    return ret;
};

module.exports.__wbg_length_a446193dc22c12f8 = function(arg0) {
    const ret = arg0.length;
    return ret;
};

module.exports.__wbg_length_e2d2a49132c1b256 = function(arg0) {
    const ret = arg0.length;
    return ret;
};

module.exports.__wbg_message_unwrap = function(arg0) {
    const ret = Message.__unwrap(arg0);
    return ret;
};

module.exports.__wbg_name_0bd482aa8cee7277 = function(arg0, arg1) {
    const ret = arg1.name();
    const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
};

module.exports.__wbg_new_018dcc2d6c8c2f6a = function() { return handleError(function () {
    const ret = new Headers();
    return ret;
}, arguments) };

module.exports.__wbg_new_23a2665fac83c611 = function(arg0, arg1) {
    try {
        var state0 = {a: arg0, b: arg1};
        var cb0 = (arg0, arg1) => {
            const a = state0.a;
            state0.a = 0;
            try {
                return __wbg_adapter_242(a, state0.b, arg0, arg1);
            } finally {
                state0.a = a;
            }
        };
        const ret = new Promise(cb0);
        return ret;
    } finally {
        state0.a = state0.b = 0;
    }
};

module.exports.__wbg_new_405e22f390576ce2 = function() {
    const ret = new Object();
    return ret;
};

module.exports.__wbg_new_5e0be73521bc8c17 = function() {
    const ret = new Map();
    return ret;
};

module.exports.__wbg_new_78feb108b6472713 = function() {
    const ret = new Array();
    return ret;
};

module.exports.__wbg_new_8a6f238a6ece86ea = function() {
    const ret = new Error();
    return ret;
};

module.exports.__wbg_new_9fd39a253424609a = function() { return handleError(function () {
    const ret = new FormData();
    return ret;
}, arguments) };

module.exports.__wbg_new_a12002a7f91c75be = function(arg0) {
    const ret = new Uint8Array(arg0);
    return ret;
};

module.exports.__wbg_new_c68d7209be747379 = function(arg0, arg1) {
    const ret = new Error(getStringFromWasm0(arg0, arg1));
    return ret;
};

module.exports.__wbg_new_e25e5aab09ff45db = function() { return handleError(function () {
    const ret = new AbortController();
    return ret;
}, arguments) };

module.exports.__wbg_newnoargs_105ed471475aaf50 = function(arg0, arg1) {
    const ret = new Function(getStringFromWasm0(arg0, arg1));
    return ret;
};

module.exports.__wbg_newwithbyteoffsetandlength_d97e637ebe145a9a = function(arg0, arg1, arg2) {
    const ret = new Uint8Array(arg0, arg1 >>> 0, arg2 >>> 0);
    return ret;
};

module.exports.__wbg_newwithstrandinit_06c535e0a867c635 = function() { return handleError(function (arg0, arg1, arg2) {
    const ret = new Request(getStringFromWasm0(arg0, arg1), arg2);
    return ret;
}, arguments) };

module.exports.__wbg_newwithu8arraysequenceandoptions_068570c487f69127 = function() { return handleError(function (arg0, arg1) {
    const ret = new Blob(arg0, arg1);
    return ret;
}, arguments) };

module.exports.__wbg_next_25feadfc0913fea9 = function(arg0) {
    const ret = arg0.next;
    return ret;
};

module.exports.__wbg_next_6574e1a8a62d1055 = function() { return handleError(function (arg0) {
    const ret = arg0.next();
    return ret;
}, arguments) };

module.exports.__wbg_push_737cfc8c1432c2c6 = function(arg0, arg1) {
    const ret = arg0.push(arg1);
    return ret;
};

module.exports.__wbg_queueMicrotask_97d92b4fcc8a61c5 = function(arg0) {
    queueMicrotask(arg0);
};

module.exports.__wbg_queueMicrotask_d3219def82552485 = function(arg0) {
    const ret = arg0.queueMicrotask;
    return ret;
};

module.exports.__wbg_resolve_4851785c9c5f573d = function(arg0) {
    const ret = Promise.resolve(arg0);
    return ret;
};

module.exports.__wbg_respond_1f279fa9f8edcb1c = function() { return handleError(function (arg0, arg1) {
    arg0.respond(arg1 >>> 0);
}, arguments) };

module.exports.__wbg_setTimeout_ca12ead8b48245e2 = function(arg0, arg1) {
    const ret = setTimeout(arg0, arg1);
    return ret;
};

module.exports.__wbg_set_37837023f3d740e8 = function(arg0, arg1, arg2) {
    arg0[arg1 >>> 0] = arg2;
};

module.exports.__wbg_set_3f1d0b984ed272ed = function(arg0, arg1, arg2) {
    arg0[arg1] = arg2;
};

module.exports.__wbg_set_65595bdd868b3009 = function(arg0, arg1, arg2) {
    arg0.set(arg1, arg2 >>> 0);
};

module.exports.__wbg_set_8fc6bf8a5b1071d1 = function(arg0, arg1, arg2) {
    const ret = arg0.set(arg1, arg2);
    return ret;
};

module.exports.__wbg_setbody_5923b78a95eedf29 = function(arg0, arg1) {
    arg0.body = arg1;
};

module.exports.__wbg_setcredentials_c3a22f1cd105a2c6 = function(arg0, arg1) {
    arg0.credentials = __wbindgen_enum_RequestCredentials[arg1];
};

module.exports.__wbg_setheaders_834c0bdb6a8949ad = function(arg0, arg1) {
    arg0.headers = arg1;
};

module.exports.__wbg_setmethod_3c5280fe5d890842 = function(arg0, arg1, arg2) {
    arg0.method = getStringFromWasm0(arg1, arg2);
};

module.exports.__wbg_setmode_5dc300b865044b65 = function(arg0, arg1) {
    arg0.mode = __wbindgen_enum_RequestMode[arg1];
};

module.exports.__wbg_setsignal_75b21ef3a81de905 = function(arg0, arg1) {
    arg0.signal = arg1;
};

module.exports.__wbg_settype_39ed370d3edd403c = function(arg0, arg1, arg2) {
    arg0.type = getStringFromWasm0(arg1, arg2);
};

module.exports.__wbg_signal_aaf9ad74119f20a4 = function(arg0) {
    const ret = arg0.signal;
    return ret;
};

module.exports.__wbg_stack_0ed75d68575b0f3c = function(arg0, arg1) {
    const ret = arg1.stack;
    const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
};

module.exports.__wbg_static_accessor_GLOBAL_88a902d13a557d07 = function() {
    const ret = typeof global === 'undefined' ? null : global;
    return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
};

module.exports.__wbg_static_accessor_GLOBAL_THIS_56578be7e9f832b0 = function() {
    const ret = typeof globalThis === 'undefined' ? null : globalThis;
    return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
};

module.exports.__wbg_static_accessor_SELF_37c5d418e4bf5819 = function() {
    const ret = typeof self === 'undefined' ? null : self;
    return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
};

module.exports.__wbg_static_accessor_WINDOW_5de37043a91a9c40 = function() {
    const ret = typeof window === 'undefined' ? null : window;
    return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
};

module.exports.__wbg_status_f6360336ca686bf0 = function(arg0) {
    const ret = arg0.status;
    return ret;
};

module.exports.__wbg_stringify_f7ed6987935b4a24 = function() { return handleError(function (arg0) {
    const ret = JSON.stringify(arg0);
    return ret;
}, arguments) };

module.exports.__wbg_text_7805bea50de2af49 = function() { return handleError(function (arg0) {
    const ret = arg0.text();
    return ret;
}, arguments) };

module.exports.__wbg_then_44b73946d2fb3e7d = function(arg0, arg1) {
    const ret = arg0.then(arg1);
    return ret;
};

module.exports.__wbg_then_48b406749878a531 = function(arg0, arg1, arg2) {
    const ret = arg0.then(arg1, arg2);
    return ret;
};

module.exports.__wbg_tooldefinition_unwrap = function(arg0) {
    const ret = ToolDefinition.__unwrap(arg0);
    return ret;
};

module.exports.__wbg_url_ae10c34ca209681d = function(arg0, arg1) {
    const ret = arg1.url;
    const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
};

module.exports.__wbg_value_cd1ffa7b1ab794f1 = function(arg0) {
    const ret = arg0.value;
    return ret;
};

module.exports.__wbg_view_fd8a56e8983f448d = function(arg0) {
    const ret = arg0.view;
    return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
};

module.exports.__wbindgen_array_new = function() {
    const ret = [];
    return ret;
};

module.exports.__wbindgen_array_push = function(arg0, arg1) {
    arg0.push(arg1);
};

module.exports.__wbindgen_bigint_from_i64 = function(arg0) {
    const ret = arg0;
    return ret;
};

module.exports.__wbindgen_bigint_from_u64 = function(arg0) {
    const ret = BigInt.asUintN(64, arg0);
    return ret;
};

module.exports.__wbindgen_bigint_get_as_i64 = function(arg0, arg1) {
    const v = arg1;
    const ret = typeof(v) === 'bigint' ? v : undefined;
    getDataViewMemory0().setBigInt64(arg0 + 8 * 1, isLikeNone(ret) ? BigInt(0) : ret, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
};

module.exports.__wbindgen_boolean_get = function(arg0) {
    const v = arg0;
    const ret = typeof(v) === 'boolean' ? (v ? 1 : 0) : 2;
    return ret;
};

module.exports.__wbindgen_cb_drop = function(arg0) {
    const obj = arg0.original;
    if (obj.cnt-- == 1) {
        obj.a = 0;
        return true;
    }
    const ret = false;
    return ret;
};

module.exports.__wbindgen_closure_wrapper1263 = function(arg0, arg1, arg2) {
    const ret = makeMutClosure(arg0, arg1, 339, __wbg_adapter_54);
    return ret;
};

module.exports.__wbindgen_closure_wrapper1369 = function(arg0, arg1, arg2) {
    const ret = makeMutClosure(arg0, arg1, 384, __wbg_adapter_57);
    return ret;
};

module.exports.__wbindgen_debug_string = function(arg0, arg1) {
    const ret = debugString(arg1);
    const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
};

module.exports.__wbindgen_error_new = function(arg0, arg1) {
    const ret = new Error(getStringFromWasm0(arg0, arg1));
    return ret;
};

module.exports.__wbindgen_in = function(arg0, arg1) {
    const ret = arg0 in arg1;
    return ret;
};

module.exports.__wbindgen_init_externref_table = function() {
    const table = wasm.__wbindgen_export_4;
    const offset = table.grow(4);
    table.set(0, undefined);
    table.set(offset + 0, undefined);
    table.set(offset + 1, null);
    table.set(offset + 2, true);
    table.set(offset + 3, false);
    ;
};

module.exports.__wbindgen_is_bigint = function(arg0) {
    const ret = typeof(arg0) === 'bigint';
    return ret;
};

module.exports.__wbindgen_is_function = function(arg0) {
    const ret = typeof(arg0) === 'function';
    return ret;
};

module.exports.__wbindgen_is_object = function(arg0) {
    const val = arg0;
    const ret = typeof(val) === 'object' && val !== null;
    return ret;
};

module.exports.__wbindgen_is_string = function(arg0) {
    const ret = typeof(arg0) === 'string';
    return ret;
};

module.exports.__wbindgen_is_undefined = function(arg0) {
    const ret = arg0 === undefined;
    return ret;
};

module.exports.__wbindgen_jsval_eq = function(arg0, arg1) {
    const ret = arg0 === arg1;
    return ret;
};

module.exports.__wbindgen_jsval_loose_eq = function(arg0, arg1) {
    const ret = arg0 == arg1;
    return ret;
};

module.exports.__wbindgen_memory = function() {
    const ret = wasm.memory;
    return ret;
};

module.exports.__wbindgen_number_get = function(arg0, arg1) {
    const obj = arg1;
    const ret = typeof(obj) === 'number' ? obj : undefined;
    getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
};

module.exports.__wbindgen_number_new = function(arg0) {
    const ret = arg0;
    return ret;
};

module.exports.__wbindgen_string_get = function(arg0, arg1) {
    const obj = arg1;
    const ret = typeof(obj) === 'string' ? obj : undefined;
    var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
};

module.exports.__wbindgen_string_new = function(arg0, arg1) {
    const ret = getStringFromWasm0(arg0, arg1);
    return ret;
};

module.exports.__wbindgen_throw = function(arg0, arg1) {
    throw new Error(getStringFromWasm0(arg0, arg1));
};

const path = require('path').join(__dirname, 'rig_wasm_bg.wasm');
const bytes = require('fs').readFileSync(path);

const wasmModule = new WebAssembly.Module(bytes);
const wasmInstance = new WebAssembly.Instance(wasmModule, imports);
wasm = wasmInstance.exports;
module.exports.__wasm = wasm;

wasm.__wbindgen_start();


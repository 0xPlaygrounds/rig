"""Independent JSON-RPC smoke: build --example scripted, then pass its path here."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Client:
    def __init__(self, binary, version, home):
        self.version = version
        self.queue = queue.Queue()
        self.sequence = 0
        self.events = []
        self.stderr = open(home / 'stderr', 'a')
        env = {key: value for key, value in os.environ.items() if not key.startswith('RIG_')}
        env['RIG_HOME'] = str(home)
        self.process = subprocess.Popen(
            [binary, 'acp'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr, text=True, env=env,
        )
        def read():
            for line in self.process.stdout:
                try:
                    self.queue.put(json.loads(line))
                except ValueError:
                    self.queue.put({'stray_stdout': line})
        threading.Thread(target=read, daemon=True).start()
        identity = {'name': 'independent-smoke', 'version': '1'}
        params = {'protocolVersion': version}
        params['info' if version == 2 else 'clientInfo'] = identity
        assert self.call('initialize', params)['protocolVersion'] == version

    def write(self, value):
        self.process.stdin.write(json.dumps({'jsonrpc': '2.0', **value}) + '\n')
        self.process.stdin.flush()

    def send(self, method, params):
        self.sequence += 1
        self.write({'id': self.sequence, 'method': method, 'params': params})
        return self.sequence

    def receive(self):
        value = self.queue.get(timeout=30)
        assert 'stray_stdout' not in value, value
        self.events.append(value)
        return value

    def call(self, method, params):
        request = self.send(method, params)
        while True:
            value = self.receive()
            if value.get('id') == request:
                assert 'error' not in value, value
                return value['result']

    def choose(self, session, model):
        params = {'sessionId': session, 'configId': 'model', 'value': model}
        if self.version == 2:
            params['type'] = 'id'
        result = self.call('session/set_config_option', params)
        assert any(o.get('configId', o.get('id')) == 'model' and o['currentValue'] == model for o in result['configOptions'])

    def prompt(self, session, permission=None, cancel=False, text="exercise the scripted model", error=False):
        start = len(self.events)
        request = self.send('session/prompt', {
            'sessionId': session, 'prompt': [{'type': 'text', 'text': text}],
        })
        accepted = False
        while True:
            value = self.receive()
            if value.get('method') == 'session/request_permission':
                assert permission is not None, value
                if cancel == "permission":
                    self.write({'method': 'session/cancel', 'params': {'sessionId': session}})
                    outcome = {'outcome': 'cancelled'}
                else:
                    outcome = {'outcome': 'selected', 'optionId': permission}
                self.write({'id': value['id'], 'result': {'outcome': outcome}})
            if value.get('id') == request:
                if error and self.version == 1:
                    assert 'error' in value, value
                    break
                assert 'error' not in value, value
                if self.version == 1:
                    assert value['result']['stopReason'] == ('cancelled' if cancel else 'end_turn'), value
                    break
                assert 'messageId' in value['result'], value
                accepted = True
            update = value.get('params', {}).get('update', {})
            if cancel == "active" and update.get('status') == 'in_progress':
                self.write({'method': 'session/cancel', 'params': {'sessionId': session}})
            if self.version == 2 and update.get('state') == 'idle' and 'stopReason' in update:
                assert accepted, 'v2 completed before prompt acceptance'
                assert update['stopReason'] == ('error' if error else 'cancelled' if cancel else 'end_turn'), update
                if error:
                    assert update['error']['message'], update
                break
        return self.events[start:]

    def reject_prompt(self, session, blocks):
        request = self.send('session/prompt', {'sessionId': session, 'prompt': blocks})
        while True:
            value = self.receive()
            if value.get('id') == request:
                assert 'error' in value, value
                break

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise
        self.stderr.close()
        assert self.process.returncode == 0, self.process.returncode


class Mcp(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_error(405)

    def do_POST(self):
        assert self.headers.get('Authorization') == 'Bearer fixture-only'
        message = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.calls.append(message['method'])
        if 'id' not in message:
            self.send_response(202)
            self.end_headers()
            return
        method = message['method']
        if method == 'initialize':
            result = {'protocolVersion': message['params']['protocolVersion'],
                      'capabilities': {'tools': {}}, 'serverInfo': {'name': 'fixture', 'version': '1'}}
        elif method == 'tools/list':
            result = {'tools': [{'name': 'hello', 'description': 'fixture tool',
                                 'inputSchema': {'type': 'object', 'properties': {}}}]}
        elif method == 'tools/call':
            result = {'content': [{'type': 'text', 'text': 'MCP authorization preserved'}]}
        else:
            raise AssertionError(message)
        body = json.dumps({'jsonrpc': '2.0', 'id': message['id'], 'result': result}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(binary, version, endpoint):
    with tempfile.TemporaryDirectory(prefix=f'rig-acp-v{version}-') as directory:
        home = Path(directory)
        one, two = home / 'one', home / 'two'
        one.mkdir()
        two.mkdir()
        (one / 'fixture.txt').write_text('ONLY SESSION ONE')
        (two / 'fixture.txt').write_text('ONLY SESSION TWO')
        (home / 'defaults.json').write_text(json.dumps({'model': 'test/read', 'effort': None}))
        client = Client(binary, version, home)
        try:
            created = client.call('session/new', {'cwd': str(one), 'mcpServers': []})
            model = next(o for o in created['configOptions'] if o.get('configId', o.get('id')) == 'model')
            assert model['currentValue'] == 'test/read', model
            assert model['options'][0]['value'] == 'test/read', model
            assert all(o['value'] != 'acp-unavailable/example' for o in model['options']), model
            a = created['sessionId']
            b = client.call('session/new', {'cwd': str(two), 'mcpServers': []})['sessionId']
            assert a != b
            client.choose(a, 'test/acp')
            client.reject_prompt(a, [])
            client.reject_prompt(a, [{'type': 'text', 'text': '   \n'}])
            history = home / 'acp' / a / 'events.jsonl'
            history.mkdir()
            client.reject_prompt(a, [{'type': 'text', 'text': 'history write must fail'}])
            history.rmdir()
            assert 'scripted reply' in json.dumps(client.prompt(a))
            client.write({'method': 'session/cancel', 'params': {'sessionId': a}})
            unknown = client.prompt(a, text='/nonexistent_command', error=True)
            assert 'Unknown command' in json.dumps(unknown), unknown
            local = client.prompt(a, text='/model test/acp')
            assert 'scripted reply' not in json.dumps(local), local
            assert 'scripted reply' in json.dumps(client.prompt(a))
            assert 'scripted reply' in json.dumps(client.prompt(a, text='/scripted_turn'))
            client.choose(a, 'test/failure')
            client.prompt(a)
            failed = client.prompt(a, error=True)
            if version == 2:
                errors = [e.get('params', {}).get('update', {}).get('error') for e in failed]
                assert any(e and e.get('message') != 'session failed' for e in errors), failed
            client.choose(a, 'test/acp')
            client.prompt(a)
            for session, expected, absent in [(a, 'ONLY SESSION ONE', 'ONLY SESSION TWO'),
                                              (b, 'ONLY SESSION TWO', 'ONLY SESSION ONE')]:
                client.choose(session, 'test/read')
                events = client.prompt(session)
                text = json.dumps(events)
                assert expected in text and absent not in text, text
            client.choose(a, 'test/tool')
            denied = client.prompt(a, permission='deny')
            assert any(e.get('method') == 'session/request_permission' for e in denied)
            assert 'permission resolved' in json.dumps(denied)
            updates = [e.get('params', {}).get('update', {}) for e in denied]
            tools = [u for u in updates if 'toolCallId' in u]
            assert tools[0].get('status', 'pending') == 'pending', tools
            assert tools[-1]['status'] == 'failed', tools
            assert len({t['toolCallId'] for t in tools}) == 1, tools
            if version == 1:
                assert tools[0]['sessionUpdate'] == 'tool_call', tools
                assert all(t['sessionUpdate'] == 'tool_call_update' for t in tools[1:]), tools
            else:
                states = [u.get('state') for u in updates]
                assert 'requires_action' in states and 'running' in states[states.index('requires_action') + 1:], states
            c = client.call('session/new', {'cwd': str(one), 'mcpServers': []})['sessionId']
            client.choose(c, 'test/tool')
            client.prompt(c, permission='deny', cancel='permission')
            d = client.call('session/new', {'cwd': str(one), 'mcpServers': []})['sessionId']
            client.choose(d, 'test/tool')
            active = client.prompt(d, permission='allow', cancel='active')
            assert any(e.get('params', {}).get('update', {}).get('status') == 'in_progress' for e in active)
            servers = [{'type': 'http', 'name': 'injected', 'url': endpoint,
                        'headers': [{'name': 'Authorization', 'value': 'Bearer fixture-only'}]}]
            m = client.call('session/new', {'cwd': str(one), 'mcpServers': servers})['sessionId']
            client.choose(m, 'test/mcp')
            events = client.prompt(m)
            assert 'MCP authorization preserved' in json.dumps(events), events
            client.choose(b, 'test/acp')
            client.prompt(b)
            if version == 2:
                listing = client.call('session/list', {})
                assert len(listing['sessions']) == 5
                client.call('session/close', {'sessionId': b})
                mark = len(client.events)
                client.call('session/resume', {'sessionId': b, 'cwd': str(two),
                            'mcpServers': [], 'replayFrom': {'type': 'start'}})
                assert 'ONLY SESSION TWO' in json.dumps(client.events[mark:])
                client.prompt(b)
        finally:
            client.close()
        # A new ACP connection resumes the persisted Rig conversation and replay.
        client = Client(binary, version, home)
        try:
            params = {'sessionId': b, 'cwd': str(two), 'mcpServers': []}
            if version == 2:
                params['replayFrom'] = {'type': 'start'}
            client.call('session/resume' if version == 2 else 'session/load', params)
            assert 'ONLY SESSION TWO' in json.dumps(client.events)
            client.prompt(b)
        finally:
            client.close()
        print(f'PASS ACP v{version}: isolation, replay, denial, cancellation, HTTP MCP headers')


def worker_pids(client):
    rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,command='], text=True)
    return [int(parts[0]) for row in rows.splitlines()
            if len(parts := row.split(None, 2)) == 3
            and int(parts[1]) == client.process.pid and 'acp-worker' in parts[2]]


def assert_reaped(pids):
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise AssertionError(f'worker {pid} survived supervisor shutdown')


def lifecycle(binary, version):
    with tempfile.TemporaryDirectory(prefix='rig-acp-lifecycle-') as directory:
        home = Path(directory)
        client = Client(binary, version, home)
        try:
            session = client.call('session/new', {'cwd': directory, 'mcpServers': []})['sessionId']
            client.choose(session, 'test/tool')
            request = client.send('session/prompt', {'sessionId': session, 'prompt': [{'type': 'text', 'text': 'wait'}]})
            while client.receive().get('method') != 'session/request_permission':
                pass
            pids = worker_pids(client)
            assert len(pids) == 1, pids
            os.kill(pids[0], signal.SIGKILL)
            while True:
                value = client.receive()
                if version == 1 and value.get('id') == request:
                    assert 'error' in value, value
                    break
                update = value.get('params', {}).get('update', {})
                if version == 2 and update.get('stopReason') == 'error':
                    break
            assert_reaped(pids)
            if version == 2:
                client.call('session/close', {'sessionId': session})
            client.call('session/resume' if version == 2 else 'session/load',
                        {'sessionId': session, 'cwd': directory, 'mcpServers': []})
            # Restoration may continue an unfinished turn; cancellation must still work.
            client.write({'method': 'session/cancel', 'params': {'sessionId': session}})
        finally:
            pids = worker_pids(client)
            client.close()
            assert_reaped(pids)
        client = Client(binary, version, home)
        session = client.call('session/new', {'cwd': directory, 'mcpServers': []})['sessionId']
        pids = worker_pids(client)
        assert len(pids) == 1, pids
        client.process.send_signal(signal.SIGTERM)
        client.process.wait(timeout=8)
        assert client.process.returncode == 130, client.process.returncode
        assert_reaped(pids)
        client.process.stdin.close()
        client.stderr.close()
        print(f'PASS ACP v{version}: worker crash/reap/resume, EOF and first SIGTERM with open stdin')


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 0), Mcp)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for version in (1, 2):
            run(str(Path(sys.argv[1]).resolve()), version,
                f'http://127.0.0.1:{server.server_port}/mcp')
        assert Mcp.calls.count('tools/call') == 2, Mcp.calls
        for version in (1, 2):
            lifecycle(str(Path(sys.argv[1]).resolve()), version)
    finally:
        server.shutdown()
        server.server_close()

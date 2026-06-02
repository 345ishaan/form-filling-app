import { useState, useCallback, useEffect } from 'react';
import { HTTP_BASE } from './lib/apiBase.js';
import { useCmsWS } from './hooks/useCmsWS.js';

function formatTool(name, input) {
  const args = input ? JSON.stringify(input).slice(0, 120) : '';
  return `${name}${args ? ` ${args}` : ''}`;
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [render, setRender] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [caseLoaded, setCaseLoaded] = useState(false);
  const [formsCount, setFormsCount] = useState(0);
  const [sessionGen, setSessionGen] = useState(0);
  const [input, setInput] = useState('');
  const [uploading, setUploading] = useState(false);

  const { connect, sendMessage, interrupt, connected, ready } = useCmsWS(`cms:${sessionGen}`);

  const addMessage = useCallback((type, content) => {
    setMessages((m) => [...m, { type, content, id: Date.now() + Math.random() }]);
  }, []);

  useEffect(() => {
    connect({
      onInit: (data) => {
        if (data.render) setRender(data.render);
        if (data.session_id) setSessionId(data.session_id);
        addMessage('system', `Session ${data.session_id}${data.resumed ? ' (resumed)' : ''}`);
      },
      onStateUpdate: (data) => { if (data.render) setRender(data.render); },
      onText: (data) => addMessage('agent', data.text),
      onThinking: (data) => addMessage('thinking', data.thinking?.slice(0, 500)),
      onToolUse: (data) => addMessage('system', `🔧 ${formatTool(data.name, data.input)}`),
      onComplete: (data) => {
        addMessage('system', 'Agent turn complete.');
        if (data.render) setRender(data.render);
      },
      onError: (data) => addMessage('system', `Error: ${data.error}`),
      onStatus: (data) => addMessage('system', data.message || 'Starting…'),
      onConnected: () => addMessage('system', 'Connected.'),
      onDisconnected: () => addMessage('system', 'Disconnected.'),
    });
    setMessages([]);
    setRender('');
    setSessionId(null);
    setCaseLoaded(false);
    setFormsCount(0);
  }, [sessionGen, connect, addMessage]);

  const uploadFile = async (endpoint, file) => {
    if (!sessionId || !file) return;
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${HTTP_BASE.replace(/\/$/, '')}/ws/sessions/${sessionId}/${endpoint}`, {
        method: 'POST',
        body: fd,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload failed');
      return data;
    } finally {
      setUploading(false);
    }
  };

  const handleCaseUpload = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    try {
      const data = await uploadFile('case', file);
      setCaseLoaded(true);
      if (data.render) setRender(data.render);
      addMessage('system', `Case uploaded (${data.file_count} files).`);
    } catch (err) {
      addMessage('system', `Case upload failed: ${err.message}`);
    }
  };

  const handleFormUpload = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    try {
      const data = await uploadFile('form', file);
      setFormsCount((data.forms || []).length);
      addMessage('system', `Form saved. Ask the agent to parse and fill it.`);
    } catch (err) {
      addMessage('system', `Form upload failed: ${err.message}`);
    }
  };

  const downloadFilled = async () => {
    if (!sessionId) return;
    try {
      const res = await fetch(`${HTTP_BASE.replace(/\/$/, '')}/ws/sessions/${sessionId}/filled`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Export failed');
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `filled-${sessionId}.json`;
      a.click();
      URL.revokeObjectURL(url);
      addMessage('system', 'Downloaded filled JSON.');
    } catch (err) {
      addMessage('system', `Export failed: ${err.message}`);
    }
  };

  const handleSend = (e) => {
    e.preventDefault();
    if (!input.trim() || !ready) return;
    addMessage('user', input.trim());
    sendMessage(input.trim());
    setInput('');
  };

  const statusClass = ready ? 'status-ready' : connected ? 'status-connecting' : 'status-off';
  const statusLabel = ready ? 'ready' : connected ? 'connecting…' : 'offline';

  return (
    <div className="app">
      <header>
        <h1>Form Filling App</h1>
        <p>Upload case documents and a blank form, then chat with the agent to fill fields.</p>
      </header>

      <div className="toolbar">
        <button type="button" className="btn" disabled={!ready} onClick={() => setSessionGen((g) => g + 1)}>
          New session
        </button>
        <label className="btn file-label">
          Upload case .zip
          <input type="file" accept=".zip" disabled={!ready || uploading} onChange={handleCaseUpload} />
        </label>
        <label className="btn file-label">
          Upload form .pdf
          <input type="file" accept=".pdf" disabled={!ready || uploading} onChange={handleFormUpload} />
        </label>
        <button type="button" className="btn" disabled={!sessionId || formsCount === 0} onClick={downloadFilled}>
          ↓ Filled JSON
        </button>
        <span className={`status-dot ${statusClass}`}>● {statusLabel}</span>
      </div>

      <div className="layout">
        <div className="panel">
          <div className="panel-header">Environment</div>
          <div className="panel-body">
            <pre className="render-pre">
              {render || (caseLoaded
                ? (formsCount > 0 ? 'Case + form loaded — ask the agent to fill the form.' : 'Case loaded — upload a form PDF.')
                : 'Upload a case .zip and form .pdf to begin.')}
            </pre>
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">Agent chat</div>
          <div className="chat-messages">
            {messages.map((msg) => (
              <div key={msg.id} className={`chat-msg ${msg.type}`}>{msg.content}</div>
            ))}
          </div>
          <form className="chat-input-row" onSubmit={handleSend}>
            <input
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="e.g. Parse the form and fill it from the case documents"
              disabled={!ready}
            />
            <button type="submit" className="btn btn-primary" disabled={!ready}>Send</button>
            <button type="button" className="btn btn-danger" disabled={!ready} onClick={interrupt}>Stop</button>
          </form>
        </div>
      </div>
    </div>
  );
}

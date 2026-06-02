import { useRef, useCallback, useState, useEffect } from 'react';
import { cmsWebSocketUrl } from '../lib/apiBase.js';

export function useCmsWS(sessionKey = 'default') {
  const wsRef = useRef(null);
  const handlersRef = useRef({});
  const pingRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [ready, setReady] = useState(false);
  const [sessionId, setSessionId] = useState(null);

  const routeMessage = useCallback((data) => {
    const h = handlersRef.current;
    switch (data.type) {
      case 'init':
        setSessionId(data.session_id);
        setReady(true);
        h.onInit?.(data);
        break;
      case 'status':
        h.onStatus?.(data);
        break;
      case 'state_update':
        h.onStateUpdate?.(data);
        break;
      case 'text':
        if (typeof data.text === 'string' && data.text.startsWith('👤 You:')) break;
        h.onText?.(data);
        break;
      case 'thinking':
        h.onThinking?.(data);
        break;
      case 'tool_use':
        h.onToolUse?.(data);
        break;
      case 'tool_result':
        h.onToolResult?.(data);
        break;
      case 'complete':
        h.onComplete?.(data);
        break;
      case 'error':
        h.onError?.(data);
        break;
      default:
        break;
    }
  }, []);

  const disconnect = useCallback(() => {
    if (pingRef.current) { clearInterval(pingRef.current); pingRef.current = null; }
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
    setConnected(false);
    setReady(false);
  }, []);

  const connect = useCallback((handlers) => {
    disconnect();
    handlersRef.current = handlers;
    const ws = new WebSocket(cmsWebSocketUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      handlers.onConnected?.();
      pingRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'ping' }));
      }, 20000);
    };

    ws.onmessage = (ev) => {
      try { routeMessage(JSON.parse(ev.data)); } catch { /* ignore */ }
    };

    ws.onclose = () => {
      setConnected(false);
      setReady(false);
      handlers.onDisconnected?.();
    };

    ws.onerror = () => handlers.onError?.({ error: 'WebSocket error' });
  }, [disconnect, routeMessage]);

  useEffect(() => () => disconnect(), [disconnect, sessionKey]);

  const sendMessage = useCallback((content) => {
    wsRef.current?.send(JSON.stringify({ type: 'message', content }));
  }, []);

  const interrupt = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: 'interrupt' }));
  }, []);

  return { connect, sendMessage, interrupt, connected, ready, sessionId, disconnect };
}

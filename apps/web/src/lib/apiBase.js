export const HTTP_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export function apiWebSocketOrigin(baseUrl = HTTP_BASE) {
  const url = new URL(baseUrl);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.origin;
}

export function cmsWebSocketUrl(baseUrl = HTTP_BASE) {
  return `${apiWebSocketOrigin(baseUrl)}/ws/cms`;
}

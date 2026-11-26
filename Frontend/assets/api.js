const configuredApiBase = import.meta.env.VITE_API_BASE_URL?.trim();
if (!configuredApiBase) {
  throw new Error("VITE_API_BASE_URL must be set in the frontend environment.");
}
const API_BASE = configuredApiBase.replace(/\/+$/, "");

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

export function redirectToLogin() {
  window.location.href = apiUrl("/auth/login");
}

export function redirectToLogout() {
  window.location.href = apiUrl("/auth/logout");
}

export async function checkAuth() {
  try {
    const response = await fetch(apiUrl("/auth/me"), {
      mode: "cors",
      credentials: "include",
    });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

async function ensureAuthenticated(response) {
  if (response.status !== 401) return;
  redirectToLogin();
  throw new Error("Session expired — redirecting to login.");
}

export async function sendChat(input, sessionId) {
  const response = await fetch(apiUrl("/chat"), {
    method: "POST",
    mode: "cors",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ inp: input, thread_id: sessionId }),
  });
  await ensureAuthenticated(response);
  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || `Request failed (${response.status})`);
  }
  return response.json();
}

export async function ingestFile(file, sessionId) {
  const query = new URLSearchParams({ thread_id: sessionId });
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(apiUrl(`/ingest?${query}`), {
    method: "POST",
    mode: "cors",
    credentials: "include",
    body: formData,
  });
  await ensureAuthenticated(response);

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Ingest failed (${response.status})`);
  }
  return data;
}

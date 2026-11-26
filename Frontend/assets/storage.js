const sessionStorageKey = (email) => `chatbot_sessions_${email}`;
const messageStorageKey = (email, sessionId) =>
  `chatbot_messages_${email}_${sessionId}`;

export function loadSessions(email) {
  const raw = localStorage.getItem(sessionStorageKey(email));
  return new Set(raw ? JSON.parse(raw) : []);
}

export function saveSessions(email, sessions) {
  localStorage.setItem(sessionStorageKey(email), JSON.stringify([...sessions]));
}

export function loadMessages(email, sessionId) {
  const raw = localStorage.getItem(messageStorageKey(email, sessionId));
  return raw ? JSON.parse(raw) : [];
}

export function saveMessages(email, sessionId, messages) {
  localStorage.setItem(
    messageStorageKey(email, sessionId),
    JSON.stringify(messages),
  );
}

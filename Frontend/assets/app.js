import {
  checkAuth,
  ingestFile,
  redirectToLogin,
  redirectToLogout,
  sendChat,
} from "./api.js";
import {
  loadMessages,
  loadSessions,
  saveMessages,
  saveSessions,
} from "./storage.js";

let currentUser = null;
let sessionId = null;
let sessions = new Set();
let messageHistory = [];

const elements = {
  app: document.getElementById("app"),
  chatInput: document.getElementById("chat-input"),
  chatWindow: document.getElementById("chat-window"),
  emptyState: document.getElementById("empty-state"),
  fileInput: document.getElementById("file-input"),
  fileName: document.getElementById("file-name"),
  ingestStatus: document.getElementById("ingest-status"),
  loginChecking: document.getElementById("login-checking"),
  loginScreen: document.getElementById("login-screen"),
  sendButton: document.getElementById("send-btn"),
  sessionList: document.getElementById("session-list"),
  threadDisplay: document.getElementById("thread-display"),
  userEmail: document.getElementById("user-email-display"),
};

function createSessionId() {
  return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, (character) =>
    (
      character ^
      (crypto.getRandomValues(new Uint8Array(1))[0] &
        (15 >> (character / 4)))
    ).toString(16),
  );
}

function persistSessions() {
  saveSessions(currentUser.email, sessions);
}

function persistMessages() {
  saveMessages(currentUser.email, sessionId, messageHistory);
}

function updateThreadDisplay() {
  elements.threadDisplay.textContent = `thread: ${sessionId}`;
}

function renderSessionList() {
  elements.sessionList.innerHTML = "";
  sessions.forEach((threadId) => {
    const button = document.createElement("button");
    button.className =
      "session-btn" + (threadId === sessionId ? " active" : "");
    button.textContent = threadId;
    button.title = threadId;
    button.addEventListener("click", () => switchToSession(threadId));
    elements.sessionList.appendChild(button);
  });
}

function appendBubble(avatar, text, shouldScroll = true) {
  elements.emptyState?.remove();

  const row = document.createElement("div");
  row.className = `msg-row ${avatar}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  row.appendChild(bubble);
  elements.chatWindow.appendChild(row);

  if (shouldScroll) {
    elements.chatWindow.scrollTop = elements.chatWindow.scrollHeight;
  }
}

function renderChat() {
  elements.chatWindow.innerHTML = "";
  if (messageHistory.length === 0) {
    const emptyState = document.createElement("div");
    emptyState.className = "empty-state";
    emptyState.id = "empty-state";
    emptyState.textContent =
      "Start a conversation, or upload a document to chat with it.";
    elements.chatWindow.appendChild(emptyState);
    elements.emptyState = emptyState;
    return;
  }

  messageHistory.forEach(({ avatar, message }) =>
    appendBubble(avatar, message, false),
  );
  elements.chatWindow.scrollTop = elements.chatWindow.scrollHeight;
}

function createNewSession() {
  sessionId = createSessionId();
  messageHistory = [];
  sessions.add(sessionId);
  persistSessions();
  renderSessionList();
  renderChat();
  updateThreadDisplay();
}

function switchToSession(threadId) {
  sessionId = threadId;
  messageHistory = loadMessages(currentUser.email, threadId);
  renderChat();
  renderSessionList();
  updateThreadDisplay();
}

function enterApp() {
  elements.loginScreen.style.display = "none";
  elements.app.classList.add("visible");
  elements.userEmail.textContent = currentUser.email;
  sessions = loadSessions(currentUser.email);

  if (sessions.size === 0) {
    createNewSession();
  } else {
    switchToSession(sessions.values().next().value);
  }
}

async function handleSend() {
  const text = elements.chatInput.value.trim();
  if (!text) return;

  elements.chatInput.value = "";
  elements.chatInput.style.height = "auto";
  elements.sendButton.disabled = true;
  messageHistory.push({ avatar: "user", message: text });
  appendBubble("user", text);
  persistMessages();

  const typing = document.createElement("div");
  typing.className = "typing";
  typing.textContent = "Assistant is typing…";
  elements.chatWindow.appendChild(typing);
  elements.chatWindow.scrollTop = elements.chatWindow.scrollHeight;

  try {
    const data = await sendChat(text, sessionId);
    typing.remove();
    const reply =
      typeof data.response === "string"
        ? data.response
        : JSON.stringify(data.response);
    messageHistory.push({ avatar: "assistant", message: reply });
    appendBubble("assistant", reply);
    persistMessages();
    if (!sessions.has(sessionId)) {
      sessions.add(sessionId);
      persistSessions();
      renderSessionList();
    }
  } catch (error) {
    typing.remove();
    appendBubble("assistant", `⚠️ Error: ${error.message}`);
  } finally {
    elements.sendButton.disabled = false;
    elements.chatInput.focus();
  }
}

async function handleFileSelection() {
  const file = elements.fileInput.files[0];
  if (!file) return;

  elements.fileName.textContent = file.name;
  elements.ingestStatus.textContent = "Uploading & processing…";
  elements.ingestStatus.className = "";

  try {
    const result = await ingestFile(file, sessionId);
    elements.ingestStatus.textContent =
      `✓ Ingested (${result.chunks_ingested} chunks)`;
    elements.ingestStatus.className = "ok";
  } catch (error) {
    elements.ingestStatus.textContent = `✗ ${error.message}`;
    elements.ingestStatus.className = "err";
  } finally {
    elements.fileInput.value = "";
  }
}

function registerEventHandlers() {
  document
    .getElementById("google-login-btn")
    .addEventListener("click", redirectToLogin);
  document
    .getElementById("logout-btn")
    .addEventListener("click", redirectToLogout);
  document.getElementById("new-conv-btn").addEventListener("click", () => {
    if (messageHistory.length > 0) createNewSession();
  });

  elements.chatInput.addEventListener("input", () => {
    elements.chatInput.style.height = "auto";
    elements.chatInput.style.height =
      `${Math.min(elements.chatInput.scrollHeight, 140)}px`;
  });
  elements.chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  });
  elements.sendButton.addEventListener("click", handleSend);
  elements.fileInput.addEventListener("change", handleFileSelection);
}

async function initialize() {
  registerEventHandlers();
  elements.loginChecking.textContent = "Checking session…";
  const user = await checkAuth();
  elements.loginChecking.textContent = "";
  if (user) {
    currentUser = user;
    enterApp();
  }
}

initialize();

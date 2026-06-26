/* ── config ──────────────────────────────────────────────────────────────── */
const BASE_URL = "http://localhost:8080";

/* ── state ───────────────────────────────────────────────────────────────── */
let token = localStorage.getItem("token") || null;
let sessionId = localStorage.getItem("sessionId") || null;
let isThinking = false;

/* ── api helpers ─────────────────────────────────────────────────────────── */
function authHeaders() {
  return {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${token}`,
  };
}

async function apiFetch(path, options = {}) {
  const resp = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: { ...authHeaders(), ...(options.headers || {}) },
  });

  if (resp.status === 401) {
    handleLogout();
    throw new Error("session expired");
  }

  return resp;
}

/* ── routing ─────────────────────────────────────────────────────────────── */
function showView(view) {
  document.getElementById("view-login").classList.toggle("hidden", view !== "login");
  document.getElementById("view-app").classList.toggle("hidden", view === "login");

  if (view !== "login") {
    const isMemory = view === "memory";
    document.getElementById("view-chat").classList.toggle("hidden", isMemory);
    document.getElementById("view-memory").classList.toggle("hidden", !isMemory);
    document.getElementById("nav-chat").classList.toggle("active", !isMemory);
    document.getElementById("nav-memory").classList.toggle("active", isMemory);
  }
}

function switchView(view) {
  showView(view);
  if (view === "memory") loadMemory();
}

/* ── login tab switching ─────────────────────────────────────────────────── */
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    clearFormMessages();
  });
});

function clearFormMessages() {
  ["login-error", "reg-error", "reg-success"].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.classList.add("hidden"); el.textContent = ""; }
  });
}

/* ── auth ────────────────────────────────────────────────────────────────── */
async function handleLogin() {
  const email = document.getElementById("login-email").value.trim();
  const password = document.getElementById("login-password").value;
  const errEl = document.getElementById("login-error");
  const btn = document.getElementById("login-btn");

  errEl.classList.add("hidden");

  if (!email || !password) {
    errEl.textContent = "Both fields are required.";
    errEl.classList.remove("hidden");
    return;
  }

  btn.disabled = true;
  btn.textContent = "signing in…";

  try {
    const resp = await fetch(`${BASE_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    const data = await resp.json();

    if (resp.ok) {
      token = data.access_token;
      localStorage.setItem("token", token);
      await initApp();
    } else {
      errEl.textContent = data.detail || "Login failed.";
      errEl.classList.remove("hidden");
    }
  } catch (e) {
    errEl.textContent = "Could not reach the server.";
    errEl.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "sign in";
  }
}

async function handleRegister() {
  const email = document.getElementById("reg-email").value.trim();
  const password = document.getElementById("reg-password").value;
  const errEl = document.getElementById("reg-error");
  const sucEl = document.getElementById("reg-success");
  const btn = document.getElementById("reg-btn");

  errEl.classList.add("hidden");
  sucEl.classList.add("hidden");

  if (!email || !password) {
    errEl.textContent = "Both fields are required.";
    errEl.classList.remove("hidden");
    return;
  }

  btn.disabled = true;
  btn.textContent = "creating account…";

  try {
    const resp = await fetch(`${BASE_URL}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    const data = await resp.json();

    if (resp.ok) {
      sucEl.textContent = "Account created — sign in above.";
      sucEl.classList.remove("hidden");
      document.getElementById("reg-email").value = "";
      document.getElementById("reg-password").value = "";
    } else {
      errEl.textContent = data.detail || "Registration failed.";
      errEl.classList.remove("hidden");
    }
  } catch (e) {
    errEl.textContent = "Could not reach the server.";
    errEl.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "create account";
  }
}

function handleLogout() {
  token = null;
  sessionId = null;
  localStorage.removeItem("token");
  localStorage.removeItem("sessionId");

  // Clear forms
  document.getElementById("login-email").value = "";
  document.getElementById("login-password").value = "";
  const regEmail = document.getElementById("reg-email");
  if (regEmail) regEmail.value = "";
  const regPassword = document.getElementById("reg-password");
  if (regPassword) regPassword.value = "";

  document.getElementById("chat-messages").innerHTML = "";
  document.getElementById("session-list").innerHTML = "";
  clearFormMessages();
  showView("login");
}

/* ── init ────────────────────────────────────────────────────────────────── */
async function initApp() {
  showView("chat");
  await loadSessions();
  if (sessionId) {
    await activateSession(sessionId, true);
  }
}

/* ── sessions ────────────────────────────────────────────────────────────── */
async function loadSessions() {
  try {
    const resp = await apiFetch("/session");
    const sessions = await resp.json();
    renderSessionList(sessions);
  } catch (e) {
    console.error("Failed to load sessions:", e);
  }
}

function renderSessionList(data) {
  const list = document.getElementById("session-list");
  list.innerHTML = "";

  const sessions = data && data.sessions ? data.sessions : [];

  if (sessions.length === 0) {
    list.innerHTML = `<p style="font-size:12px;color:var(--text-muted);padding:8px 10px;">no sessions yet</p>`;
    return;
  }

  sessions.forEach(sid => {
    const item = document.createElement("div");
    item.className = "session-item";
    item.id = `session-item-${sid}`;

    const label = document.createElement("button");
    label.className = "session-label" + (sid === sessionId ? " active" : "");
    label.textContent = sid.slice(0, 8) + "…";
    label.title = sid;
    label.onclick = () => activateSession(sid);

    const del = document.createElement("button");
    del.className = "session-delete";
    del.title = "delete session";
    del.innerHTML = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none">
      <path d="M1 1l10 10M11 1L1 11" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
    </svg>`;
    del.onclick = (e) => { e.stopPropagation(); handleEndSession(sid); };

    item.appendChild(label);
    item.appendChild(del);
    list.appendChild(item);
  });
}

async function loadSessionHistory(sid) {
  const overlay = document.getElementById("chat-loading-overlay");
  if (overlay) {
    overlay.querySelector(".loading-overlay-text").textContent = "loading session history…";
    overlay.classList.remove("hidden");
  }
  try {
    const resp = await apiFetch(`/session/${sid}/context?k=100`);
    const turns = await resp.json();

    const messagesContainer = document.getElementById("chat-messages");
    messagesContainer.innerHTML = "";

    // Turns are in reverse chronological order (newest first).
    // Reverse it to display oldest first.
    const chronologicalTurns = [...turns].reverse();

    chronologicalTurns.forEach(turn => {
      appendMessage("user", turn.user);
      appendMessage("assistant", turn.assistant, turn.meta || null);
    });
  } catch (e) {
    console.error("Failed to load session history:", e);
  } finally {
    if (overlay) {
      overlay.classList.add("hidden");
      overlay.querySelector(".loading-overlay-text").textContent = "distilling conversation & saving to memory…";
    }
    // Update welcome state if history is empty
    updateEmptyState();
  }
}

async function activateSession(sid, loadHistory = true) {
  sessionId = sid;
  localStorage.setItem("sessionId", sid);

  // update active state in list
  document.querySelectorAll(".session-label").forEach(el => {
    el.classList.toggle("active", el.title === sid);
  });

  document.getElementById("chat-messages").innerHTML = "";

  // show/hide empty state
  updateEmptyState();

  // enable input
  document.getElementById("chat-input").disabled = false;
  document.getElementById("send-btn").disabled = false;
  document.getElementById("chat-input").focus();

  if (loadHistory) {
    await loadSessionHistory(sid);
  }
}

function updateEmptyState() {
  const hasSession = !!sessionId;
  document.getElementById("chat-empty").classList.toggle("hidden", hasSession);
  
  const messages = document.getElementById("chat-messages");
  messages.classList.toggle("hidden", !hasSession);

  // Show a welcome message if the session is empty
  const emptyWelcome = document.getElementById("chat-new-welcome");
  if (hasSession && messages.children.length === 0) {
    if (!emptyWelcome) {
      const welcome = document.createElement("div");
      welcome.id = "chat-new-welcome";
      welcome.className = "new-session-welcome";
      welcome.innerHTML = `
        <div class="welcome-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
        </div>
        <p class="welcome-title">New Session Started</p>
        <p class="welcome-sub">Ask a question to begin exploring your knowledge base or user memory.</p>
      `;
      messages.appendChild(welcome);
    }
  } else {
    if (emptyWelcome) {
      emptyWelcome.remove();
    }
  }

  const header = document.getElementById("chat-header");
  if (header) {
    header.classList.toggle("hidden", !hasSession);
    if (hasSession) {
      document.getElementById("chat-header-session-name").textContent = `Session: ${sessionId.slice(0, 8)}…`;
      document.getElementById("chat-header-session-name").title = sessionId;
    }
  }
}

async function createSession() {
  const overlay = document.getElementById("chat-loading-overlay");
  if (overlay) {
    overlay.querySelector(".loading-overlay-text").textContent = "creating new session…";
    overlay.classList.remove("hidden");
  }
  try {
    const resp = await apiFetch("/session", { method: "POST" });
    const data = await resp.json();
    await loadSessions();
    await activateSession(data.session_id);
  } catch (e) {
    console.error("Failed to create session:", e);
  } finally {
    if (overlay) {
      overlay.classList.add("hidden");
      overlay.querySelector(".loading-overlay-text").textContent = "distilling conversation & saving to memory…";
    }
  }
}

async function deleteSession(sid) {
  const overlay = document.getElementById("chat-loading-overlay");
  if (overlay) overlay.classList.remove("hidden");
  try {
    await apiFetch(`/session/${sid}/end`, { method: "DELETE" });
    if (sessionId === sid) {
      sessionId = null;
      localStorage.removeItem("sessionId");
      document.getElementById("chat-messages").innerHTML = "";
      document.getElementById("chat-input").disabled = true;
      document.getElementById("send-btn").disabled = true;
      updateEmptyState();
    }
    await loadSessions();
  } catch (e) {
    console.error("Failed to delete session:", e);
  } finally {
    if (overlay) overlay.classList.add("hidden");
  }
}

async function handleEndSession(sid) {
  if (!sid) return;
  if (confirm("Are you sure you want to end this session? This will summarize the conversation and save it to your Long-Term Memory (LTM).")) {
    await deleteSession(sid);
  }
}

async function handleEndCurrentSession() {
  await handleEndSession(sessionId);
}

/* ── chat ────────────────────────────────────────────────────────────────── */
function handleInputKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 160) + "px";
}

async function sendMessage() {
  const input = document.getElementById("chat-input");
  const question = input.value.trim();

  if (!question || isThinking || !sessionId) return;

  // render user bubble
  appendMessage("user", question);
  input.value = "";
  input.style.height = "auto";

  // show thinking indicator
  isThinking = true;
  const thinkingEl = appendThinking();

  try {
    const resp = await apiFetch("/agent/qa", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, question }),
    });

    const data = await resp.json();
    thinkingEl.remove();
    console.log(data.evidence);
    appendMessage("assistant", data.answer, {
      tools_used: data.tools_used || [],
      evidence: data.evidence || [],
    });
  } catch (e) {
    thinkingEl.remove();
    appendMessage("assistant", "Something went wrong. Please try again.", {});
  } finally {
    isThinking = false;
    scrollToBottom();
  }
}

function appendMessage(role, content, meta = null) {
  const messages = document.getElementById("chat-messages");

  // Remove empty welcome if present
  const welcome = document.getElementById("chat-new-welcome");
  if (welcome) {
    welcome.remove();
  }

  const msg = document.createElement("div");
  msg.className = `message ${role}`;

  const inner = document.createElement("div");
  inner.className = "message-inner";

  if (role === "assistant") {
    const roleLabel = document.createElement("div");
    roleLabel.className = "role-label";
    roleLabel.textContent = "assistant";
    inner.appendChild(roleLabel);
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") {
    if (typeof marked !== "undefined" && typeof marked.parse === "function") {
      bubble.innerHTML = marked.parse(content);
    } else if (typeof marked === "function") {
      bubble.innerHTML = marked(content);
    } else {
      bubble.textContent = content;
    }
  } else {
    bubble.textContent = content;
  }
  inner.appendChild(bubble);

  // pills + evidence for assistant
  if (role === "assistant" && meta) {
    const { tools_used, evidence } = meta;

    if (tools_used && tools_used.length > 0) {
      const pillsRow = document.createElement("div");
      pillsRow.className = "pills-row";

      tools_used.forEach(tool => {
        const pill = document.createElement("span");
        const cfg = TOOL_CONFIG[tool] || { cls: "pill-kb", icon: "⚙", label: tool };
        pill.className = `pill ${cfg.cls}`;
        pill.innerHTML = `${cfg.icon} ${cfg.label}`;
        pillsRow.appendChild(pill);
      });

      inner.appendChild(pillsRow);
    }

    if (evidence && evidence.length > 0) {
      const toggle = document.createElement("button");
      toggle.className = "evidence-toggle";
      toggle.innerHTML = `
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
          <path d="M2 2h8M2 5h8M2 8h5" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
        </svg>
        ${evidence.length} source${evidence.length > 1 ? "s" : ""} <span class="toggle-arrow">▾</span>`;

      const panel = document.createElement("div");
      panel.className = "evidence-panel";
      panel.style.display = "none";

      evidence.forEach(chunk => {
        const chunkEl = document.createElement("div");
        chunkEl.className = "evidence-chunk";

        const text = document.createElement("div");
        text.textContent = chunk.snippet || chunk.content || chunk.page_content || "";
        chunkEl.appendChild(text);

        const source = chunk.title || chunk.filename || chunk.source || chunk.metadata?.source || "";
        if (source) {
          const src = document.createElement("div");
          src.className = "evidence-source";
          src.textContent = source;
          chunkEl.appendChild(src);
        }

        panel.appendChild(chunkEl);
      });

      toggle.addEventListener("click", () => {
        const open = panel.style.display !== "none";
        panel.style.display = open ? "none" : "flex";
        toggle.querySelector(".toggle-arrow").textContent = open ? "▾" : "▴";
      });

      inner.appendChild(toggle);
      inner.appendChild(panel);
    }
  }

  msg.appendChild(inner);
  messages.appendChild(msg);
  scrollToBottom();
  return msg;
}

function appendThinking() {
  const messages = document.getElementById("chat-messages");

  const msg = document.createElement("div");
  msg.className = "message assistant";

  const inner = document.createElement("div");
  inner.className = "message-inner";

  const roleLabel = document.createElement("div");
  roleLabel.className = "role-label";
  roleLabel.textContent = "assistant";
  inner.appendChild(roleLabel);

  const thinking = document.createElement("div");
  thinking.className = "thinking";
  thinking.innerHTML = `
    <div class="dot-pulse">
      <span></span><span></span><span></span>
    </div>
    thinking…`;
  inner.appendChild(thinking);

  msg.appendChild(inner);
  messages.appendChild(msg);
  scrollToBottom();
  return msg;
}

function scrollToBottom() {
  const messages = document.getElementById("chat-messages");
  messages.scrollTop = messages.scrollHeight;
}

const TOOL_CONFIG = {
  search_knowledge_base: {
    cls: "pill-kb",
    icon: `<svg width="11" height="11" viewBox="0 0 11 11" fill="none">
             <circle cx="5" cy="5" r="3.5" stroke="currentColor" stroke-width="1.2"/>
             <path d="M7.5 7.5L10 10" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
           </svg>`,
    label: "knowledge base",
  },
  web_search: {
    cls: "pill-web",
    icon: `<svg width="11" height="11" viewBox="0 0 11 11" fill="none">
             <circle cx="5.5" cy="5.5" r="4" stroke="currentColor" stroke-width="1.2"/>
             <path d="M5.5 1.5c-1.2 1.2-2 2.4-2 4s.8 2.8 2 4M5.5 1.5c1.2 1.2 2 2.4 2 4s-.8 2.8-2 4M1.5 5.5h8" stroke="currentColor" stroke-width="1.2"/>
           </svg>`,
    label: "web search",
  },
  recall_user_memory: {
    cls: "pill-mem",
    icon: `<svg width="11" height="11" viewBox="0 0 11 11" fill="none">
             <circle cx="5.5" cy="5.5" r="4" stroke="currentColor" stroke-width="1.2"/>
             <circle cx="5.5" cy="5.5" r="1.5" stroke="currentColor" stroke-width="1.2"/>
           </svg>`,
    label: "memory used",
  },
};

/* ── memory ──────────────────────────────────────────────────────────────── */
async function loadMemory() {
  const list = document.getElementById("memory-list");
  list.innerHTML = `<div class="memory-loading">loading memories…</div>`;

  try {
    const resp = await apiFetch("/memory/ltm");
    const memories = await resp.json();

    if (!memories || memories.length === 0) {
      list.innerHTML = `<div class="memory-empty">
        no long-term memories yet.<br>
        they're created automatically when sessions end.
      </div>`;
      return;
    }

    list.innerHTML = `<p style="font-size:12px;color:var(--text-muted);margin-bottom:4px">${memories.length} memor${memories.length === 1 ? "y" : "ies"} stored</p>`;

    memories.forEach(mem => {
      const card = document.createElement("div");
      card.className = "memory-card";

      const text = document.createElement("div");
      text.className = "memory-text";
      if (typeof marked !== "undefined" && typeof marked.parse === "function") {
        text.innerHTML = marked.parse(mem.summary || "");
      } else if (typeof marked === "function") {
        text.innerHTML = marked(mem.summary || "");
      } else {
        text.textContent = mem.summary || "";
      }

      const footer = document.createElement("div");
      footer.className = "memory-footer";

      if (mem.created_at) {
        const date = document.createElement("span");
        date.className = "memory-date";
        date.textContent = formatDate(mem.created_at);
        footer.appendChild(date);
      }

      if (mem.session_id) {
        const sess = document.createElement("span");
        sess.className = "memory-session";
        sess.textContent = `session ${mem.session_id.slice(0, 8)}`;
        footer.appendChild(sess);
      }

      card.appendChild(text);
      card.appendChild(footer);
      list.appendChild(card);
    });

  } catch (e) {
    list.innerHTML = `<div class="memory-empty">failed to load memories.</div>`;
  }
}

function formatDate(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
      + " · " + d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  } catch {
    return iso;
  }
}

/* ── keyboard shortcut: Enter to submit login ────────────────────────────── */
document.addEventListener("keydown", e => {
  if (e.key === "Enter" && !document.getElementById("view-login").classList.contains("hidden")) {
    const activeTab = document.querySelector(".tab-btn.active")?.dataset?.tab;
    if (activeTab === "login") handleLogin();
    else if (activeTab === "register") handleRegister();
  }
});

/* ── boot ────────────────────────────────────────────────────────────────── */
(async () => {
  if (token) {
    // verify token is still valid by hitting a protected endpoint
    try {
      const resp = await fetch(`${BASE_URL}/session`, {
        headers: { "Authorization": `Bearer ${token}` },
      });
      if (resp.status === 401) throw new Error("expired");
      await initApp();
    } catch {
      handleLogout();
    }
  } else {
    showView("login");
  }
})();

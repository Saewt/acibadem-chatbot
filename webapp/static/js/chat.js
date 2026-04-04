(() => {
  const form = document.getElementById("chat-form");
  const stream = document.getElementById("message-stream");
  const questionInput = document.getElementById("question");
  const statusText = document.getElementById("status-text");
  const submitButton = form?.querySelector('button[type="submit"]');
  const newChatButton = document.getElementById("new-chat-btn");
  const conversationItems = Array.from(document.querySelectorAll(".conversation-item"));
  const WELCOME_MESSAGE =
    "Merhaba. Sorunu Türkçe yazabilirsin. Yanıtlar yalnızca indekslenmiş resmi kaynaklardan üretilecek.";

  if (!form || !stream || !questionInput || !statusText || !submitButton) {
    return;
  }

  let conversationId = null;

  function scrollToBottom() {
    stream.scrollTop = stream.scrollHeight;
  }

  function setPendingState(isPending) {
    questionInput.disabled = isPending;
    submitButton.disabled = isPending;
    if (newChatButton) {
      newChatButton.disabled = isPending;
    }
    conversationItems.forEach((item) => {
      item.disabled = isPending;
    });
  }

  function setActiveConversationItem(activeItem = null) {
    conversationItems.forEach((item) => {
      item.classList.toggle("active", item === activeItem);
    });
  }

  function appendSources(wrapper, sources) {
    if (!sources.length) {
      return;
    }

    const sourceList = document.createElement("ul");
    sourceList.className = "source-list";
    sources.forEach((source, index) => {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noreferrer noopener";
      link.textContent = `[${index + 1}] ${source.label || source.title}`;
      item.appendChild(link);
      sourceList.appendChild(item);
    });
    wrapper.appendChild(sourceList);
    scrollToBottom();
  }

  function renderAssistantMeta(meta, cached) {
    meta.textContent = "";

    const label = document.createElement("span");
    label.className = "message-label";
    label.textContent = "ACU ASISTAN";
    meta.appendChild(label);

    if (!cached) {
      return;
    }

    const badge = document.createElement("span");
    badge.className = "cache-badge";
    badge.textContent = "cache";
    meta.appendChild(badge);
  }

  function createMessage(role, { text = "", cached = false, streaming = false } = {}) {
    const wrapper = document.createElement("article");
    wrapper.className = `message ${role}`;
    if (streaming) {
      wrapper.classList.add("streaming");
    }

    const meta = document.createElement("div");
    meta.className = "message-meta";
    if (role === "user") {
      const label = document.createElement("span");
      label.className = "message-label";
      label.textContent = "SEN";
      meta.appendChild(label);
    } else {
      renderAssistantMeta(meta, cached);
    }
    wrapper.appendChild(meta);

    const body = document.createElement("p");
    body.className = "message-body";
    body.textContent = text;
    wrapper.appendChild(body);

    stream.appendChild(wrapper);
    scrollToBottom();
    return { wrapper, meta, body };
  }

  function addWelcomeMessage() {
    createMessage("assistant", { text: WELCOME_MESSAGE });
  }

  function resetChat() {
    conversationId = null;
    stream.textContent = "";
    addWelcomeMessage();
    setActiveConversationItem();
    questionInput.value = "";
    statusText.textContent = "Hazır.";
    questionInput.focus();
  }

  function processSSEEvent(block, onEvent) {
    let eventName = "message";
    const dataLines = [];

    block.split("\n").forEach((line) => {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    });

    if (dataLines.length) {
      onEvent(eventName, dataLines.join("\n"));
    }
  }

  function consumeSSEBuffer(buffer, onEvent) {
    const normalized = buffer.replace(/\r\n/g, "\n");
    const parts = normalized.split("\n\n");
    const remainder = parts.pop() || "";

    parts.forEach((block) => {
      if (block.trim()) {
        processSSEEvent(block, onEvent);
      }
    });

    return remainder;
  }

  async function submitQuestion(event) {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question) {
      return;
    }

    createMessage("user", { text: question });
    questionInput.value = "";
    statusText.textContent = "Yanıt akışı başlatılıyor...";
    setPendingState(true);

    const assistantMessage = createMessage("assistant", { streaming: true });
    let isCached = false;
    let streamFinished = false;

    try {
      const response = await fetch("/api/chat/stream/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          conversation_id: conversationId,
        }),
      });

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => ({}));
        throw new Error(errorPayload.error || "İstek başarısız oldu.");
      }

      if (!response.body) {
        throw new Error("Tarayıcı streaming yanıtını desteklemiyor.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        buffer = consumeSSEBuffer(buffer, (eventName, data) => {
          if (data === "[DONE]") {
            streamFinished = true;
            assistantMessage.wrapper.classList.remove("streaming");
            statusText.textContent = isCached ? "Yanıt cache üzerinden geldi." : "Yanıt hazır.";
            return;
          }

          const payload = JSON.parse(data);
          if (eventName === "meta") {
            if (Number.isInteger(payload.conversation_id)) {
              conversationId = payload.conversation_id;
            }
            isCached = Boolean(payload.cached);
            renderAssistantMeta(assistantMessage.meta, isCached);
            statusText.textContent = isCached ? "Cache yanıtı gönderiliyor..." : "Yanıt akıyor...";
            return;
          }

          if (eventName === "token") {
            assistantMessage.body.textContent += payload.text || "";
            scrollToBottom();
            return;
          }

          if (eventName === "sources") {
            appendSources(assistantMessage.wrapper, payload.sources || []);
          }
        });

        if (done) {
          break;
        }
      }

      if (!streamFinished) {
        throw new Error("Yanıt akışı beklenmeden sonlandı.");
      }
    } catch (error) {
      assistantMessage.wrapper.classList.remove("streaming");
      assistantMessage.body.textContent =
        error.message || "Beklenmeyen bir hata oluştu.";
      statusText.textContent = "Hata oluştu.";
    } finally {
      setPendingState(false);
      questionInput.focus();
      scrollToBottom();
    }
  }

  if (newChatButton) {
    newChatButton.addEventListener("click", resetChat);
  }

  conversationItems.forEach((item) => {
    item.addEventListener("click", () => {
      const nextConversationId = Number.parseInt(item.dataset.conversationId || "", 10);
      conversationId = Number.isInteger(nextConversationId) ? nextConversationId : null;
      setActiveConversationItem(item);
      statusText.textContent = "Seçili konuşmada devam ediliyor.";
      questionInput.focus();
    });
  });

  form.addEventListener("submit", submitQuestion);
})();

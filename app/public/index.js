/* app/public/index.js */

(function () {
    console.log("Vexilon Forensic UI Initialized");

    /**
     * Interaction Logic: Enter-to-Submit
     * (Mandated by UI Standards Section 2.3)
     */
    function setupEnterToSubmit() {
        const chatInput = document.querySelector("textarea");
        if (!chatInput || chatInput.dataset.listenerAttached) return;
        chatInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                // #chat-submit is Chainlit's current submit button id
                const sendBtn = document.querySelector("#chat-submit");
                
                if (sendBtn) {
                    // Button exists: preventDefault to block Chainlit's handler
                    // (Chainlit doesn't check disabled state, would submit anyway)
                    e.preventDefault();
                    
                    // Only click if enabled
                    if (!sendBtn.disabled) {
                        sendBtn.click();
                    }
                }
                // If no button found: don't preventDefault, allow default behavior
                // This prevents silent keystroke swallowing if selector breaks
            }
        });
        chatInput.dataset.listenerAttached = "true";
    }

    // ── Knowledge Base button a11y (WCAG 2.5.3 Label in Name) ────────────────
    // CSS in style.css relabels the visible text; aria-label keeps the
    // accessible name in sync for screen readers and voice control.

    function labelKnowledgeBaseButton() {
        const btn = document.querySelector("#readme-button");
        if (!btn || btn.dataset.knowledgeBaseLabeled) return;
        btn.setAttribute("aria-label", "Knowledge Base");
        btn.dataset.knowledgeBaseLabeled = "true";
    }

    // ── Core chat control a11y (WCAG 4.1.2 Name, Role, Value) ───────────────
    // Chainlit renders icon-only controls with no accessible name; set
    // aria-label directly so axe button-name passes and screen readers announce
    // purpose. Re-applied via setInterval when React re-renders the composer.

    const CHAT_CONTROL_LABELS = {
        "chat-profiles": "Choose persona",
        "upload-button": "Attach file",
        "chat-settings-open-modal": "Open chat settings",
        "chat-submit": "Send message",
        "stop-button": "Stop generation",
    };

    function labelChatControls() {
        for (const [id, label] of Object.entries(CHAT_CONTROL_LABELS)) {
            const el = document.getElementById(id);
            if (!el || el.dataset.chatControlLabeled) continue;
            el.setAttribute("aria-label", label);
            el.dataset.chatControlLabeled = "true";
        }
    }

    // ── Hide Readme drawer title ──────────────────────────────────────────────

    function hideReadmeDrawerTitle() {
        document.querySelectorAll("h2").forEach((el) => {
            if (el.textContent.trim() === "Readme") el.style.display = "none";
        });
    }

    // ── Build SHA ─────────────────────────────────────────────────────────────

    let buildSha = "dev";

    fetch("/api/version")
        .then((res) => res.json())
        .then((data) => {
            if (data.version) buildSha = data.version;
            replaceBuildSha();
        })
        .catch((err) => console.error("Error fetching version:", err));

    function replaceBuildSha() {
        document.querySelectorAll("code, span, p, li, a").forEach((el) => {
            if (el.textContent.includes("{{BUILD_SHA}}")) {
                el.innerHTML = el.innerHTML.replace("{{BUILD_SHA}}", buildSha);
            }
        });
    }

    // ── Brand Configuration ────────────────────────────────────────────────────

    let welcomeTitle = "BCGEU Navigator";

    fetch("/api/brand")
        .then((res) => res.json())
        .then((data) => {
            if (data.welcome_title) welcomeTitle = data.welcome_title;
        })
        .catch((err) => console.error("Error fetching brand:", err));

    function manageWelcomeTitle() {
        const chatArea = document.querySelector(".flex-grow.overflow-y-auto");
        if (!chatArea) return;

        const messages = document.querySelectorAll(".message");
        const existingTitle = document.getElementById("custom-welcome-title");

        if (messages.length === 0) {
            if (!existingTitle) {
                const titleEl = document.createElement("h1");
                titleEl.id = "custom-welcome-title";
                titleEl.style.textAlign = "center";
                titleEl.style.fontSize = "2.25rem";
                titleEl.style.fontWeight = "700";
                titleEl.style.marginTop = "1rem"; // Default is 4rem, reduced for aesthetic reasons
                titleEl.style.marginBottom = "1rem"; // Default is 2rem, reduced for aesthetic reasons
                titleEl.style.color = "inherit";
                titleEl.style.opacity = "0.9";
                titleEl.textContent = welcomeTitle;
                chatArea.prepend(titleEl);
            } else if (existingTitle.textContent.trim() !== welcomeTitle.trim()) {
                existingTitle.textContent = welcomeTitle;
            }
        } else {
            if (existingTitle) {
                existingTitle.remove();
            }
        }
    }

    // Poll for React-rendered elements that appear/disappear on navigation.
    setInterval(() => {
        setupEnterToSubmit();
        labelKnowledgeBaseButton();
        labelChatControls();
        hideReadmeDrawerTitle();
        replaceBuildSha();
        manageWelcomeTitle();
    }, 500);

    setupEnterToSubmit();
    labelKnowledgeBaseButton();
    labelChatControls();
    hideReadmeDrawerTitle();
    replaceBuildSha();
    manageWelcomeTitle();

    // ── Pseudonymous client ID ──────────────────────────────────────────────
    // Random, client-generated UUID persisted in localStorage. Not derived
    // from IP/device/any real identifying data — see PRIVACY.md. Used only
    // to distinguish discrete sessions for rate-limiting and log correlation.
    function getOrCreateClientId() {
        let id = localStorage.getItem("vexilon_client_id");
        if (!id) {
            id = crypto.randomUUID();
            localStorage.setItem("vexilon_client_id", id);
        }
        return id;
    }

    const clientId = getOrCreateClientId();

    function postClientId() {
        window.postMessage({ type: "vexilon_client_id", clientId }, window.location.origin);
    }

    // Retry briefly in case Chainlit's window-message listener hasn't
    // mounted yet on first paint.
    postClientId();
    setTimeout(postClientId, 300);
    setTimeout(postClientId, 1000);
})();

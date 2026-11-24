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
            if (!el || el.getAttribute("aria-label") === label) continue;
            el.setAttribute("aria-label", label);
        }
    }

    // ── Hide Readme drawer title ──────────────────────────────────────────────

    function hideReadmeDrawerTitle() {
        document.querySelectorAll("h2").forEach((el) => {
            if (el.textContent.trim() === "Readme") el.style.display = "none";
        });
    }

    // ── Knowledge Base drawer a11y (WCAG 1.4.3 color-contrast) ───────────────
    // Tag the Readme dialog by content so custom CSS can scope link-color overrides (#642).

    function isKnowledgeBaseDrawer(dialog) {
        const heading = dialog.querySelector("h2, h3");
        if (!heading) return false;
        const title = heading.textContent.trim();
        return title === "Knowledge Base" || title === "Readme";
    }

    function labelKnowledgeBaseDrawer() {
        document.querySelectorAll('[role="dialog"]').forEach((dialog) => {
            if (!dialog.hasAttribute("data-knowledge-base-drawer") && isKnowledgeBaseDrawer(dialog)) {
                dialog.setAttribute("data-knowledge-base-drawer", "true");
            }
        });
    }

    function setupKnowledgeBaseDrawerLabel() {
        const btn = document.querySelector("#readme-button");
        if (!btn || btn.dataset.drawerLabelAttached) return;
        btn.addEventListener("click", () => {
            setTimeout(labelKnowledgeBaseDrawer, 0);
        });
        btn.dataset.drawerLabelAttached = "true";
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


    // ── GoatCounter analytics ───────────────────────────────────────────────
    // Cookieless pageview and feature-usage counting. The script loads only
    // when a valid GOATCOUNTER_CODE is configured; otherwise nothing is
    // requested. Event payloads use a fixed allowlist so query text, filenames,
    // document titles, and error strings are never sent. See PRIVACY.md.

    const GOATCOUNTER_SCRIPT_SRC = "https://gc.zgo.at/count.js";
    const GOATCOUNTER_CODE_RE = /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/;
    const GOATCOUNTER_EVENTS = {
        "event/chat-submit": "Chat submit",
        "event/persona-lookup": "Persona: Lookup",
        "event/persona-grieve": "Persona: Grieve",
        "event/persona-manage": "Persona: Manage",
        "event/knowledge-base-open": "Knowledge Base open",
        "event/session-save": "Session save",
        "event/error-ui": "UI error",
    };
    const PERSONA_EVENTS = {
        lookup: "event/persona-lookup",
        grieve: "event/persona-grieve",
        manage: "event/persona-manage",
    };

    let lastChatProfile = null;
    let currentPersona = null;

    function setGoatCounterState(state) {
        document.documentElement.dataset.goatcounter = state;
    }

    function loadGoatCounter(code) {
        if (!code || !GOATCOUNTER_CODE_RE.test(code)) {
            setGoatCounterState("off");
            return;
        }
        if (!document.querySelector(`script[src="${GOATCOUNTER_SCRIPT_SRC}"]`)) {
            const script = document.createElement("script");
            script.async = true;
            script.src = GOATCOUNTER_SCRIPT_SRC;
            script.setAttribute(
                "data-goatcounter",
                `https://${code}.goatcounter.com/count`,
            );
            script.setAttribute(
                "data-goatcounter-settings",
                JSON.stringify({ allow_frame: true }),
            );
            document.head.appendChild(script);
        }
        setGoatCounterState("on");
    }

    function trackGoatCounterEvent(eventPath) {
        try {
            if (document.documentElement.dataset.goatcounter !== "on") return;
            const title = GOATCOUNTER_EVENTS[eventPath];
            if (!title) return;
            const gc = window.goatcounter;
            if (!gc || typeof gc.count !== "function") return;
            gc.count({
                path: eventPath,
                title: title,
                event: true,
            });
        } catch (_err) {
            // Missing, blocked, or failing analytics must never affect the UI.
        }
    }

    function normalizePersonaKey(text) {
        const key = (text || "").trim().toLowerCase();
        return PERSONA_EVENTS[key] ? key : null;
    }

    function syncChatProfilePersona() {
        const profileBtn = document.getElementById("chat-profiles");
        if (!profileBtn) return;

        if (!profileBtn.dataset.gcProfileObserved && typeof MutationObserver === "function") {
            const observer = new MutationObserver(() => syncChatProfilePersona());
            observer.observe(profileBtn, {
                childList: true,
                subtree: true,
                characterData: true,
            });
            profileBtn.dataset.gcProfileObserved = "true";
        }

        const key = normalizePersonaKey(profileBtn.textContent);
        if (!key) return;

        if (lastChatProfile === null) {
            lastChatProfile = key;
            if (currentPersona === null) {
                currentPersona = key;
            }
            return;
        }

        if (key !== lastChatProfile) {
            lastChatProfile = key;
            currentPersona = key;
            trackGoatCounterEvent(PERSONA_EVENTS[key]);
        }
    }

    function trackErrorToasts() {
        document.querySelectorAll('[data-sonner-toast][data-type="error"]').forEach((toast) => {
            if (toast.dataset.gcTracked) return;
            toast.dataset.gcTracked = "true";
            trackGoatCounterEvent("event/error-ui");
        });
    }

    document.addEventListener(
        "click",
        (event) => {
            const target = event.target instanceof Element ? event.target : null;
            if (!target) return;

            const submitBtn = target.closest("#chat-submit");
            if (submitBtn && !submitBtn.disabled) {
                trackGoatCounterEvent("event/chat-submit");
                if (currentPersona && PERSONA_EVENTS[currentPersona]) {
                    trackGoatCounterEvent(PERSONA_EVENTS[currentPersona]);
                }
                return;
            }

            const starterBtn = target.closest('button[id^="starter-"]');
            if (starterBtn && !starterBtn.disabled) {
                trackGoatCounterEvent("event/chat-submit");
                return;
            }

            const readmeBtn = target.closest("#readme-button");
            if (readmeBtn && !readmeBtn.disabled) {
                trackGoatCounterEvent("event/knowledge-base-open");
                return;
            }

            const confirmBtn = target.closest("#chat-settings #confirm");
            if (confirmBtn && !confirmBtn.disabled) {
                const personaSelect = document.querySelector("#chat-settings #Persona");
                const key = personaSelect ? normalizePersonaKey(personaSelect.textContent) : null;
                if (key) {
                    currentPersona = key;
                    trackGoatCounterEvent(PERSONA_EVENTS[key]);
                }
                return;
            }

            const actionBtn = target.closest("button");
            if (actionBtn && !actionBtn.disabled) {
                const label = (actionBtn.textContent || "").replace(/\s+/g, " ").trim();
                if (label === "💾 Save Session" || label === "Save Session") {
                    trackGoatCounterEvent("event/session-save");
                }
            }
        },
        true,
    );

    fetch("/api/goatcounter")
        .then((res) => (res.ok ? res.json() : {}))
        .then((data) => loadGoatCounter(data && data.goatcounter_code))
        .catch(() => setGoatCounterState("off"));

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
        labelKnowledgeBaseDrawer();
        setupKnowledgeBaseDrawerLabel();
        syncChatProfilePersona();
        trackErrorToasts();
        replaceBuildSha();
        manageWelcomeTitle();
    }, 500);

    setupEnterToSubmit();
    labelKnowledgeBaseButton();
    labelChatControls();
    hideReadmeDrawerTitle();
    setupKnowledgeBaseDrawerLabel();
    labelKnowledgeBaseDrawer();
    syncChatProfilePersona();
    trackErrorToasts();
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

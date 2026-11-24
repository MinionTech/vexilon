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
    // Keep the visible label, aria-label, and inner text in sync as
    // "Knowledge Base" (Chainlit's default visible text is "Readme").

    function labelKnowledgeBaseButton() {
        const btn = document.querySelector("#readme-button");
        if (!btn) return;
        if (btn.getAttribute("aria-label") !== "Knowledge Base") {
            btn.setAttribute("aria-label", "Knowledge Base");
        }
        const label = btn.querySelector("span");
        if (label && label.textContent.trim() !== "Knowledge Base") {
            label.textContent = "Knowledge Base";
        }
        if (btn.dataset.knowledgeBaseLabeled !== "true") {
            btn.dataset.knowledgeBaseLabeled = "true";
        }
    }

    // ── Core chat control a11y (WCAG 4.1.2 Name, Role, Value) ───────────────
    // Chainlit renders icon-only controls with no accessible name; set
    // aria-label directly so axe button-name passes and screen readers announce
    // purpose. Re-applied via MutationObserver and setInterval when React
    // mounts or re-renders the composer.

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
            if (!el) continue;
            // Chainlit marks the upload control as presentational, which
            // hides it from assistive tech and makes aria-label invalid.
            if (el.getAttribute("role") === "presentation") {
                el.removeAttribute("role");
            }
            // The upload node can briefly render as #upload-button-loading first;
            // never leave the visible attach control hidden from assistive tech.
            if (el.getAttribute("aria-hidden") === "true") {
                el.removeAttribute("aria-hidden");
            }
            if (el.getAttribute("aria-label") !== label) {
                el.setAttribute("aria-label", label);
            }
        }

        // Distinct loading indicator (not the attach button itself).
        const loading = document.getElementById("upload-button-loading");
        const upload = document.getElementById("upload-button");
        if (loading && loading !== upload) {
            if (loading.getAttribute("role") === "presentation") {
                loading.removeAttribute("role");
            }
            const isInteractive =
                loading.tagName === "BUTTON" ||
                Boolean(loading.querySelector("button, [role='button']"));
            if (!isInteractive) {
                if (loading.getAttribute("aria-hidden") !== "true") {
                    loading.setAttribute("aria-hidden", "true");
                }
            } else {
                if (loading.getAttribute("aria-hidden") === "true") {
                    loading.removeAttribute("aria-hidden");
                }
                const controls = loading.matches("button, [role='button']")
                    ? [loading]
                    : Array.from(loading.querySelectorAll("button, [role='button']"));
                for (const control of controls) {
                    if (
                        !control.textContent.trim() &&
                        !control.getAttribute("aria-label") &&
                        !control.getAttribute("aria-labelledby")
                    ) {
                        control.setAttribute("aria-label", "Uploading");
                    }
                }
            }
        }
    }

    function labelChatInput() {
        const input = document.getElementById("chat-input");
        if (!input) return;
        if (!input.getAttribute("aria-label") && !input.getAttribute("aria-labelledby")) {
            input.setAttribute("aria-label", "Type your message");
        }
    }

    function labelLandmarks() {
        const header = document.getElementById("header");
        if (header && header.getAttribute("role") !== "banner") {
            header.setAttribute("role", "banner");
        }
        let main = document.querySelector("main, [role='main']");
        if (!main && header && header.nextElementSibling) {
            main = header.nextElementSibling;
            main.setAttribute("role", "main");
        }
        if (main) {
            if (!main.id) {
                main.id = "main-content";
            }
            if (!main.hasAttribute("tabindex")) {
                main.setAttribute("tabindex", "-1");
            }
        }
        if (header && main && !document.getElementById("skip-to-main")) {
            const skip = document.createElement("a");
            skip.id = "skip-to-main";
            skip.href = `#${main.id}`;
            skip.className = "skip-link";
            skip.textContent = "Skip to main content";
            skip.addEventListener("click", (e) => {
                e.preventDefault();
                const target = document.getElementById(main.id);
                if (target) {
                    target.focus();
                }
            });
            header.prepend(skip);
        }
    }

    // ── Decorative 1x1 welcome logo placeholder ──────────────────────────────
    // config.toml uses a 1x1 transparent PNG to suppress Chainlit's default
    // logo; hide it from assistive tech as well as layout (style.css).

    function hidePlaceholderWelcomeLogo() {
        document
            .querySelectorAll(
                '.welcome-screen img[src^="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"]',
            )
            .forEach((img) => {
                if (img.getAttribute("alt") !== "") {
                    img.setAttribute("alt", "");
                }
                if (img.getAttribute("aria-hidden") !== "true") {
                    img.setAttribute("aria-hidden", "true");
                }
            });
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
            const describedBy = dialog.getAttribute("aria-describedby");
            if (describedBy && !document.getElementById(describedBy)) {
                dialog.removeAttribute("aria-describedby");
            }
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
    // Cookieless pageview counting. The script loads only when a valid
    // GOATCOUNTER_CODE is configured; otherwise nothing is requested.
    // See PRIVACY.md.

    const GOATCOUNTER_SCRIPT_SRC = "https://gc.zgo.at/count.js";
    const GOATCOUNTER_CODE_RE = /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/;

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

    function applyDomPatches() {
        setupEnterToSubmit();
        labelKnowledgeBaseButton();
        labelChatControls();
        labelChatInput();
        labelLandmarks();
        hidePlaceholderWelcomeLogo();
        hideReadmeDrawerTitle();
        labelKnowledgeBaseDrawer();
        setupKnowledgeBaseDrawerLabel();
        replaceBuildSha();
        manageWelcomeTitle();
    }

    // Re-apply immediately when React mounts or updates composer/header nodes
    // so accessible names and landmarks are never missing between poll ticks.
    const observerTarget = document.body || document.documentElement;
    const observerConfig = {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ["id", "role", "aria-hidden", "aria-label", "aria-labelledby"],
    };
    const domObserver = new MutationObserver(() => {
        domObserver.disconnect();
        try {
            applyDomPatches();
        } finally {
            domObserver.observe(observerTarget, observerConfig);
        }
    });

    applyDomPatches();
    if (observerTarget) {
        domObserver.observe(observerTarget, observerConfig);
    }

    // Fallback poll for React-rendered elements that appear/disappear on navigation.
    setInterval(applyDomPatches, 500);

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

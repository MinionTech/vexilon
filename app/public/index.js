/* app/public/index.js */

(function () {
    console.log("Vexilon Forensic UI Initialized");

    // ── Helpers ──────────────────────────────────────────────────────────────

    function triggerSave() {
        const textarea = document.querySelector("textarea");
        const sendBtn =
            document.querySelector('button[aria-label="Send message"]') ||
            document.querySelector("button.send-button");
        if (!textarea || !sendBtn) {
            console.warn("[vexilon] triggerSave: textarea or sendBtn not found");
            return;
        }
        const nativeSetter = Object.getOwnPropertyDescriptor(
            window.HTMLTextAreaElement.prototype,
            "value"
        ).set;
        nativeSetter.call(textarea, "__VEXILON_SAVE__");
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
        setTimeout(() => {
            if (!sendBtn.disabled) sendBtn.click();
        }, 50);
    }

    // ── Enter-to-Submit ───────────────────────────────────────────────────────

    function setupEnterToSubmit() {
        const chatInput = document.querySelector("textarea");
        if (!chatInput || chatInput.dataset.listenerAttached) return;
        chatInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                const sendBtn =
                    document.querySelector('button[aria-label="Send message"]') ||
                    document.querySelector("button.send-button");
                if (sendBtn && !sendBtn.disabled) sendBtn.click();
            }
        });
        chatInput.dataset.listenerAttached = "true";
    }

    // ── Hide Readme drawer title ──────────────────────────────────────────────

    function hideReadmeDrawerTitle() {
        document.querySelectorAll("h2").forEach((el) => {
            if (el.textContent.trim() === "Readme") el.style.display = "none";
        });
    }

    // ── Toolbar Save Button (next to paperclip) ───────────────────────────────
    // Depends on React having rendered the attach button; polled by interval.

    function setupToolbarSaveButton() {
        if (document.getElementById("vexilon-save-btn")) return;
        const attachBtn = document.querySelector('button[aria-label*="ttach"]');
        if (!attachBtn) return;
        const toolbar = attachBtn.parentElement;
        if (!toolbar) return;

        const btn = document.createElement("button");
        btn.id = "vexilon-save-btn";
        btn.title = "Save Session";
        btn.type = "button";
        btn.innerHTML =
            '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>';
        btn.style.cssText =
            "background:none;border:none;cursor:pointer;padding:4px;display:flex;align-items:center;color:inherit;opacity:0.7;";
        btn.addEventListener("mouseenter", () => (btn.style.opacity = "1"));
        btn.addEventListener("mouseleave", () => (btn.style.opacity = "0.7"));
        btn.addEventListener("click", triggerSave);
        toolbar.insertBefore(btn, attachBtn);
        console.log("[vexilon] Toolbar save button injected");
    }

    // ── Bottom-right Fixed Save Button ────────────────────────────────────────
    // Appended once to document.body. Uses a MutationObserver so we wait until
    // the React root has actually mounted (body has children) before injecting,
    // preventing React hydration from evicting the node.

    function injectBottomSaveButton() {
        if (document.getElementById("vexilon-save-btn-bottom")) return;

        const btn = document.createElement("button");
        btn.id = "vexilon-save-btn-bottom";
        btn.title = "Save Session";
        btn.type = "button";
        btn.setAttribute("aria-label", "Save Session");
        btn.innerHTML =
            '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>';

        // Explicit per-property styles to avoid cssText conflict (no double background)
        btn.style.position = "fixed";
        btn.style.bottom = "80px";   // above the Chainlit input bar
        btn.style.right = "16px";
        btn.style.zIndex = "9999";
        btn.style.background = "rgba(255,255,255,0.15)";
        btn.style.border = "1px solid rgba(255,255,255,0.3)";
        btn.style.borderRadius = "6px";
        btn.style.padding = "8px";
        btn.style.cursor = "pointer";
        btn.style.color = "inherit";
        btn.style.opacity = "0.75";
        btn.style.backdropFilter = "blur(4px)";
        btn.style.boxShadow = "0 2px 8px rgba(0,0,0,0.25)";
        btn.style.display = "flex";
        btn.style.alignItems = "center";
        btn.style.justifyContent = "center";

        btn.addEventListener("mouseenter", () => (btn.style.opacity = "1"));
        btn.addEventListener("mouseleave", () => (btn.style.opacity = "0.75"));
        btn.addEventListener("click", triggerSave);

        document.body.appendChild(btn);
        console.log("[vexilon] Bottom save button injected");
    }

    // Wait for the React root to mount before injecting the fixed button.
    // The observer fires as soon as #root (or any child of body) gets children,
    // which is after React hydration — so the node won't get wiped.
    function setupBottomSaveButton() {
        if (document.getElementById("vexilon-save-btn-bottom")) return;

        // If the React root already has children, inject immediately.
        const root = document.getElementById("root") || document.body;
        if (root.children.length > 0) {
            injectBottomSaveButton();
            return;
        }

        // Otherwise wait for the first mutation on body/root.
        const observer = new MutationObserver(() => {
            if ((document.getElementById("root") || document.body).children.length > 0) {
                observer.disconnect();
                injectBottomSaveButton();
            }
        });
        observer.observe(document.body, { childList: true, subtree: false });
    }

    // ── Build SHA ─────────────────────────────────────────────────────────────

    let buildSha = "unknown";

    fetch("/api/version")
        .then((res) => res.json())
        .then((data) => {
            if (data.version === "Dev mode") {
                buildSha = "Dev mode";
            } else {
                const shaShort = data.sha ? data.sha.substring(0, 7) : "";
                buildSha = `${data.version}${shaShort ? ` (${shaShort})` : ""}`;
            }
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

    // ── Bootstrap ─────────────────────────────────────────────────────────────

    // Bottom button: inject once, right now (or after React mounts).
    setupBottomSaveButton();

    // Poll for React-rendered elements that appear/disappear on navigation.
    setInterval(() => {
        setupEnterToSubmit();
        hideReadmeDrawerTitle();
        replaceBuildSha();
        setupToolbarSaveButton();
        // Re-inject fixed button if it was evicted (e.g. hard re-render)
        if (!document.getElementById("vexilon-save-btn-bottom")) {
            injectBottomSaveButton();
        }
    }, 500);

    setupEnterToSubmit();
    hideReadmeDrawerTitle();
    replaceBuildSha();
})();

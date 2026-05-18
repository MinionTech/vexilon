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

    // ── Build SHA ─────────────────────────────────────────────────────────────

    let buildSha = "unknown";

    fetch("/api/version")
        .then((res) => res.json())
        .then((data) => {
            const isDevSha = data.sha && data.sha.toLowerCase() === "dev mode";
            const shaShort = (!isDevSha && data.sha) ? data.sha.substring(0, 7) : "";
            
            let shaSuffix = "";
            if (isDevSha) {
                shaSuffix = "";
            } else if (shaShort) {
                shaSuffix = " (" + shaShort + ")";
            }
            
            buildSha = (data.version || "unknown") + shaSuffix;
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
                titleEl.textContent = "BCGEU Navigator";
                chatArea.prepend(titleEl);
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
        hideReadmeDrawerTitle();
        replaceBuildSha();
        manageWelcomeTitle();
    }, 500);

    setupEnterToSubmit();
    hideReadmeDrawerTitle();
    replaceBuildSha();
    manageWelcomeTitle();
})();

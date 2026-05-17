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

    // Poll for React-rendered elements that appear/disappear on navigation.
    setInterval(() => {
        setupEnterToSubmit();
        hideReadmeDrawerTitle();
        replaceBuildSha();
    }, 500);

    setupEnterToSubmit();
    hideReadmeDrawerTitle();
    replaceBuildSha();
})();

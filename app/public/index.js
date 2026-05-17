/* app/public/index.js */

(function() {
    console.log("Vexilon Forensic UI Initialized");


    /**
     * Interaction Logic: Enter-to-Submit
     * (Mandated by UI Standards Section 2.3)
     */
    function setupEnterToSubmit() {
        const chatInput = document.querySelector('textarea');
        if (!chatInput || chatInput.dataset.listenerAttached) return;

        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                const sendBtn = document.querySelector('button[aria-label="Send message"]') || 
                                document.querySelector('button.send-button');
                if (sendBtn && !sendBtn.disabled) {
                    sendBtn.click();
                }
            }
        });
        
        chatInput.dataset.listenerAttached = "true";
    }

    function hideReadmeDrawerTitle() {
        document.querySelectorAll('h2').forEach(el => {
            if (el.textContent.trim() === 'Readme') el.style.display = 'none';
        });
    }

    let cachedVersionInfo = null;

    // Fetch version info dynamically from endpoint
    fetch('/api/version')
        .then(res => res.json())
        .then(data => {
            if (data.version === "Dev mode") {
                cachedVersionInfo = "Dev mode";
            } else {
                const shaShort = data.sha ? data.sha.substring(0, 7) : "";
                cachedVersionInfo = `${data.version}${shaShort ? ` (${shaShort})` : ""}`;
            }
            // Trigger immediately if container is already in the DOM
            const container = document.querySelector('.input-footer-container');
            if (container && !document.getElementById('input-footer-ver-node')) {
                const verSpan = document.createElement('span');
                verSpan.id = 'input-footer-ver-node';
                verSpan.className = 'input-footer-separator';
                verSpan.textContent = '•';
                
                const valSpan = document.createElement('span');
                valSpan.className = 'input-footer-version';
                valSpan.textContent = cachedVersionInfo;
                
                container.appendChild(verSpan);
                container.appendChild(valSpan);
            }
        })
        .catch(err => console.error("Error fetching version:", err));

    function setupInputFooter() {
        document.querySelectorAll('div[role="article"]').forEach(el => {
            if (el.textContent.trim().includes('LLMs can make mistakes') && !el.dataset.footerInjected) {
                const versionSegment = cachedVersionInfo 
                    ? `<span class="input-footer-separator">•</span><span class="input-footer-version">${cachedVersionInfo}</span>` 
                    : "";
                
                el.innerHTML = `
                    <div class="input-footer-container">
                        <a href="https://github.com/MinionTech/vexilon" class="input-footer-link" target="_blank">Source Code</a>
                        <span class="input-footer-separator">•</span>
                        <a href="/public/docs/PRIVACY.md" class="input-footer-link" target="_blank">Privacy Policy</a>
                        ${versionSegment}
                    </div>
                `;
                el.dataset.footerInjected = "true";
            }
        });
    }

    // Run periodically to catch re-renders
    setInterval(() => {
        setupEnterToSubmit();
        hideReadmeDrawerTitle();
        setupInputFooter();
    }, 500);

    setupEnterToSubmit();
    hideReadmeDrawerTitle();
    setupInputFooter();
})();

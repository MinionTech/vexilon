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
            const container = document.getElementById('vexilon-input-footer');
            if (container && !document.getElementById('input-footer-ver-node')) {
                const verSpan = document.createElement('span');
                verSpan.id = 'input-footer-ver-node';
                verSpan.className = 'input-footer-separator';
                verSpan.textContent = '•';
                
                const valSpan = document.createElement('a');
                valSpan.href = 'https://github.com/MinionTech/vexilon/pkgs/container/vexilon%2Fagnav';
                valSpan.target = '_blank';
                valSpan.className = 'input-footer-link input-footer-version';
                valSpan.textContent = cachedVersionInfo;
                
                container.appendChild(verSpan);
                container.appendChild(valSpan);
            }
        })
        .catch(err => console.error("Error fetching version:", err));

    function setupInputFooter() {
        const disclaimer = document.querySelector('div[role="article"]');
        if (!disclaimer) return;

        const parent = disclaimer.parentNode;
        if (!parent || document.getElementById('vexilon-input-footer')) return;

        const footer = document.createElement('div');
        footer.id = 'vexilon-input-footer';
        
        const versionSegment = cachedVersionInfo 
            ? `<span class="input-footer-separator">•</span><a href="https://github.com/MinionTech/vexilon/pkgs/container/vexilon%2Fagnav" class="input-footer-link input-footer-version" target="_blank">${cachedVersionInfo}</a>` 
            : "";
        
        footer.innerHTML = `
            <a href="https://github.com/MinionTech/vexilon" class="input-footer-link" target="_blank">Source Code</a>
            <span class="input-footer-separator">•</span>
            <a href="/public/docs/PRIVACY.md" class="input-footer-link" target="_blank">Privacy Policy</a>
            ${versionSegment}
        `;
        
        parent.insertBefore(footer, disclaimer.nextSibling);
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

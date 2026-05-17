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

    function setupInputFooter() {
        document.querySelectorAll('div[role="article"]').forEach(el => {
            if (el.textContent.trim().includes('LLMs can make mistakes') && !el.dataset.footerInjected) {
                el.innerHTML = `
                    <a href="https://github.com/MinionTech/vexilon" class="input-footer-link" target="_blank">Source Code</a>
                    <span class="input-footer-separator">•</span>
                    <a href="/public/docs/PRIVACY.md" class="input-footer-link" target="_blank">Privacy Policy</a>
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

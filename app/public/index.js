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

    /**
     * Rebranding: Rename 'Readme' to 'Knowledge Base'
     */
    function renameReadme() {
        const elements = document.querySelectorAll('button, a, span, p');
        elements.forEach(el => {
            if (el.textContent.trim() === 'Readme') {
                el.textContent = 'Knowledge Base';
            }
        });
    }

    // Run periodically to catch re-renders
    setInterval(() => {
        setupEnterToSubmit();
        renameReadme();
    }, 1000);

    setupEnterToSubmit();
    renameReadme();
    buildKbSidebar();
})();

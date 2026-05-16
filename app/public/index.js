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

    function hideDisclaimer() {
        document.querySelectorAll('div[role="article"]').forEach(el => {
            if (el.textContent.trim() === 'LLMs can make mistakes. Check important info.') el.style.display = 'none';
        });
    }

    /**
     * Save/Load Buttons
     * Add Save and Load buttons beside the settings gear in the bottom-left UI area
     */
    function setupSaveLoadButtons() {
        // Check if buttons already exist to avoid duplication
        if (document.getElementById('save-conversation-btn')) return;

        // Find the gear icon container (settings button area)
        const settingsBtn = document.querySelector('[aria-label="Open settings"]');
        if (!settingsBtn) return;

        const container = settingsBtn.parentElement || settingsBtn.closest('[class*="bottom"]');
        if (!container) return;

        // Create Save button
        const saveBtn = document.createElement('button');
        saveBtn.id = 'save-conversation-btn';
        saveBtn.textContent = 'Save';
        saveBtn.title = 'Save conversation to file';
        saveBtn.style.cssText = `
            padding: 6px 12px;
            margin-right: 8px;
            background: none;
            border: 1px solid #ccc;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.875rem;
            color: #2c3e50;
        `;
        saveBtn.addEventListener('click', () => {
            window.chainlit?.callAction({id: 'save_conversation', payload: {}});
        });

        // Create Load button
        const loadBtn = document.createElement('button');
        loadBtn.id = 'load-conversation-btn';
        loadBtn.textContent = 'Load';
        loadBtn.title = 'Load conversation from file';
        loadBtn.style.cssText = `
            padding: 6px 12px;
            background: none;
            border: 1px solid #ccc;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.875rem;
            color: #2c3e50;
        `;
        loadBtn.addEventListener('click', () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.addEventListener('change', (e) => {
                const file = e.target.files?.[0];
                if (!file) return;
                const reader = new FileReader();
                reader.addEventListener('load', (event) => {
                    const content = event.target?.result;
                    if (typeof content === 'string') {
                        window.chainlit?.callAction({
                            id: 'load_conversation',
                            payload: {files: [content]}
                        });
                    }
                });
                reader.readAsText(file);
            });
            input.click();
        });

        // Insert buttons before settings button (or in the same container)
        if (container && container !== settingsBtn) {
            container.insertBefore(loadBtn, settingsBtn);
            container.insertBefore(saveBtn, settingsBtn);
        } else {
            settingsBtn.parentElement?.insertBefore(loadBtn, settingsBtn);
            settingsBtn.parentElement?.insertBefore(saveBtn, settingsBtn);
        }
    }

    // Run periodically to catch re-renders
    setInterval(() => {
        setupEnterToSubmit();
        hideReadmeDrawerTitle();
        hideDisclaimer();
        setupSaveLoadButtons();
    }, 500);

    setupEnterToSubmit();
    hideReadmeDrawerTitle();
    hideDisclaimer();
    setupSaveLoadButtons();
})();

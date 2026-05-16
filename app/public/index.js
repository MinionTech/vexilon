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
     * Chainlit's default settings location is "message_composer" (bottom of chat area)
     */
    function setupSaveLoadButtons() {
        // Check if buttons already exist to avoid duplication
        if (document.getElementById('save-conversation-btn')) return;

        // Try to find the message composer / settings area
        // Chainlit places settings buttons in a toolbar near the message input
        let container = null;

        // Strategy 1: Find by looking for elements near the textarea
        const textarea = document.querySelector('textarea');
        if (textarea) {
            // Walk up the DOM to find a button container (usually parent has buttons)
            let elem = textarea;
            for (let i = 0; i < 5; i++) {
                elem = elem.parentElement;
                if (!elem) break;
                // Check if this element contains buttons (likely a toolbar)
                const buttons = elem.querySelectorAll('button');
                if (buttons.length > 0) {
                    container = elem;
                    break;
                }
            }
        }

        // Strategy 2: Look for common Chainlit button container classes
        if (!container) {
            const selectors = [
                '[class*="composer"]',
                '[class*="message-input"]',
                '[class*="actions"]',
                'footer',
            ];
            for (const sel of selectors) {
                const elem = document.querySelector(sel);
                if (elem && elem.querySelectorAll('button').length > 0) {
                    container = elem;
                    break;
                }
            }
        }

        if (!container) {
            console.log('[Save/Load] Container not found yet (UI may still be initializing)');
            return;
        }

        console.log('[Save/Load] Found container, injecting buttons');

        // Create a wrapper div for our buttons (to group them and control layout)
        const buttonWrapper = document.createElement('div');
        buttonWrapper.id = 'save-load-button-wrapper';
        buttonWrapper.style.cssText = `
            display: inline-flex !important;
            gap: 6px !important;
            margin-right: 8px !important;
            height: auto !important;
        `;

        // Create button styles (must override Chainlit's aggressive styling)
        const buttonStyle = `
            padding: 6px 10px !important;
            min-width: 50px !important;
            width: auto !important;
            height: auto !important;
            background: #fff !important;
            border: 1px solid #ccc !important;
            border-radius: 4px !important;
            cursor: pointer !important;
            font-size: 0.8rem !important;
            color: #2c3e50 !important;
            transition: all 0.2s !important;
            white-space: nowrap !important;
            flex-shrink: 0 !important;
            display: inline-block !important;
        `;

        // Create Save button
        const saveBtn = document.createElement('button');
        saveBtn.id = 'save-conversation-btn';
        saveBtn.type = 'button';
        saveBtn.textContent = 'Save';
        saveBtn.title = 'Save conversation to file';
        saveBtn.style.cssText = buttonStyle;
        saveBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('[Save/Load] Save clicked');
            window.chainlit?.callAction({id: 'save_conversation', payload: {}});
        });
        saveBtn.addEventListener('mouseover', () => {
            saveBtn.style.backgroundColor = '#f5f5f5 !important';
        });
        saveBtn.addEventListener('mouseout', () => {
            saveBtn.style.backgroundColor = '#fff !important';
        });

        // Create Load button
        const loadBtn = document.createElement('button');
        loadBtn.id = 'load-conversation-btn';
        loadBtn.type = 'button';
        loadBtn.textContent = 'Load';
        loadBtn.title = 'Load conversation from file';
        loadBtn.style.cssText = buttonStyle;
        loadBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('[Save/Load] Load clicked');
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.addEventListener('change', (changeEvent) => {
                const file = changeEvent.target.files?.[0];
                if (!file) return;
                console.log('[Save/Load] File selected:', file.name);
                const reader = new FileReader();
                reader.addEventListener('load', (loadEvent) => {
                    const content = loadEvent.target?.result;
                    if (typeof content === 'string') {
                        console.log('[Save/Load] File loaded, calling load_conversation');
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
        loadBtn.addEventListener('mouseover', () => {
            loadBtn.style.backgroundColor = '#f5f5f5 !important';
        });
        loadBtn.addEventListener('mouseout', () => {
            loadBtn.style.backgroundColor = '#fff !important';
        });

        // Add buttons to wrapper
        buttonWrapper.appendChild(saveBtn);
        buttonWrapper.appendChild(loadBtn);

        // Prepend wrapper to container (before other buttons)
        container.insertBefore(buttonWrapper, container.firstChild);
        console.log('[Save/Load] Buttons injected successfully');
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

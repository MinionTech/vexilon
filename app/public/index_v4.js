/* app/public/index_v4.js */

(function() {
    console.log("Vexilon Forensic UI Initialized (v4)");

    /**
     * Rebranding: Hide standard Chainlit artifacts and UI fluff.
     */
    function sanitizeUI() {
        // 1. Hide LLM disclaimer (Forensic precision means we stand by our grounding)
        document.querySelectorAll('div[role="article"]').forEach(el => {
            if (el.textContent.trim().includes('LLMs can make mistakes')) {
                el.style.display = 'none';
            }
        });

        // 2. Ensure "Readme" drawer titles are hidden (CSS handles the button itself)
        document.querySelectorAll('h2').forEach(el => {
            if (el.textContent.trim() === 'Readme') {
                el.textContent = 'Knowledge Base';
            }
        });
    }

    /**
     * Save/Load Buttons
     * Injects forensic session control buttons into the composer toolbar.
     */
    function setupSaveLoadButtons() {
        if (document.getElementById('save-load-button-wrapper')) return;

        // Target the composer toolbar (area with settings gear and send button)
        const selectors = [
            '.cl-composer-toolbar',
            '[class*="composer"] [class*="toolbar"]',
            'div[role="presentation"] footer',
        ];
        
        let container = null;
        for (const sel of selectors) {
            container = document.querySelector(sel);
            if (container) break;
        }

        if (!container) return;

        const wrapper = document.createElement('div');
        wrapper.id = 'save-load-button-wrapper';
        
        const saveBtn = document.createElement('button');
        saveBtn.id = 'save-conversation-btn';
        saveBtn.type = 'button';
        saveBtn.textContent = 'Save';
        saveBtn.title = 'Save conversation (Forensic Export)';
        saveBtn.onclick = (e) => {
            e.preventDefault();
            window.chainlit?.callAction({id: 'save_conversation', payload: {}});
        };

        const loadBtn = document.createElement('button');
        loadBtn.id = 'load-conversation-btn';
        loadBtn.type = 'button';
        loadBtn.textContent = 'Load';
        loadBtn.title = 'Load conversation (Restore Session)';
        loadBtn.onclick = (e) => {
            e.preventDefault();
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.md,.json';
            input.onchange = (ev) => {
                const file = ev.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = (re) => {
                    const content = re.target.result;
                    window.chainlit?.callAction({
                        id: 'load_conversation',
                        payload: {files: [content]}
                    });
                };
                reader.readAsText(file);
            };
            input.click();
        };

        wrapper.appendChild(saveBtn);
        wrapper.appendChild(loadBtn);
        
        // Insert at the start of the toolbar
        container.insertBefore(wrapper, container.firstChild);
    }

    /**
     * Custom Forensic Footer
     * Injects a clean, centered footer at the bottom of the viewport.
     */
    function setupFooter() {
        if (document.getElementById('forensic-footer')) return;

        const footer = document.createElement('div');
        footer.id = 'forensic-footer';
        footer.innerHTML = `
            <a href="https://github.com/MinionTech/vexilon" target="_blank">Source Code</a>
            <span class="footer-separator">•</span>
            <a href="/public/docs/PRIVACY.md" target="_blank">Privacy Policy</a>
        `;
        
        document.body.appendChild(footer);
    }

    // Initialize MutationObserver for reactive UI elements
    const observer = new MutationObserver(() => {
        sanitizeUI();
        setupSaveLoadButtons();
        setupFooter();
    });

    observer.observe(document.body, { childList: true, subtree: true });

    // Initial run
    sanitizeUI();
    setupSaveLoadButtons();
    setupFooter();
})();

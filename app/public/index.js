/* app/public/index.js */

(function() {
    console.log("AgNav UI Sync Initialized");

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

    function injectPersonaSelector() {
        // Find the header - use a broad selector to catch Chainlit's variants
        const header = document.querySelector('.header') || document.querySelector('header');
        if (!header || document.querySelector('#persona-selector-container')) return;

        const container = document.createElement('div');
        container.id = 'persona-selector-container';
        
        const label = document.createElement('span');
        label.id = 'persona-selector-label';
        label.innerText = 'PERSONA';

        const select = document.createElement('select');
        select.id = 'persona-selector';
        
        const personas = ['Lookup', 'Grieve', 'Audit', 'Manage'];
        personas.forEach(mode => {
            const option = document.createElement('option');
            option.value = mode;
            option.text = mode;
            select.appendChild(option);
        });

        // Initialize with session state if possible
        select.addEventListener('change', (e) => {
            const newMode = e.target.value;
            console.log("Switching persona to:", newMode);
            syncWithChainlitSettings(newMode);
        });

        container.appendChild(label);
        container.appendChild(select);
        
        // Find the title element or just append to start
        const title = header.querySelector('.title') || header.querySelector('h1') || header.firstChild;
        if (title && title.parentNode === header) {
            header.insertBefore(container, title.nextSibling);
        } else {
            header.appendChild(container);
        }
        
        console.log("Persona selector injected into header");
    }

    function syncWithChainlitSettings(mode) {
        // 1. Find the settings button
        const gearIcon = document.querySelector('#chat-settings-open-modal');
        if (!gearIcon) {
            console.error('Chat settings button not found. Is it hidden?');
            return;
        }
        
        // 2. Click it
        gearIcon.click();

        // 3. Wait for the modal and the select element
        let attempts = 0;
        const findAndChange = setInterval(() => {
            const modalSelect = document.querySelector('#Persona'); // ID from ChatSettings
            if (modalSelect) {
                modalSelect.value = mode;
                modalSelect.dispatchEvent(new Event('change', { bubbles: true }));
                
                // 4. Close the modal after a short delay
                setTimeout(() => {
                    const closeBtn = document.querySelector('button[aria-label="close"]') || document.querySelector('.MuiBackdrop-root');
                    if (closeBtn) closeBtn.click();
                    else gearIcon.click(); // Toggle it off
                }, 400);
                
                clearInterval(findAndChange);
            }
            if (++attempts > 40) {
                console.error("Timed out waiting for settings modal");
                clearInterval(findAndChange);
            }
        }, 50);
    }

    function injectFooter() {
        if (document.querySelector('#agnav-footer')) return;
        
        // Find the main container or body
        const mainContainer = document.querySelector('main') || document.body;
        if (!mainContainer) return;

        const footer = document.createElement('div');
        footer.id = 'agnav-footer';
        
        const githubLink = document.createElement('a');
        githubLink.href = 'https://github.com/MinionTech/vexilon';
        githubLink.target = '_blank';
        githubLink.innerText = 'GitHub';
        
        const privacyLink = document.createElement('a');
        privacyLink.href = 'https://github.com/MinionTech/vexilon/blob/main/app/docs/PRIVACY.md';
        privacyLink.target = '_blank';
        privacyLink.innerText = 'Privacy';
        
        const versionSpan = document.createElement('span');
        versionSpan.innerText = 'v2026.05.15'; // Hardcoded for forensic stability
        
        const dot1 = document.createTextNode(' • ');
        const dot2 = document.createTextNode(' • ');
        
        footer.appendChild(githubLink);
        footer.appendChild(dot1);
        footer.appendChild(privacyLink);
        footer.appendChild(dot2);
        footer.appendChild(versionSpan);
        
        // Append to body to ensure it stays at bottom
        document.body.appendChild(footer);
        console.log("Custom AgNav footer injected");
    }

    function updateWatermark() {
        const watermark = document.querySelector('.watermark');
        if (watermark) {
            watermark.innerHTML = `
                <a href="https://github.com/MinionTech/vexilon" target="_blank">GitHub</a> • 
                <a href="https://github.com/MinionTech/vexilon/blob/main/app/docs/PRIVACY.md" target="_blank">Privacy</a> • 
                <span>v2026.05.15</span>
            `;
        }
    }

    function killSparkles() {
        document.querySelectorAll('span').forEach(span => {
            const rect = span.getBoundingClientRect();
            if (rect.width > 0 && rect.width < 40 && rect.height > 0 && rect.height < 40) {
                if (span.innerText.trim() === '') {
                    span.style.display = 'none';
                    span.style.visibility = 'hidden';
                    span.style.width = '0';
                    span.style.height = '0';
                }
            }
        });
        // Targeted sparkle suppression only
        document.querySelectorAll('.sparkle, [class*="sparkle"]').forEach(sparkle => {
            sparkle.style.display = 'none';
        });
    }

    // Run periodically to catch re-renders
    setInterval(() => {
        setupEnterToSubmit();
        injectPersonaSelector();
        updateWatermark();
        killSparkles();
    }, 100);
    setupEnterToSubmit();
    injectPersonaSelector();
    updateWatermark();
    killSparkles();
})();

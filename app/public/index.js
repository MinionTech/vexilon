/* app/public/index.js */

(function() {
    console.log("AgNav UI Sync Initialized");

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
        }, 50).catch(err => console.error(err));
    }

    // Run periodically to catch re-renders
    setInterval(injectPersonaSelector, 1000);
    injectPersonaSelector();
})();

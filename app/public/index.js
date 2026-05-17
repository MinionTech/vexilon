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

    let buildSha = "unknown";

    // Fetch version info dynamically from endpoint
    fetch('/api/version')
        .then(res => res.json())
        .then(data => {
            if (data.version === "Dev mode") {
                buildSha = "Dev mode";
            } else {
                const shaShort = data.sha ? data.sha.substring(0, 7) : "";
                buildSha = `${data.version}${shaShort ? ` (${shaShort})` : ""}`;
            }
            replaceBuildSha();
        })
        .catch(err => console.error("Error fetching version:", err));

    function replaceBuildSha() {
        document.querySelectorAll('code, span, p, li, a').forEach(el => {
            if (el.textContent.includes('{{BUILD_SHA}}')) {
                el.innerHTML = el.innerHTML.replace('{{BUILD_SHA}}', buildSha);
            }
        });
    }
    function setupFileAutoSubmit() {
        if (window.fileAutoSubmitIntervalAttached) return;

        let autoSubmitPending = false;
        
        setInterval(() => {
            const inputArea = document.querySelector('#chat-input') || document.querySelector('form') || document.querySelector('.MuiFormControl-root');
            if (!inputArea) return;
            
            // Clone the DOM node to extract text EXCEPT for what the user is typing in the textarea
            // This ensures we only trigger on actual UI File Chips and not if the user manually types ".md"
            const cloned = inputArea.cloneNode(true);
            cloned.querySelectorAll('textarea').forEach(t => t.remove());
            const text = cloned.textContent?.toLowerCase() || '';
            
            const hasFile = text.includes('.md') || text.includes('.json');
            
            if (hasFile) {
                const sendBtn = document.getElementById('send-button') || 
                                document.querySelector('button[id="send-button"]') ||
                                document.querySelector('button[aria-label="Send message"]') ||
                                document.querySelector('button[type="submit"]');
                                
                if (sendBtn && !sendBtn.disabled) {
                    if (!autoSubmitPending) {
                        autoSubmitPending = true;
                        sendBtn.click();
                        
                        // Fallback execution if the button click doesn't bubble correctly in React
                        setTimeout(() => {
                            const textarea = inputArea.querySelector('textarea');
                            if (textarea) {
                                textarea.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true }));
                            }
                        }, 50);
                    }
                }
            } else {
                // Reset state when the file chip disappears (after successful send)
                autoSubmitPending = false;
            }
        }, 200);

        window.fileAutoSubmitIntervalAttached = true;
    }

    // Run periodically to catch re-renders
    setInterval(() => {
        setupEnterToSubmit();
        hideReadmeDrawerTitle();
        replaceBuildSha();
        setupFileAutoSubmit();
    }, 100);

    setupEnterToSubmit();
    hideReadmeDrawerTitle();
    replaceBuildSha();
})();

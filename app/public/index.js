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
    /**
     * Interaction Logic: Session Save/Load Click Handlers via clinical `/persist` command pipeline
     */
    function setupSessionClickHandlers() {
        if (document.datasetSessionHandlersAttached) return;
        
        document.addEventListener('click', (e) => {
            const target = e.target.closest('a');
            if (!target) return;

            const href = target.getAttribute('href');
            if (href === '#save') {
                e.preventDefault();
                console.log("[session] Triggering save_conversation via clinical pipeline...");
                submitTechnicalCommand('/persist save');
            } else if (href === '#load') {
                e.preventDefault();
                console.log("[session] Prompting for file upload...");
                triggerFilePicker();
            }
        });
        
        document.datasetSessionHandlersAttached = "true";
    }

    function submitTechnicalCommand(command) {
        const chatInput = document.getElementById('chat-input') || document.querySelector('textarea');
        if (!chatInput) {
            alert("System notice: Unable to find chat input text area. Please make sure the chat tab is active.");
            return;
        }
        
        // React 16+ setter bypass hack
        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
        if (nativeInputValueSetter) {
            nativeInputValueSetter.call(chatInput, command);
        } else {
            chatInput.value = command; // fallback
        }
        chatInput.dispatchEvent(new Event('input', { bubbles: true }));
        
        setTimeout(() => {
            const sendBtn = document.getElementById('send-button') || 
                            document.querySelector('button[aria-label="Send message"]') || 
                            document.querySelector('button.send-button');
            if (sendBtn && !sendBtn.disabled) {
                sendBtn.click();
            } else {
                const enterEvent = new KeyboardEvent('keydown', {
                    key: 'Enter',
                    code: 'Enter',
                    keyCode: 13,
                    which: 13,
                    bubbles: true
                });
                chatInput.dispatchEvent(enterEvent);
            }
        }, 50);
    }

    function triggerFilePicker() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.md,.MD,.mD,.Md';
        
        input.addEventListener('change', (e) => {
            const file = e.target.files?.[0];
            if (!file) return;

            // Strict case-insensitive extension check
            const suffixMatch = file.name.match(/\.[^.]+$/);
            const suffix = suffixMatch ? suffixMatch[0].toLowerCase() : "";
            if (suffix !== ".md") {
                alert(`Upload failed: The file "${file.name}" is not a markdown file. Only markdown (.md) session files are accepted.`);
                return;
            }

            const reader = new FileReader();
            reader.addEventListener('load', (event) => {
                const content = event.target?.result;
                if (typeof content !== 'string') {
                    alert("Upload failed: Unable to read file content as a string.");
                    return;
                }

                // Client-side pre-flight validation
                if (!content.includes('Technical Metadata (JSON)') && !content.includes('```json')) {
                    alert("Upload failed: The selected file does not contain valid technical session metadata.");
                    return;
                }

                console.log(`[session] File parsed successfully, sending to backend: ${file.name}`);
                submitTechnicalCommand(`/persist load ${content}`);
            });

            reader.addEventListener('error', () => {
                alert("Upload failed: A disk error occurred while reading the selected file.");
            });

            reader.readAsText(file);
        });

        input.click();
    }

    function hidePersistCommands() {
        document.querySelectorAll('.message, .message-user, .message-content').forEach(el => {
            if (el.textContent.includes('/persist')) {
                const messageContainer = el.closest('.message');
                if (messageContainer && messageContainer.style.display !== 'none') {
                    messageContainer.style.display = 'none';
                }
            }
        });
    }

    // Run periodically to catch re-renders
    setInterval(() => {
        setupEnterToSubmit();
        hideReadmeDrawerTitle();
        replaceBuildSha();
        setupSessionClickHandlers();
        hidePersistCommands();
    }, 100);

    setupEnterToSubmit();
    hideReadmeDrawerTitle();
    replaceBuildSha();
    setupSessionClickHandlers();
    hidePersistCommands();
})();

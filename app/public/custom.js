// Chainlit Custom JS Bridge (AgNav Forensic Version v33 - Perfect Alignment)
console.log("[AgNav] Initializing Forensic UI v33...");

// Signal to forensics
document.title = "BCGEU Navigator [Forensic v33]";

function toggleSidebar() {
    const sidebar = document.getElementById('agnav-forensic-sidebar');
    const main = document.querySelector('main');
    if (!sidebar) return;

    if (sidebar.style.right === '0px') {
        sidebar.style.right = '-300px';
        if (main) main.style.marginRight = '0px';
        localStorage.setItem('agnav-sidebar-state', 'closed');
    } else {
        sidebar.style.right = '0px';
        if (main) main.style.marginRight = '300px';
        localStorage.setItem('agnav-sidebar-state', 'open');
    }
}

function injectSidebar() {
    if (document.getElementById('agnav-forensic-sidebar')) return;

    const sidebar = document.createElement('div');
    sidebar.id = 'agnav-forensic-sidebar';
    const savedState = localStorage.getItem('agnav-sidebar-state') || 'open';
    
    sidebar.style.cssText = `
        position: fixed;
        top: 0;
        right: ${savedState === 'open' ? '0px' : '-300px'};
        width: 300px;
        height: 100vh;
        background-color: #1a1a1a;
        border-left: 1px solid #333;
        z-index: 2000;
        padding: 20px;
        box-sizing: border-box;
        color: #fff;
        overflow-y: auto;
        font-family: Inter, system-ui, -apple-system, sans-serif;
        transition: right 0.3s ease;
        box-shadow: -5px 0 15px rgba(0,0,0,0.3);
    `;

    sidebar.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #333; padding-bottom:10px; margin-bottom:20px;">
            <h2 style="margin:0; font-size:1.1rem; letter-spacing:0.5px;">Forensic Toolbox</h2>
            <button onclick="toggleSidebar()" style="background:transparent; border:none; color:#888; cursor:pointer; font-size:1.2rem;">&times;</button>
        </div>
        
        <div>
            <h3 style="font-size:0.8rem; color:#666; text-transform:uppercase; letter-spacing:1px; margin-bottom:15px;">Reference Documents</h3>
            <ul style="list-style:none; padding:0; margin:0;">
                <li style="margin-bottom:12px;"><a href="/knowledge-base/01_primary/BCGEU%2019th%20Main%20Agreement.pdf" target="_blank" style="color:#4a90e2; text-decoration:none; font-size:0.9rem;">19th Main Agreement</a></li>
                <li style="margin-bottom:12px;"><a href="/knowledge-base/01_primary/BC%20Labour%20Relations%20Code.pdf" target="_blank" style="color:#4a90e2; text-decoration:none; font-size:0.9rem;">Labour Relations Code</a></li>
                <li style="margin-bottom:12px;"><a href="/knowledge-base/04_jurisprudence/Nexus%20Test%20and%20Off-Duty%20Conduct.pdf" target="_blank" style="color:#4a90e2; text-decoration:none; font-size:0.9rem;">Nexus Test (PDF)</a></li>
            </ul>
        </div>
    `;

    document.body.appendChild(sidebar);

    const main = document.querySelector('main');
    if (main && savedState === 'open') {
        main.style.marginRight = '300px';
    }
}

function forceLayout() {
    const header = document.querySelector('header') || document.querySelector('[role="banner"]') || document.querySelector('.cl-header');
    if (!header) return;

    // 1. Target the true layout container
    const container = header.querySelector('div') || header;
    container.style.setProperty('display', 'flex', 'important');
    container.style.setProperty('align-items', 'center', 'important');
    container.style.setProperty('justify-content', 'flex-start', 'important');

    // 2. New Chat Button (Far Left)
    const newChatBtn = document.getElementById('new-chat-button');
    if (newChatBtn) {
        newChatBtn.style.setProperty('order', '1', 'important');
        newChatBtn.style.setProperty('margin-left', '15px', 'important');
        newChatBtn.style.setProperty('margin-right', '0', 'important');
    }

    // 3. Persona Selector (Perspective Box) - Aligned with Chat Title
    let persona = document.getElementById('agnav-header-persona');
    if (!persona) {
        persona = document.createElement('div');
        persona.id = 'agnav-header-persona';
        persona.innerHTML = `
            <select id="persona-selector-header" style="background:transparent; color:#fff; border:none; font-size:0.85rem; cursor:pointer; outline:none; font-weight:600;">
                <option value="Lookup">Lookup</option>
                <option value="Grieve">Grieve</option>
                <option value="Manage">Manage</option>
            </select>
        `;
        persona.querySelector('select').onchange = (e) => {
            // Trigger the native ChatProfile selector if possible, or reload with param
            window.location.href = window.location.origin + '?profile=' + e.target.value;
        };
        container.appendChild(persona);
    }

    persona.style.setProperty('order', '2', 'important');
    // Precision Alignment: Title is at X=119. NewChat(15) + Width(~40) = 55. 
    // Gap needed = 119 - 55 = 64px.
    persona.style.setProperty('margin-left', '64px', 'important'); 
    persona.style.setProperty('display', 'flex', 'important');
    persona.style.setProperty('align-items', 'center', 'important');
    persona.style.setProperty('padding', '4px 12px', 'important');
    persona.style.setProperty('background', '#222', 'important');
    persona.style.setProperty('border', '1px solid #444', 'important');
    persona.style.setProperty('border-radius', '4px', 'important');

    // 4. BCGEU Navigator Title (Header version - we might hide this to avoid double title)
    const titleEl = Array.from(container.querySelectorAll('span, p, div')).find(el => el.innerText === "BCGEU Navigator");
    if (titleEl) {
        titleEl.style.setProperty('display', 'none', 'important'); // Hide header title as it's in the chat area
    }

    // 5. Toolbox Toggle (Far Right)
    let toolbox = document.getElementById('agnav-sidebar-toggle');
    if (!toolbox) {
        toolbox = document.createElement('button');
        toolbox.id = 'agnav-sidebar-toggle';
        toolbox.innerText = 'Toolbox';
        toolbox.onclick = toggleSidebar;
        container.appendChild(toolbox);
    }

    toolbox.style.setProperty('order', '10', 'important');
    toolbox.style.setProperty('margin-left', 'auto', 'important');
    toolbox.style.setProperty('margin-right', '15px', 'important');
    toolbox.style.setProperty('padding', '6px 12px', 'important');
    toolbox.style.setProperty('background', '#4a90e2', 'important');
    toolbox.style.setProperty('color', '#fff', 'important');
    toolbox.style.setProperty('border', 'none', 'important');
    toolbox.style.setProperty('border-radius', '4px', 'important');
    toolbox.style.setProperty('cursor', 'pointer', 'important');
    toolbox.style.setProperty('font-size', '0.85rem', 'important');
    toolbox.style.setProperty('font-weight', '600', 'important');

    // 6. Cleanup Unwanted Header Links (Source Code, Privacy)
    const unwantedTerms = ["Source Code", "Privacy Policy", "Readme", "GitHub"];
    Array.from(header.querySelectorAll('a, button')).forEach(el => {
        const text = el.innerText || "";
        if (unwantedTerms.some(term => text.includes(term))) {
            if (el.id !== 'agnav-sidebar-toggle') { // Don't hide our own toggle
                el.style.setProperty('display', 'none', 'important');
            }
        }
    });

    // Hide theme toggle
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) themeToggle.style.setProperty('display', 'none', 'important');
}

// ── Custom Footer ─────────────────────────────────────────────────────────────
const _FOOTER = {
    repo_url:    'https://github.com/MinionTech/vexilon',
    privacy_url: 'https://github.com/MinionTech/vexilon/blob/main/docs/PRIVACY.md',
    pkg_url:     'https://github.com/MinionTech/vexilon/pkgs/container/agnav',
    version:     'v17-restored',
};

function injectFooter() {
    const disclaimers = Array.from(document.querySelectorAll('p, span, div'))
        .filter(el =>
            el.childElementCount === 0 &&
            el.innerText &&
            el.innerText.includes('LLMs can make mistakes')
        );

    const footerHTML = [
        `<a href="${_FOOTER.repo_url}" target="_blank" rel="noopener" style="color:#4a90e2;text-decoration:none;">GitHub</a>`,
        `<span style="color:#555;margin:0 6px;">·</span>`,
        `<a href="${_FOOTER.privacy_url}" target="_blank" rel="noopener" style="color:#4a90e2;text-decoration:none;">Privacy</a>`,
        `<span style="color:#555;margin:0 6px;">·</span>`,
        `<a href="${_FOOTER.pkg_url}" target="_blank" rel="noopener" style="color:#4a90e2;text-decoration:none;font-family:monospace;">${_FOOTER.version}</a>`,
    ].join('');

    disclaimers.forEach(el => {
        if (el.dataset.agnavFooter) return;
        el.dataset.agnavFooter = '1';
        el.innerHTML = footerHTML;
        el.style.cssText = 'text-align:center;font-size:0.78rem;opacity:0.75;padding:4px 0;';
    });
}

const observer = new MutationObserver(() => {
    injectSidebar();
    forceLayout();
    injectFooter();
});

observer.observe(document.body, { childList: true, subtree: true });
setInterval(() => {
    injectSidebar();
    forceLayout();
    injectFooter();
}, 500);

console.log("[AgNav] Forensic UI v33 ready.");

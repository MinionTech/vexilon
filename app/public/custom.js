// Chainlit Custom JS Bridge (AgNav Forensic Version v13 - THE SIDEBAR)
console.log("[AgNav] Initializing Forensic Sidebar v13...");

// Signal to forensics
document.title = "BCGEU Navigator [Forensic v13]";

async function triggerForensicAction(action) {
    console.log(`[AgNav] Triggering action: ${action}`);
    try {
        // We use the Python Action buttons in the chat for these
        const buttons = Array.from(document.querySelectorAll('button'));
        const target = buttons.find(b => b.innerText.trim().includes(action));
        if (target) {
            target.click();
        } else {
            console.warn(`[AgNav] Action button '${action}' not found in chat history.`);
        }
    } catch (err) {
        console.error(`[AgNav] Action failed: ${action}`, err);
    }
}

function injectSidebar() {
    if (document.getElementById('agnav-forensic-sidebar')) return;

    const sidebar = document.createElement('div');
    sidebar.id = 'agnav-forensic-sidebar';
    sidebar.style.cssText = `
        position: fixed;
        top: 0;
        right: 0;
        width: 300px;
        height: 100vh;
        background-color: #1e1e1e;
        border-left: 1px solid #333;
        z-index: 1000;
        padding: 20px;
        box-sizing: border-box;
        color: #fff;
        overflow-y: auto;
        font-family: Inter, system-ui, -apple-system, sans-serif;
    `;

    sidebar.innerHTML = `
        <h2 style="margin-top:0; font-size:1.2rem; border-bottom:1px solid #444; padding-bottom:10px;">Forensic Toolbox</h2>
        
        <div style="margin-top:20px;">
            <h3 style="font-size:0.9rem; color:#888; text-transform:uppercase; letter-spacing:1px;">Reference Documents</h3>
            <ul style="list-style:none; padding:0; margin:10px 0;">
                <li style="margin-bottom:10px;"><a href="/knowledge-base/01_primary/BCGEU%2019th%20Main%20Agreement.pdf" target="_blank" style="color:#4a90e2; text-decoration:none;">19th Main Agreement</a></li>
                <li style="margin-bottom:10px;"><a href="/knowledge-base/01_primary/BC%20Labour%20Relations%20Code.pdf" target="_blank" style="color:#4a90e2; text-decoration:none;">Labour Relations Code</a></li>
                <li style="margin-bottom:10px;"><a href="/knowledge-base/04_jurisprudence/Nexus%20Test%20and%20Off-Duty%20Conduct.pdf" target="_blank" style="color:#4a90e2; text-decoration:none;">Nexus Test (PDF)</a></li>
            </ul>
        </div>

        <div style="margin-top:30px;">
            <h3 style="font-size:0.9rem; color:#888; text-transform:uppercase; letter-spacing:1px;">Steward Tools</h3>
            <button onclick="triggerForensicAction('Export')" style="width:100%; padding:10px; margin-top:10px; background:#333; border:1px solid #444; color:#fff; cursor:pointer; text-align:left;">Export Chat (.md)</button>
            <button onclick="triggerForensicAction('Import')" style="width:100%; padding:10px; margin-top:10px; background:#333; border:1px solid #444; color:#fff; cursor:pointer; text-align:left;">Import Chat (.md)</button>
        </div>

        <div style="margin-top:auto; padding-top:40px; font-size:0.75rem; color:#555;">
            BCGEU Navigator Forensic Suite v13
        </div>
    `;

    document.body.appendChild(sidebar);

    // Adjust the main content to not overlap
    const main = document.querySelector('main') || document.body.firstChild;
    if (main) {
        main.style.marginRight = '300px';
    }
    console.log("[AgNav] Persistent Sidebar injected.");
}

function injectHeaderPersona() {
    if (document.getElementById('agnav-header-persona')) return;

    // Find a stable anchor in the header
    const kbBtn = Array.from(document.querySelectorAll('button, a')).find(el => el.innerText.includes('Knowledge Base'));
    const header = kbBtn ? kbBtn.closest('div').parentNode : (document.querySelector('header') || document.querySelector('.cl-header'));
    
    if (!header) return;

    const container = document.createElement('div');
    container.id = 'agnav-header-persona';
    container.style.cssText = 'display:flex; align-items:center; margin-right:20px; padding:2px 8px; background:#2a2a2a; border:1px solid #444; border-radius:4px; z-index:9999;';

    const label = document.createElement('span');
    label.innerText = 'Perspective:';
    label.style.cssText = 'font-size:0.8rem; margin-right:8px; color:#aaa; font-weight:bold;';

    const select = document.createElement('select');
    select.id = 'persona-selector-header';
    select.style.cssText = 'background:transparent; color:#fff; border:none; font-size:0.85rem; cursor:pointer; outline:none; font-family:inherit;';
    
    ['Lookup', 'Grieve', 'Manage'].forEach(p => {
        const opt = document.createElement('option');
        opt.value = p; opt.innerText = p;
        opt.style.backgroundColor = '#333';
        select.appendChild(opt);
    });

    select.onchange = (e) => {
        console.log(`[AgNav] Header persona change: ${e.target.value}`);
        // Trigger the hidden action buttons in the welcome message
        const buttons = Array.from(document.querySelectorAll('button'));
        const target = buttons.find(b => b.innerText.trim().includes(`${e.target.value} Perspective`) || b.innerText.trim() === e.target.value);
        if (target) target.click();
    };

    container.appendChild(label);
    container.appendChild(select);
    
    if (kbBtn && kbBtn.parentNode) {
        kbBtn.parentNode.insertBefore(container, kbBtn);
    } else {
        header.appendChild(container);
    }
    console.log("[AgNav] Header Persona Selector injected (v13.1).");
}

// Global cleanup of unrequested UI elements
function cleanupUnrequested() {
    // Remove the sticky message if it exists
    const sticky = document.querySelector('.cl-message-list > div:first-child[style*="sticky"]');
    if (sticky) sticky.style.position = 'static';
}

const observer = new MutationObserver(() => {
    injectSidebar();
    injectHeaderPersona();
    cleanupUnrequested();
});

observer.observe(document.body, { childList: true, subtree: true });
setInterval(() => { injectSidebar(); injectHeaderPersona(); }, 2000);

console.log("[AgNav] Forensic Sidebar v13 ready.");

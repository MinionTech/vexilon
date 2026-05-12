// Chainlit Custom JS Bridge (AgNav Forensic Version v11-final-robust)
console.log("[AgNav] Forensic Bridge V11-Final Booting...");

// Signal to subagent/forensics that script is live
document.title = "BCGEU Navigator [Forensic V11]";

async function triggerForensicAction(action) {
    console.log(`[AgNav] Triggering forensic action: ${action}`);
    try {
        await fetch(`/forensic-action/${action}`, { method: 'POST' });
    } catch (err) {
        console.error(`[AgNav] Action failed: ${action}`, err);
    }
}

function injectPersonaSelector() {
    if (document.getElementById('agnav-persona-container')) return;

    // Aggressive header search
    const header = document.querySelector('header') || 
                   document.querySelector('.cl-header') || 
                   document.querySelector('.MuiAppBar-root');
    
    if (!header) {
        console.warn("[AgNav] Header not found, retrying...");
        return;
    }

    const container = document.createElement('div');
    container.id = 'agnav-persona-container';
    container.style.cssText = 'display:flex; align-items:center; margin-left:20px; padding:4px 10px; background:rgba(255,255,255,0.1); border-radius:4px; border:1px solid rgba(255,255,255,0.2);';

    const label = document.createElement('span');
    label.innerText = 'Perspective:';
    label.style.cssText = 'font-size:0.85rem; margin-right:8px; color:#fff; font-weight:bold;';

    const select = document.createElement('select');
    select.id = 'persona-selector';
    select.style.cssText = 'background:transparent; color:#fff; border:none; font-size:0.85rem; cursor:pointer; outline:none; font-family:inherit;';

    ['Lookup', 'Grieve', 'Manage'].forEach(p => {
        const opt = document.createElement('option');
        opt.value = p;
        opt.innerText = p;
        opt.style.backgroundColor = '#333';
        select.appendChild(opt);
    });

    select.onchange = (e) => triggerForensicAction(`persona:${e.target.value}`);

    container.appendChild(label);
    container.appendChild(select);

    // Find the title or the first child to insert next to
    const title = header.querySelector('h1') || header.querySelector('.MuiTypography-root') || header.firstChild;
    if (title && title.parentNode === header) {
        header.insertBefore(container, title.nextSibling);
    } else {
        header.appendChild(container);
    }
    console.log("[AgNav] Persona Selector injected.");
}

// Intercept Header Link clicks
document.addEventListener('click', (e) => {
    const link = e.target.closest('a');
    if (!link) return;

    const text = link.innerText.trim();
    if (['Toolbox', 'Export', 'Import'].includes(text)) {
        // Exclude links inside messages
        if (link.closest('.cl-message') || link.closest('.message')) return;

        e.preventDefault();
        e.stopPropagation();
        triggerForensicAction(text.toLowerCase());
    }
}, true);

// MutationObserver to handle React re-renders
const observer = new MutationObserver((mutations) => {
    injectPersonaSelector();
});

observer.observe(document.body, { childList: true, subtree: true });

// Fallback interval
setInterval(injectPersonaSelector, 2000);

console.log("[AgNav] Bridge V11-Final ready.");

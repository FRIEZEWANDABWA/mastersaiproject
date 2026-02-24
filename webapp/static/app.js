/* ════════════════════════════════════════════════════════════════
   MaizeGuard AI — Frontend Logic v2
   Handles: theme toggle, drag-drop, predict (TTA), results, full report modal
   ════════════════════════════════════════════════════════════════ */

// ── Theme toggle (Dark / Light) ───────────────────────────────────────────────
(function initTheme() {
    const saved = localStorage.getItem('mg-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    updateToggleLabel(saved);
})();

function updateToggleLabel(theme) {
    const btn = document.getElementById('themeToggle');
    if (!btn) return;
    const isDark = theme === 'dark';
    btn.querySelector('.toggle-icon').textContent = isDark ? '🌙' : '☀️';
    btn.querySelector('.toggle-label').textContent = isDark ? 'Dark' : 'Light';
}

document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('themeToggle');
    if (!btn) return;
    btn.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('mg-theme', next);
        updateToggleLabel(next);
        showToast(next === 'light' ? '☀️ Light mode on' : '🌙 Dark mode on', 'info');
    });
});

// ── DOM refs ──────────────────────────────────────────────────────────────────
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadIdle = document.getElementById('uploadIdle');
const uploadPreview = document.getElementById('uploadPreview');
const previewImg = document.getElementById('previewImg');
const clearBtn = document.getElementById('clearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnSpinner = document.getElementById('btnSpinner');
const emptyState = document.getElementById('emptyState');
const loadingState = document.getElementById('loadingState');
const resultContent = document.getElementById('resultContent');
const reportBtn = document.getElementById('reportBtn');
const modalOverlay = document.getElementById('modalOverlay');
const modalLoading = document.getElementById('modalLoading');
const modalContent = document.getElementById('modalContent');

let selectedFile = null;

// ── Drag & Drop ───────────────────────────────────────────────────────────────
uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});
['dragleave', 'dragend'].forEach(ev =>
    uploadZone.addEventListener(ev, () => uploadZone.classList.remove('drag-over'))
);
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
});
uploadZone.addEventListener('click', e => {
    if (e.target === clearBtn || clearBtn.contains(e.target)) return;
    fileInput.click();
});
fileInput.addEventListener('change', e => {
    if (e.target.files[0]) setFile(e.target.files[0]);
});

// ── File selection ────────────────────────────────────────────────────────────
function setFile(file) {
    if (!file.type.match(/image\/(jpeg|jpg|png|webp|bmp)/i)) {
        showToast('⚠️ Please upload a JPG, PNG or WEBP image', 'warn');
        return;
    }
    selectedFile = file;
    previewImg.src = URL.createObjectURL(file);
    uploadIdle.style.display = 'none';
    uploadPreview.style.display = 'block';
    analyzeBtn.disabled = false;
    showState('empty');
}

clearBtn.addEventListener('click', e => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    uploadPreview.style.display = 'none';
    uploadIdle.style.display = 'block';
    analyzeBtn.disabled = true;
    reportBtn.style.display = 'none';
    showState('empty');
});

// ── Prediction (with TTA) ─────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    showState('loading');
    analyzeBtn.disabled = true;
    document.querySelector('.btn-text').style.display = 'none';
    btnSpinner.style.display = 'inline-flex';

    try {
        const fd = new FormData();
        fd.append('file', selectedFile);
        const res = await fetch('/predict', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Prediction failed');
        renderResult(data);
    } catch (err) {
        showState('empty');
        showToast('❌ ' + err.message, 'error');
    } finally {
        analyzeBtn.disabled = false;
        document.querySelector('.btn-text').style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
});

// ── Render quick result ───────────────────────────────────────────────────────
function renderResult(data) {
    // Header
    document.getElementById('resultIcon').textContent = data.icon;
    document.getElementById('resultLabel').textContent = data.label;
    document.getElementById('resultLabel').style.color = data.color;
    document.getElementById('resultConf').textContent = `Confidence: ${data.confidence}%`;

    // Severity badge
    const sb = document.getElementById('severityBadge');
    const severityColors = { None: '#22c55e', 'Moderate–Severe': '#f59e0b', Critical: '#ef4444' };
    sb.textContent = `Severity: ${data.severity}`;
    sb.style.background = hexToRgba(severityColors[data.severity] || '#888', 0.15);
    sb.style.color = severityColors[data.severity] || '#888';
    sb.style.border = `1px solid ${hexToRgba(severityColors[data.severity] || '#888', 0.3)}`;

    // Header tint
    const hdr = document.getElementById('resultHeader');
    hdr.style.background = hexToRgba(data.color, 0.07);
    hdr.style.borderColor = hexToRgba(data.color, 0.2);

    // Advice
    document.getElementById('adviceText').textContent = data.short_advice;

    // Uncertainty
    document.getElementById('uncertaintyBox').style.display = data.is_uncertain ? 'block' : 'none';

    // Probability bars
    const probBars = document.getElementById('probBars');
    probBars.innerHTML = '';
    const colorMap = { maize_healthy: '#22c55e', maize_mln: '#ef4444', maize_streak: '#f59e0b' };
    const labelMap = { maize_healthy: 'Healthy', maize_mln: 'Lethal Necrosis (MLN)', maize_streak: 'Maize Streak Virus' };

    Object.entries(data.all_probs)
        .sort((a, b) => b[1] - a[1])
        .forEach(([cls, pct]) => {
            const row = document.createElement('div');
            row.className = 'prob-row';
            row.innerHTML = `
        <div class="prob-label">
          <span>${labelMap[cls] || cls}</span>
          <span style="color:${colorMap[cls]}">${pct}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" data-w="${pct}" style="background:${colorMap[cls]}; width:0%"></div>
        </div>`;
            probBars.appendChild(row);
        });

    showState('result');

    // Animate bars
    requestAnimationFrame(() => {
        document.querySelectorAll('.prob-fill').forEach(el => {
            el.style.width = el.dataset.w + '%';
        });
    });

    // Grad-CAM
    if (data.gradcam) {
        document.getElementById('gradcamImg').src = 'data:image/png;base64,' + data.gradcam;
        document.getElementById('gradcamSection').style.display = 'block';
    } else {
        document.getElementById('gradcamSection').style.display = 'none';
    }

    // Demo notice
    document.getElementById('demoNotice').style.display = data.demo_mode ? 'block' : 'none';

    // Show report button
    reportBtn.style.display = 'flex';
}

// ── State helper ──────────────────────────────────────────────────────────────
function showState(state) {
    emptyState.style.display = state === 'empty' ? 'flex' : 'none';
    loadingState.style.display = state === 'loading' ? 'flex' : 'none';
    resultContent.style.display = state === 'result' ? 'block' : 'none';
}

// ══════════════════════════════════════════════════════════════════
//  FULL REPORT MODAL
// ══════════════════════════════════════════════════════════════════

reportBtn.addEventListener('click', openReport);
document.getElementById('modalClose').addEventListener('click', closeModal);
document.getElementById('closeReportBtn').addEventListener('click', closeModal);
modalOverlay.addEventListener('click', e => { if (e.target === modalOverlay) closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

function openModal() {
    modalOverlay.style.display = 'flex';
    modalLoading.style.display = 'flex';
    modalContent.style.display = 'none';
    document.body.style.overflow = 'hidden';
}
function closeModal() {
    modalOverlay.style.display = 'none';
    document.body.style.overflow = '';
}

async function openReport() {
    if (!selectedFile) return;
    openModal();

    try {
        const fd = new FormData();
        fd.append('file', selectedFile);
        const res = await fetch('/detailed-report', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Report failed');
        renderReport(data);
    } catch (err) {
        closeModal();
        showToast('❌ Report error: ' + err.message, 'error');
    }
}

function renderReport(d) {
    // Header
    document.getElementById('reportMeta').innerHTML =
        `Report ID: <strong>${d.report_id}</strong> &nbsp;·&nbsp; Generated: ${d.timestamp}`;

    // Diagnosis block
    const diagBlock = document.getElementById('diagnosisBlock');
    diagBlock.style.borderLeft = `4px solid ${d.color}`;
    document.getElementById('rLabel').textContent = `${d.icon}  ${d.label}`;
    document.getElementById('rLabel').style.color = d.color;
    document.getElementById('rConf').textContent = d.confidence + '%';
    document.getElementById('rSeverity').textContent = d.severity;
    const sevEl = document.getElementById('rSeverity');
    const sevColors = { None: '#22c55e', 'Moderate–Severe': '#f59e0b', Critical: '#ef4444' };
    sevEl.style.color = sevColors[d.severity] || '#888';
    document.getElementById('rId').textContent = d.report_id;
    document.getElementById('rTime').textContent = d.timestamp;

    // Uncertainty
    document.getElementById('rUncertainty').style.display = d.is_uncertain ? 'block' : 'none';

    // Text sections
    setText('rOverview', d.overview);
    setText('rWhatIs', d.what_it_is);
    setText('rSpread', d.how_it_spreads);
    setText('rEcon', d.economic_impact);

    // Lists
    setList('rSymptoms', d.visual_symptoms_detected, '👁️');
    setList('rImmediate', d.immediate_actions, '→');
    setList('rShortTerm', d.short_term_actions, '→');
    setList('rLongTerm', d.long_term_prevention, '→');
    setList('rTreatment', d.treatment_options, '💊');
    setOrderedList('rRefs', d.references);

    // Differential diagnosis
    const diffEl = document.getElementById('rDifferential');
    diffEl.innerHTML = '';
    d.differential.forEach(item => {
        const bar = document.createElement('div');
        bar.className = 'diff-row';
        const colors = { maize_healthy: '#22c55e', maize_mln: '#ef4444', maize_streak: '#f59e0b' };
        bar.innerHTML = `
      <div class="diff-label">
        <span>${item.label}</span>
        <span style="color:${colors[item.class]}">${item.prob}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" data-w="${item.prob}"
             style="background:${colors[item.class]}; width:0%"></div>
      </div>`;
        diffEl.appendChild(bar);
    });
    requestAnimationFrame(() => {
        diffEl.querySelectorAll('.prob-fill').forEach(el => {
            el.style.width = el.dataset.w + '%';
        });
    });

    // Grad-CAM
    if (d.gradcam) {
        document.getElementById('rGradcam').src = 'data:image/png;base64,' + d.gradcam;
        document.getElementById('rGradcamSection').style.display = 'block';
    } else {
        document.getElementById('rGradcamSection').style.display = 'none';
    }

    // Demo notice in report
    if (d.demo_mode) {
        const notice = document.createElement('div');
        notice.className = 'demo-notice';
        notice.innerHTML = '⚠️ <strong>Demo Mode:</strong> This is an illustrative report. Real predictions require a trained model (50+ images per class).';
        document.getElementById('modalContent').prepend(notice);
    }

    // Show
    modalLoading.style.display = 'none';
    modalContent.style.display = 'block';
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setText(id, text) { document.getElementById(id).textContent = text; }

function setList(id, items, bullet = '→') {
    const el = document.getElementById(id);
    el.innerHTML = items.map(i => `<li>${i}</li>`).join('');
}

function setOrderedList(id, items) {
    const el = document.getElementById(id);
    el.innerHTML = items.map(i => `<li>${i}</li>`).join('');
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

function showToast(msg, type = 'info') {
    const colors = { info: '#22c55e', warn: '#f59e0b', error: '#ef4444' };
    const t = document.createElement('div');
    t.style.cssText = `
    position:fixed;bottom:24px;right:24px;z-index:99999;
    background:#0d1a1f;border:1px solid ${colors[type]};
    color:#e2f5ec;padding:14px 20px;border-radius:12px;
    font-family:Inter,sans-serif;font-size:14px;
    box-shadow:0 8px 32px rgba(0,0,0,.5);max-width:340px;
    animation:fadeUp .3s ease;
  `;
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 5000);
}

// Inject animation
const s = document.createElement('style');
s.textContent = `
  @keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
`;
document.head.appendChild(s);

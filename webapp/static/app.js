/* ────────────────────────────────────────────────────────────────
   MaizeGuard AI — Frontend Logic
   Handles: drag-drop upload, preview, API call, results rendering
   ──────────────────────────────────────────────────────────────── */

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

let selectedFile = null;

/* ── Drag & Drop ─────────────────────────────────────────────────── */
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

/* ── Set selected file ───────────────────────────────────────────── */
function setFile(file) {
    if (!file.type.match(/image\/(jpeg|jpg|png|webp|bmp)/i)) {
        showToast('⚠️ Please upload a JPG, PNG or WEBP image', 'warn');
        return;
    }
    selectedFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    uploadIdle.style.display = 'none';
    uploadPreview.style.display = 'block';
    analyzeBtn.disabled = false;

    // Reset results
    showState('empty');
}

/* ── Clear ───────────────────────────────────────────────────────── */
clearBtn.addEventListener('click', e => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    uploadPreview.style.display = 'none';
    uploadIdle.style.display = 'block';
    analyzeBtn.disabled = true;
    showState('empty');
});

/* ── Analyse ─────────────────────────────────────────────────────── */
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    showState('loading');
    analyzeBtn.disabled = true;
    document.querySelector('.btn-text').style.display = 'none';
    btnSpinner.style.display = 'inline-flex';

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const res = await fetch('/predict', { method: 'POST', body: formData });
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

/* ── Show/hide states ────────────────────────────────────────────── */
function showState(state) {
    emptyState.style.display = state === 'empty' ? 'flex' : 'none';
    loadingState.style.display = state === 'loading' ? 'flex' : 'none';
    resultContent.style.display = state === 'result' ? 'block' : 'none';
}

/* ── Render result ───────────────────────────────────────────────── */
function renderResult(data) {
    // Header
    document.getElementById('resultIcon').textContent = data.icon;
    document.getElementById('resultLabel').textContent = data.label;
    document.getElementById('resultLabel').style.color = data.color;
    document.getElementById('resultConf').textContent = `Confidence: ${data.confidence}%`;

    // Header background tint
    const hdr = document.getElementById('resultHeader');
    hdr.style.background = hexToRgba(data.color, .07);
    hdr.style.borderColor = hexToRgba(data.color, .2);

    // Advice
    document.getElementById('adviceText').textContent = data.advice;

    // Probability bars
    const probBars = document.getElementById('probBars');
    probBars.innerHTML = '';
    const colorMap = {
        maize_healthy: '#22c55e',
        maize_mln: '#ef4444',
        maize_streak: '#f59e0b'
    };
    const labelMap = {
        maize_healthy: 'Healthy',
        maize_mln: 'Lethal Necrosis (MLN)',
        maize_streak: 'Maize Streak Virus'
    };

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
          <div class="prob-fill" data-w="${pct}"
               style="background:${colorMap[cls]}; width:0%"></div>
        </div>`;
            probBars.appendChild(row);
        });

    showState('result');

    // Animate bars after paint
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
}

/* ── Toast notification ──────────────────────────────────────────── */
function showToast(msg, type = 'info') {
    const t = document.createElement('div');
    const colors = { info: '#22c55e', warn: '#f59e0b', error: '#ef4444' };
    t.style.cssText = `
    position:fixed; bottom:24px; right:24px; z-index:9999;
    background:#0d1a1f; border:1px solid ${colors[type]};
    color:#e2f5ec; padding:14px 20px; border-radius:12px;
    font-family:Inter,sans-serif; font-size:14px;
    box-shadow:0 8px 32px rgba(0,0,0,.5);
    animation:fade-up .3s ease; max-width:320px;
  `;
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 4000);
}

/* ── Utilities ───────────────────────────────────────────────────── */
function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

/* ── Inject fade-up keyframes if not present ─────────────────────── */
if (!document.getElementById('toast-kf')) {
    const s = document.createElement('style');
    s.id = 'toast-kf';
    s.textContent = `@keyframes fade-up{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}`;
    document.head.appendChild(s);
}

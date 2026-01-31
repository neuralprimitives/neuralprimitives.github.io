document.addEventListener('DOMContentLoaded', () => {
  const buttons = document.querySelectorAll(
    'button.paper-btn.paper-btn-ghost[data-bib-id]'
  );
  console.log('[bibtex] buttons found:', buttons.length);

  buttons.forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.preventDefault();
      e.stopPropagation();

      const card = btn.closest('.paper-card');
      const panel = card?.querySelector('.bibtex-panel');
      if (!panel) {
        console.warn('[bibtex] bibtex-panel not found in card');
        return;
      }

      const bibId = btn.dataset.bibId;              // ✅ 正确
      const bibEl = bibId ? document.getElementById(bibId) : null;
      const bib = bibEl ? bibEl.textContent.trim() : '';
      console.log('[bibtex] clicked, len=', bib.length, 'id=', bibId);

      // ✅ toggle 展开/收回
      const isOpen = panel.classList.toggle('is-open');

      // ✅ 按钮变色 + 文案
      btn.classList.toggle('is-active', isOpen);
      btn.textContent = isOpen ? 'Hide Bib' : 'BibTeX';

      // ✅ 展开时填充 + 复制 + toast
      if (isOpen) {
        panel.textContent = bib;

        try {
          await navigator.clipboard.writeText(bib);
          showToast('BibTeX copied ✓');
        } catch (err) {
          fallbackCopy(bib);
          showToast('BibTeX copied ✓');
        }
      }
    });
  });

  function fallbackCopy(text){
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
  }

  function showToast(msg){
    const t = document.getElementById('toast');
    if (!t) {
      console.warn('[bibtex] #toast not found. Add <div id="toast" class="bibtex-toast"></div>');
      return;
    }
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(window.__toastTimer);
    window.__toastTimer = setTimeout(() => t.classList.remove('show'), 1400);
  }
});

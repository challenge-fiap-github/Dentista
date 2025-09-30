// ============================
// Configuração da API
// ============================
const API_BASE = 'http://127.0.0.1:5001/api'; // ajuste se publicar online

async function apiGet(path, params = {}) {
  const url = new URL(API_BASE + path);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') {
      url.searchParams.append(k, v);
    }
  });
  const res = await fetch(url, { headers: authHeaders() });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiPatch(path, body) {
  const res = await fetch(API_BASE + path, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function authHeaders() {
  const token = localStorage.getItem('ov-token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// ============================
// Renderização da Tabela
// ============================
function rowHtml(item) {
  return `
    <tr>
      <td>${item.id}</td>
      <td>${item.paciente?.nome ?? '—'}</td>
      <td>${item.paciente?.cpf ?? '—'}</td>
      <td>${item.procedimento ?? '—'}</td>
      <td>${item.dentista?.nome ?? '—'}</td>
      <td>${item.status ?? '—'}</td>
      <td>${new Date(item.createdAt).toLocaleString('pt-BR')}</td>
      <td>
        <button class="btn ghost" onclick="verDetalhe(${item.id})">Ver</button>
      </td>
    </tr>
  `;
}

async function carregarTabela(page = 0) {
  const search = document.getElementById('search')?.value || '';
  const status = document.getElementById('status')?.value || '';
  const startDate = document.getElementById('startDate')?.value || '';
  const endDate = document.getElementById('endDate')?.value || '';

  try {
    const data = await apiGet('/diagnosticos', {
      search,
      status,
      page,
      size: 10,
      startDate,
      endDate
    });
    const items = data.content || [];
    const tbody = document.querySelector('#tabela tbody');
    tbody.innerHTML = items.map(rowHtml).join('');

    renderPaginacao(data.totalPages, data.number);
  } catch (e) {
    console.error(e);
    showToast('Erro ao carregar diagnósticos.');
  }
}

function renderPaginacao(totalPages, currentPage) {
  const el = document.getElementById('paginacao');
  if (!el) return;

  // helper de botão
  const makeBtn = (label, page, disabled = false, extraClass = 'ghost') =>
    `<button class="btn ${extraClass}" ${disabled ? 'disabled' : ''} data-page="${page}">${label}</button>`;

  let html = '';

  // setas (primeira / anterior)
  html += makeBtn('«', 0, currentPage === 0);
  html += makeBtn('‹', Math.max(0, currentPage - 1), currentPage === 0);

  // janela ao redor da página atual
  const windowSize = 2; // mostra atual ±2
  const start = Math.max(0, currentPage - windowSize);
  const end = Math.min(totalPages - 1, currentPage + windowSize);

  const pageBtn = (i) =>
    makeBtn(String(i + 1), i, i === currentPage, i === currentPage ? 'primary' : 'ghost');

  // primeira + reticências
  if (start > 0) {
    html += pageBtn(0);
    if (start > 1) html += `<span class="ellipsis">…</span>`;
  }

  // janela do meio
  for (let i = start; i <= end; i++) html += pageBtn(i);

  // última + reticências
  if (end < totalPages - 1) {
    if (end < totalPages - 2) html += `<span class="ellipsis">…</span>`;
    html += pageBtn(totalPages - 1);
  }

  // setas (próxima / última)
  html += makeBtn('›', Math.min(totalPages - 1, currentPage + 1), currentPage === totalPages - 1);
  html += makeBtn('»', totalPages - 1, currentPage === totalPages - 1);

  el.innerHTML = html;

  // navegação
  el.querySelectorAll('button[data-page]').forEach((btn) => {
    if (btn.disabled) return;
    btn.addEventListener('click', () => {
      const p = parseInt(btn.getAttribute('data-page'), 10);
      carregarTabela(p);
      el.scrollIntoView({ behavior: 'smooth', block: 'end', inline: 'center' });
    });
  });
}

// ============================
// Detalhes + Atualização de status
// ============================
async function verDetalhe(id) {
  try {
    const d = await apiGet(`/diagnosticos/${id}`);
    const modal = document.getElementById('detalheModal');
    modal.querySelector('.modal-body').innerHTML = renderDetail(d);
    modal.showModal();

    // Botões de status
    modal.querySelector('#btnEmAnalise').onclick = () => alterarStatus(id, 'em_analise');
    modal.querySelector('#btnAprovar').onclick = () => alterarStatus(id, 'aprovado');
    modal.querySelector('#btnReprovar').onclick = () => alterarStatus(id, 'reprovado');
  } catch (e) {
    console.error(e);
    showToast('Erro ao carregar detalhe.');
  }
}

function renderDetail(d) {
  return `
    <h3>Paciente</h3>
    <p><b>Nome:</b> ${d.paciente?.nome ?? '—'}<br>
       <b>CPF:</b> ${d.paciente?.cpf ?? '—'}</p>

    <h3>Consulta</h3>
    <p><b>Procedimento:</b> ${d.procedimento}<br>
       <b>Executado:</b> ${d.executado}<br>
       <b>Prescrição:</b> ${d.prescricao || '—'}</p>

    <h3>Dentista</h3>
    <p>${d.dentista?.nome ?? '—'}</p>

    <h3>Status Atual:</h3>
    <p>${d.status}</p>

    <div class="actions">
      <button id="btnEmAnalise" class="btn ghost">Marcar em análise</button>
      <button id="btnAprovar" class="btn primary">Aprovar</button>
      <button id="btnReprovar" class="btn danger">Reprovar</button>
    </div>
  `;
}

async function alterarStatus(id, status) {
  try {
    await apiPatch(`/diagnosticos/${id}/status`, { status });
    showToast(`Status alterado para ${status}`);
    document.getElementById('detalheModal').close();
    carregarTabela();
  } catch (e) {
    console.error(e);
    showToast('Erro ao atualizar status.');
  }
}

// ============================
// Toast
// ============================
const toast = document.getElementById('toast');
function showToast(msg, ms = 2400) {
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), ms);
}

// ============================
// Inicialização
// ============================
document.addEventListener('DOMContentLoaded', () => {
  carregarTabela();
  document.getElementById('searchBtn')?.addEventListener('click', () => carregarTabela());
});
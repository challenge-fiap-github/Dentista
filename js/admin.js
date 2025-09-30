// ====== Guarda de rota (login + papel admin opcional) ======
(function guard(){
  const logged = localStorage.getItem('ov-auth');
  const role = localStorage.getItem('ov-auth-role');
  if(!logged){ window.location.href = 'login.html'; return; }
  // Se quiser obrigar papel admin, descomente:
  // if(role !== 'admin'){ window.location.href = 'index.html'; }
})();

// ====== Tema persistente ======
(function initTheme(){
  const saved = localStorage.getItem('ov-theme');
  if (saved === 'dark') document.documentElement.classList.add('dark');
})();
document.getElementById('btnDarkMode').addEventListener('click', () => {
  document.documentElement.classList.toggle('dark');
  localStorage.setItem('ov-theme', document.documentElement.classList.contains('dark') ? 'dark' : 'light');
});

// Header
document.getElementById('year').textContent = new Date().getFullYear();
document.getElementById('btnLogout').addEventListener('click', () => {
  localStorage.removeItem('ov-auth');
  localStorage.removeItem('ov-auth-role');
  window.location.href = 'login.html';
});

// ====== Config da API (ajuste a URL da sua API) ======
const API_BASE = 'http://localhost:8080/api'; // <== TROQUE PELA SUA BASE
const PAGE_SIZE = 10;

const state = {
  q: '',
  status: '',
  start: '',
  end: '',
  page: 0,
  totalPages: 0
};

// Toast
const toast = document.getElementById('toast');
function showToast(msg, ms=2400){
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(()=>toast.classList.remove('show'), ms);
}

// Helpers API
function authHeaders() {
  // Se você tiver token JWT/Session, recupere de localStorage
  const token = localStorage.getItem('ov-token'); // opcional
  return token ? { 'Authorization': `Bearer ${token}` } : {};
}
async function apiGet(path, params = {}) {
  const qs = new URLSearchParams(params).toString();
  const url = `${API_BASE}${path}${qs ? `?${qs}` : ''}`;
  const res = await fetch(url, { headers: { 'Accept':'application/json', ...authHeaders() }});
  if(!res.ok) throw new Error(`GET ${path} ${res.status}`);
  return res.json();
}
async function apiPatch(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'PATCH',
    headers: { 'Content-Type':'application/json', ...authHeaders() },
    body: JSON.stringify(body)
  });
  if(!res.ok) throw new Error(`PATCH ${path} ${res.status}`);
  return res.json().catch(()=> ({}));
}

// ====== UI refs ======
const qEl = document.getElementById('q');
const statusEl = document.getElementById('status');
const startEl = document.getElementById('start');
const endEl = document.getElementById('end');
const btnFiltrar = document.getElementById('btnFiltrar');
const btnLimparFiltros = document.getElementById('btnLimparFiltros');
const tbody = document.getElementById('tbody');
const pageInfo = document.getElementById('pageInfo');
const prevBtn = document.getElementById('prev');
const nextBtn = document.getElementById('next');

// Detalhe modal
const detailModal = document.getElementById('detailModal');
const closeDetail = document.getElementById('closeDetail');
const detailBody = document.getElementById('detailBody');
const btnExportJSON = document.getElementById('btnExportJSON');
const btnMarkAnalyzed = document.getElementById('btnMarkAnalyzed');

closeDetail.addEventListener('click', () => detailModal.close());

// ====== Eventos filtros/paginação ======
btnFiltrar.addEventListener('click', () => {
  state.q = qEl.value.trim();
  state.status = statusEl.value;
  state.start = startEl.value;
  state.end = endEl.value;
  state.page = 0;
  loadTable();
});
btnLimparFiltros.addEventListener('click', () => {
  qEl.value = ''; statusEl.value = ''; startEl.value = ''; endEl.value = '';
  state.q = state.status = state.start = state.end = '';
  state.page = 0;
  loadTable();
});
prevBtn.addEventListener('click', () => {
  if(state.page > 0){ state.page--; loadTable(); }
});
nextBtn.addEventListener('click', () => {
  if(state.page + 1 < state.totalPages){ state.page++; loadTable(); }
});

// ====== Carregar Tabela ======
async function loadTable(){
  tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;">Carregando…</td></tr>`;
  try{
    const params = {
      search: state.q,
      status: state.status,
      page: state.page,
      size: PAGE_SIZE,
      startDate: state.start || undefined,
      endDate: state.end || undefined
    };
    const data = await apiGet('/diagnosticos', params);

    // Tentar adaptar formatos comuns de paginação:
    // Spring Page: { content, totalPages, number }
    // Lista simples: []
    const items = Array.isArray(data) ? data
      : Array.isArray(data.content) ? data.content
      : Array.isArray(data.items) ? data.items
      : (data.results || []);

    state.totalPages = Number.isInteger(data.totalPages) ? data.totalPages
                   : ((data.total_pages ?? Math.ceil((data.total ?? items.length)/PAGE_SIZE)) || 1);

    if(!items || items.length === 0){
      tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;">Nenhum registro encontrado.</td></tr>`;
    }else{
      tbody.innerHTML = items.map(rowHtml).join('');
      // bind botões Ver
      document.querySelectorAll('[data-view]').forEach(btn=>{
        btn.addEventListener('click', () => openDetail(btn.getAttribute('data-view')));
      });
    }
    pageInfo.textContent = `Página ${state.page+1} de ${state.totalPages}`;
  }catch(e){
    console.error(e);
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:#ef4444">Erro ao carregar dados.</td></tr>`;
    showToast('Falha ao carregar diagnósticos.');
  }
}

function rowHtml(item){
  // Mapeamento defensivo de campos
  const id = item.id ?? item.codigo ?? '—';
  const paciente = item.paciente?.nome ?? item.nomePaciente ?? '—';
  const cpf = item.paciente?.cpf ?? item.cpf ?? '—';
  const procedimento = item.procedimento ?? item.nomeProcedimento ?? '—';
  const data = fmtDate(item.data ?? item.createdAt ?? item.criadoEm);
  const status = item.status ?? 'novo';
  const dentista = item.dentista?.nome ?? item.usuario ?? '—';

  return `
    <tr>
      <td>${id}</td>
      <td>${escapeHtml(paciente)}</td>
      <td>${escapeHtml(cpf)}</td>
      <td>${escapeHtml(procedimento)}</td>
      <td>${data}</td>
      <td><span class="badge ${badgeClass(status)}">${escapeHtml(labelStatus(status))}</span></td>
      <td>${escapeHtml(dentista)}</td>
      <td style="text-align:right;">
        <button class="btn ghost" data-view="${id}">Ver</button>
      </td>
    </tr>`;
}

// ====== Detalhe ======
let currentDetailId = null;

async function openDetail(id){
  currentDetailId = id;
  detailBody.innerHTML = 'Carregando…';
  detailModal.showModal();
  try{
    const data = await apiGet(`/diagnosticos/${id}`);
    detailBody.innerHTML = renderDetail(data);
  }catch(e){
    detailBody.innerHTML = `<p style="color:#ef4444">Erro ao carregar detalhe.</p>`;
  }
}

btnExportJSON.addEventListener('click', () => {
  if(!currentDetailId) return;
  apiGet(`/diagnosticos/${currentDetailId}`).then(data=>{
    const blob = new Blob([JSON.stringify(data, null, 2)], {type:'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `diagnostico_${currentDetailId}.json`;
    a.click();
  });
});

btnMarkAnalyzed.addEventListener('click', async () => {
  if(!currentDetailId) return;
  try{
    await apiPatch(`/diagnosticos/${currentDetailId}/status`, { status: 'em_analise' });
    showToast('Marcado como "Em análise".');
    detailModal.close();
    loadTable();
  }catch(e){
    showToast('Falha ao atualizar status.');
  }
});

// ====== Utils ======
function fmtDate(iso){
  if(!iso) return '—';
  const d = new Date(iso);
  if(isNaN(d)) return '—';
  return d.toLocaleString('pt-BR', { hour12:false });
}
function badgeClass(status){
  switch((status||'').toLowerCase()){
    case 'aprovado': return 'badge-success';
    case 'reprovado': return 'badge-danger';
    case 'em_analise': return 'badge-warning';
    default: return 'badge-neutral';
  }
}
function labelStatus(s){
  const v = (s||'').toLowerCase();
  return v === 'em_analise' ? 'Em análise'
       : v === 'aprovado' ? 'Aprovado'
       : v === 'reprovado' ? 'Reprovado'
       : 'Novo';
}
function escapeHtml(str){ return (str??'').toString().replace(/[&<>"']/g, m=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;' }[m])); }
function renderDetail(d){
  const paciente = d.paciente?.nome ?? d.nomePaciente ?? '—';
  const cpf = d.paciente?.cpf ?? d.cpf ?? '—';
  const procedimento = d.procedimento ?? d.nomeProcedimento ?? '—';
  const executado = d.executado ?? d.descricao ?? '—';
  const prescricao = d.prescricao ?? '—';
  const status = labelStatus(d.status);
  const created = fmtDate(d.data ?? d.createdAt);
  const dentista = d.dentista?.nome ?? d.usuario ?? '—';

  return `
    <p><strong>Paciente:</strong> ${escapeHtml(paciente)}</p>
    <p><strong>CPF:</strong> ${escapeHtml(cpf)}</p>
    <p><strong>Procedimento:</strong> ${escapeHtml(procedimento)}</p>
    <p><strong>Status:</strong> ${escapeHtml(status)}</p>
    <p><strong>Dentista:</strong> ${escapeHtml(dentista)}</p>
    <p><strong>Data:</strong> ${escapeHtml(created)}</p>
    <hr>
    <p><strong>Executado:</strong><br>${escapeHtml(executado).replace(/\n/g,'<br>')}</p>
    <p><strong>Prescrição:</strong><br>${escapeHtml(prescricao).replace(/\n/g,'<br>')}</p>
  `;
}

// Inicialização
loadTable();

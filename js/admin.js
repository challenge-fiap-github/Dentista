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
// Helpers de UI
// ============================
const toast = document.getElementById('toast');
function showToast(msg, ms = 2400) {
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), ms);
}

function badgeClassFor(status, suspicious) {
  const s = (status || (suspicious ? 'atencao' : 'normal')).toLowerCase();
  if (s === 'atencao') return 'badge badge-atencao';
  if (s === 'normal') return 'badge badge-normal';
  if (s === 'em_analise') return 'badge badge-em-analise';
  if (s === 'aprovado') return 'badge badge-aprovado';
  if (s === 'reprovado') return 'badge badge-reprovado';
  return 'badge badge-neutral';
}

function formatDate(iso) {
  try { return new Date(iso).toLocaleString('pt-BR'); } catch { return iso; }
}

function yearFromISO(iso) {
  const d = new Date(iso);
  return isNaN(d) ? null : d.getFullYear();
}

function monthIndex(iso) {
  const d = new Date(iso);
  return isNaN(d) ? null : d.getMonth(); // 0..11
}

const MONTHS_PT = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'];

// ===== Helpers de status/motivo =====
const norm = s => (s ?? '').toString().normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase();
const isAtencao = s => norm(s) === 'atencao';

// também trate cenários onde o back manda uma flag (fraude: 1)
function isAttentionLike(obj) {
  return isAtencao(obj?.status) || obj?.fraude === 1 || obj?.fraude === '1' || norm(obj?.status) === 'suspeito';
}

function badgeHtml(status) {
  const s = norm(status);
  if (s === 'aprovado')   return `<span class="badge badge-success" title="Aprovado">Aprovado</span>`;
  if (s === 'reprovado')  return `<span class="badge badge-danger" title="Reprovado">Reprovado</span>`;
  if (s === 'em_analise') return `<span class="badge badge-warning" title="Em análise">Em análise</span>`;
  if (s === 'atencao')    return `<span class="badge badge-warning" title="Caso em atenção">Atenção</span>`;
  if (s === 'novo')       return `<span class="badge badge-neutral" title="Novo">Novo</span>`;
  const label = (status ?? '—').toString().replace('_', ' ');
  return `<span class="badge badge-neutral">${label}</span>`;
}

// >>> AJUSTE CRÍTICO: ler motivoAtencao (camelCase) vindo da API.
// fallback exibindo "Atenção" quando o caso é atenção e não há motivo.
function getMotivo(obj) {
  const motivo =
    obj?.motivoAtencao ||     // API atual (normalize_row)
    obj?.motivo_atencao ||    // CSV lido direto (casos antigos)
    obj?.motivos ||           // coluna auxiliar que você gera no CSV
    obj?.attentionReason ||   // legado
    obj?.motivo ||            // qualquer outro
    '';

  if (isAttentionLike(obj)) {
    return (motivo && String(motivo).trim()) ? motivo : 'Atenção';
  }
  return '—';
}

// ============================
// Renderização da Tabela
// ============================
function rowHtml(item) {
  return `
    <tr>
      <td>${item.id}</td>
      <td>${item.paciente?.nome ?? item.paciente_nome ?? '—'}</td>
      <td>${item.paciente?.cpf ?? item.cpf ?? '—'}</td>
      <td>${item.procedimento ?? '—'}</td>
      <td>${item.dentista?.nome ?? item.dentista ?? '—'}</td>
      <td>${badgeHtml(item.status)}</td>
      <td>${getMotivo(item)}</td>
      <td>${formatDate(item.createdAt ?? item.created_at)}</td>
      <td>
        <button class="btn ghost" onclick="verDetalhe(${item.id})">Ver</button>
      </td>
    </tr>
  `;
}

async function carregarTabela(page = 0) {
  const search = document.getElementById('search')?.value || '';
  const statusSel = document.getElementById('status')?.value || '';
  const startDate = document.getElementById('startDate')?.value || '';
  const endDate = document.getElementById('endDate')?.value || '';
  const fraudeFilter = document.getElementById('fraudeFilter')?.value || 'all';
  const yearSel = document.getElementById('yearSelect')?.value || '';

  try {
    const data = await apiGet('/diagnosticos', {
      search,
      status: statusSel,
      page,
      size: 10,
      startDate, // ignorado pelo back (ok)
      endDate    // ignorado pelo back (ok)
    });

    let items = data.content || [];

    // aplica filtro por ano (local)
    if (yearSel) {
      items = items.filter(x => String(yearFromISO(x.createdAt)) === String(yearSel));
    }

    // filtro local de possíveis fraudes
    if (fraudeFilter === 'suspicious') {
      items = items.filter(x => x.suspicious === true || isAtencao(x.status));
    } else if (fraudeFilter === 'normal') {
      items = items.filter(x => (x.suspicious === false) && ((x.status || '').toLowerCase() === 'normal'));
    }

    // render tabela
    const tbody = document.querySelector('#tabela tbody');
    if (!tbody) return;
    if (!items.length) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;">Nenhum registro</td></tr>`;
    } else {
      tbody.innerHTML = items.map(rowHtml).join('');
    }

    // paginação baseada no total do servidor
    renderPaginacao(data.totalPages, data.number);

    // popular ano com base no conjunto exibido
    popularAnosSelect(items);

    // atualiza gráfico do topo (se você estiver usando esse gráfico)
    atualizarGrafico(items);
  } catch (e) {
    console.error(e);
    showToast('Erro ao carregar diagnósticos.');
  }
}

function renderPaginacao(totalPages, currentPage) {
  const el = document.getElementById('paginacao');
  if (!el) return;

  const makeBtn = (label, page, disabled = false, extraClass = 'ghost') =>
    `<button class="btn ${extraClass}" ${disabled ? 'disabled' : ''} data-page="${page}">${label}</button>`;

  let html = '';
  html += makeBtn('«', 0, currentPage === 0);
  html += makeBtn('‹', Math.max(0, currentPage - 1), currentPage === 0);

  const windowSize = 2;
  const start = Math.max(0, currentPage - windowSize);
  const end = Math.min(totalPages - 1, currentPage + windowSize);

  const pageBtn = (i) =>
    makeBtn(String(i + 1), i, i === currentPage, i === currentPage ? 'primary' : 'ghost');

  if (start > 0) {
    html += pageBtn(0);
    if (start > 1) html += `<span class="ellipsis">…</span>`;
  }
  for (let i = start; i <= end; i++) html += pageBtn(i);
  if (end < totalPages - 1) {
    if (end < totalPages - 2) html += `<span class="ellipsis">…</span>`;
    html += pageBtn(totalPages - 1);
  }

  html += makeBtn('›', Math.min(totalPages - 1, currentPage + 1), currentPage === totalPages - 1);
  html += makeBtn('»', totalPages - 1, currentPage === totalPages - 1);

  el.innerHTML = html;
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
// Gráfico (Chart.js) — linhas/barras com controle de séries
// ============================
let chart;

function popularAnosSelect(items) {
  const el = document.getElementById('yearSelect');
  if (!el) return;
  const years = Array.from(new Set(items.map(x => yearFromISO(x.createdAt)).filter(Boolean))).sort();
  const prev = el.value;
  el.innerHTML = `<option value="">Todos</option>` + years.map(y => `<option value="${y}">${y}</option>`).join('');
  if (prev && years.includes(Number(prev))) el.value = prev;

  // ativa séries por sexo só se houver algum item com sexo M/F
  const hasMale   = items.some(x => (x.paciente?.sexo || '').toUpperCase() === 'M');
  const hasFemale = items.some(x => (x.paciente?.sexo || '').toUpperCase() === 'F');
  const menWrap = document.getElementById('serieHomensWrap');
  const womenWrap = document.getElementById('serieMulheresWrap');
  if (menWrap) menWrap.style.display = hasMale ? 'inline-flex' : 'none';
  if (womenWrap) womenWrap.style.display = hasFemale ? 'inline-flex' : 'none';
}

function aggregateByMonth(items) {
  const acc = {
    total: Array(12).fill(0),
    atencao: Array(12).fill(0),
    normal: Array(12).fill(0),
    homens: Array(12).fill(0),
    mulheres: Array(12).fill(0),
  };
  items.forEach(x => {
    const mi = monthIndex(x.createdAt);
    if (mi === null) return;

    acc.total[mi] += 1;

    const s = (x.status || (x.suspicious ? 'atencao' : 'normal')).toLowerCase();
    if (s === 'atencao') acc.atencao[mi] += 1;
    if (s === 'normal')  acc.normal[mi]  += 1;

    const sexo = (x.paciente?.sexo || '').toUpperCase();
    if (sexo === 'M') acc.homens[mi] += 1;
    if (sexo === 'F') acc.mulheres[mi] += 1;
  });
  return acc;
}

function atualizarGrafico(items) {
  const el = document.getElementById('chartFraudes');
  if (!el || typeof Chart === 'undefined') return;

  const agg = aggregateByMonth(items);

  const showTotal    = document.getElementById('serieTotal')?.checked ?? true;
  const showAtencao  = document.getElementById('serieAtencao')?.checked ?? true;
  const showNormal   = document.getElementById('serieNormal')?.checked ?? true;
  const showHomens   = document.getElementById('serieHomens')?.checked ?? false;
  const showMulheres = document.getElementById('serieMulheres')?.checked ?? false;
  const isBars       = document.getElementById('chartBars')?.checked ?? false;

  const datasets = [];
  if (showTotal)    datasets.push({ label: 'Total',    data: agg.total,    tension: 0.35, borderWidth: 2 });
  if (showAtencao)  datasets.push({ label: 'Atenção',  data: agg.atencao,  tension: 0.35, borderWidth: 2 });
  if (showNormal)   datasets.push({ label: 'Normal',   data: agg.normal,   tension: 0.35, borderWidth: 2 });
  if (showHomens)   datasets.push({ label: 'Homens',   data: agg.homens,   tension: 0.35, borderWidth: 2 });
  if (showMulheres) datasets.push({ label: 'Mulheres', data: agg.mulheres, tension: 0.35, borderWidth: 2 });

  if (chart) chart.destroy();
  chart = new Chart(el, {
    type: isBars ? 'bar' : 'line',
    data: { labels: MONTHS_PT, datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y}`
          }
        }
      },
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 } }
      },
      elements: { point: { radius: 3, hoverRadius: 5 } }
    }
  });
}

// ============================
// Detalhes + Atualização de status
// ============================
async function verDetalhe(id) {
  try {
    const d = await apiGet(`/diagnosticos/${id}`);
    const modal = document.getElementById('detalheModal');
    if (!modal) return;
    modal.querySelector('.modal-body').innerHTML = renderDetail(d);
    modal.showModal();

    modal.querySelector('#btnEmAnalise').onclick = () => alterarStatus(id, 'em_analise');
    modal.querySelector('#btnAprovar').onclick   = () => alterarStatus(id, 'aprovado');
    modal.querySelector('#btnReprovar').onclick  = () => alterarStatus(id, 'reprovado');
  } catch (e) {
    console.error(e);
    showToast('Erro ao carregar detalhe.');
  }
}

function renderDetail(d) {
  const motivo = getMotivo(d);
  return `
    <h3>Paciente</h3>
    <p><b>Nome:</b> ${d.paciente?.nome ?? d.paciente_nome ?? '—'}<br>
       <b>CPF:</b> ${d.paciente?.cpf ?? d.cpf ?? '—'}</p>

    <h3>Consulta</h3>
    <p><b>Procedimento:</b> ${d.procedimento ?? '—'}<br>
       <b>Executado:</b> ${d.executado ?? '—'}<br>
       <b>Prescrição:</b> ${d.prescricao || '—'}</p>

    <h3>Dentista</h3>
    <p>${d.dentista?.nome ?? d.dentista ?? '—'}</p>

    <h3>Status Atual:</h3>
    <p>${badgeHtml(d.status)}</p>

    ${isAttentionLike(d) ? `<h3>Motivo (atenção):</h3><p>${motivo}</p>` : ``}

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
    showToast(`Status alterado para ${status.replace('_',' ')}`);
    document.getElementById('detalheModal')?.close();
    carregarTabela();
    // também atualiza o resumo
    carregarGrafico();
  } catch (e) {
    console.error(e);
    showToast('Erro ao atualizar status.');
  }
}

// ============================
// Gráfico de resumo (segundo gráfico)
// ============================
async function fetchAllDiagnosticos(params = {}) {
  const size = 100; // busca de 100 em 100
  let page = 0;
  let all = [];
  while (true) {
    const data = await apiGet('/diagnosticos', { ...params, page, size });
    const items = data?.content ?? [];
    all = all.concat(items);
    page++;
    if (page >= (data?.totalPages ?? 1)) break;
  }
  return all;
}

function getYear(d) {
  try { return new Date(d).getFullYear(); } catch { return null; }
}
function getMonthIdx(d) {
  try { return new Date(d).getMonth(); } catch { return null; } // 0..11
}

function buildMonthlySeries(items, year, filter) {
  const normal = Array(12).fill(0);
  const atencao = Array(12).fill(0);

  items.forEach(it => {
    const y = getYear(it.createdAt);
    const m = getMonthIdx(it.createdAt);
    if (y !== year || m == null) return;

    const st = String(it.status || '').toLowerCase();
    if (st === 'atencao') atencao[m]++;
    else normal[m]++;
  });

  if (filter === 'normal')  return { normal, atencao: Array(12).fill(0) };
  if (filter === 'atencao') return { normal: Array(12).fill(0), atencao };
  return { normal, atencao };
}

let fraudChart; // instancia Chart.js

function fillYearSelect(items) {
  const sel = document.getElementById('chartYear');
  if (!sel) return;

  const allYears = Array.from(
    new Set(items.map(x => getYear(x.createdAt)).filter(Boolean))
  ).sort((a, b) => a - b);

  const years = allYears.length ? allYears : [new Date().getFullYear()];
  sel.innerHTML = years.map(y => `<option value="${y}">${y}</option>`).join('');
  sel.value = String(years[years.length - 1]);
}

function renderFraudChart(series) {
  const ctx = document.getElementById('fraudChart');
  if (!ctx) return;

  const labels = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'];
  const data = {
    labels,
    datasets: [
      { label: 'Normal',  data: series.normal,  borderWidth: 2, tension: 0.25, pointRadius: 3 },
      { label: 'Atenção', data: series.atencao, borderWidth: 2, tension: 0.25, pointRadius: 3 }
    ]
  };
  const opts = {
    responsive: true,
    plugins: { legend: { position: 'top' }, tooltip: { mode: 'index', intersect: false } },
    scales: { y: { beginAtZero: true, ticks: { precision: 0 } } }
  };

  if (fraudChart) {
    fraudChart.data = data;
    fraudChart.options = opts;
    fraudChart.update();
  } else {
    fraudChart = new Chart(ctx, { type: 'line', data, options: opts });
  }
}

async function carregarGrafico() {
  try {
    const all = await fetchAllDiagnosticos({});
    fillYearSelect(all);

    const yearSel = document.getElementById('chartYear');
    const filterSel = document.getElementById('chartFilter');
    const year = parseInt(yearSel?.value ?? new Date().getFullYear(), 10);
    const filter = (filterSel?.value || 'todos').toLowerCase();

    const series = buildMonthlySeries(all, year, filter);
    renderFraudChart(series);
  } catch (e) {
    console.error(e);
    showToast('Não foi possível montar o resumo.');
  }
}

// ============================
// Inicialização ÚNICA
// ============================
document.addEventListener('DOMContentLoaded', () => {
  carregarTabela();
  carregarGrafico();

  document.getElementById('searchBtn')?.addEventListener('click', () => carregarTabela(0));
  document.getElementById('fraudeFilter')?.addEventListener('change', () => carregarTabela(0));
  document.getElementById('yearSelect')?.addEventListener('change', () => carregarTabela(0));

  ['serieTotal','serieAtencao','serieNormal','serieHomens','serieMulheres','chartBars']
    .forEach(id => document.getElementById(id)?.addEventListener('change', () => carregarTabela(0)));

  document.getElementById('chartYear')?.addEventListener('change', carregarGrafico);
  document.getElementById('chartFilter')?.addEventListener('change', carregarGrafico);

  // Tema & logout
  (function initTheme(){
    const saved = localStorage.getItem('ov-theme');
    if (saved === 'dark') document.documentElement.classList.add('dark');
  })();
  document.getElementById('btnDarkMode')?.addEventListener('click', () => {
    document.documentElement.classList.toggle('dark');
    localStorage.setItem('ov-theme', document.documentElement.classList.contains('dark') ? 'dark' : 'light');
  });
  const yearEl = document.getElementById('year');
  if (yearEl) yearEl.textContent = new Date().getFullYear();
  document.getElementById('btnLogout')?.addEventListener('click', () => {
    localStorage.removeItem('ov-auth');
    localStorage.removeItem('ov-auth-role');
    window.location.href = 'login.html';
  });
});
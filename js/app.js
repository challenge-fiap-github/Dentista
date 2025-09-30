// ====== Proteção simples de rota (precisa ter feito login) ======
(function guard(){
  const logged = localStorage.getItem('ov-auth');
  if(!logged){ window.location.href = 'login.html'; }
})();

// ====== Tema persistente ======
(function initTheme(){
  const saved = localStorage.getItem('ov-theme');
  if (saved === 'dark') document.documentElement.classList.add('dark');
})();
const btnDarkMode = document.getElementById('btnDarkMode');
btnDarkMode.addEventListener('click', () => {
  document.documentElement.classList.toggle('dark');
  localStorage.setItem('ov-theme', document.documentElement.classList.contains('dark') ? 'dark' : 'light');
});

// ====== Header extra ======
document.getElementById('year').textContent = new Date().getFullYear();
const btnLogout = document.getElementById('btnLogout');
btnLogout.addEventListener('click', () => {
  localStorage.removeItem('ov-auth');
  window.location.href = 'login.html';
});

// ====== Toast ======
const toast = document.getElementById('toast');
function showToast(msg, ms=2400){
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(()=>toast.classList.remove('show'), ms);
}

// ====== CPF: máscara + validação ======
function maskCPF(v) {
  return v
    .replace(/\D/g,'')
    .slice(0,11)
    .replace(/(\d{3})(\d)/, '$1.$2')
    .replace(/(\d{3})(\d)/, '$1.$2')
    .replace(/(\d{3})(\d{1,2})$/, '$1-$2');
}
function isValidCPF(cpf) {
  cpf = (cpf || '').replace(/[^\d]+/g, '');
  if (!cpf || cpf.length !== 11) return false;
  if (/^(\d)\1+$/.test(cpf)) return false;
  let soma = 0, resto;
  for (let i = 1; i <= 9; i++) soma += parseInt(cpf.substring(i-1, i)) * (11 - i);
  resto = (soma * 10) % 11;
  if (resto === 10 || resto === 11) resto = 0;
  if (resto !== parseInt(cpf.substring(9, 10))) return false;
  soma = 0;
  for (let i = 1; i <= 10; i++) soma += parseInt(cpf.substring(i-1, i)) * (12 - i);
  resto = (soma * 10) % 11;
  if (resto === 10 || resto === 11) resto = 0;
  return resto === parseInt(cpf.substring(10, 11));
}

// ====== elementos ======
const form = document.getElementById('consultaForm');
const paciente = document.getElementById('paciente');
const cpfInput = document.getElementById('cpf');
const procedimento = document.getElementById('procedimento');
const executado = document.getElementById('executado');
const prescricao = document.getElementById('prescricao');
const btnLimpar = document.getElementById('btnLimpar');

// [MOTIVOS] refs
const motivosWrap = document.getElementById('motivosWrap');
const motivosPanel = motivosWrap?.querySelector('.multiselect-panel');
const motivosChecks = motivosPanel?.querySelectorAll('input[type="checkbox"]');
const motivosHidden = document.getElementById('motivosHidden');
const motivosChips = document.getElementById('motivosChips');
const motivosCount = document.getElementById('motivosCount');
document.getElementById('btnMotivosLimpar')?.addEventListener('click', () => {
  motivosChecks.forEach(c => c.checked = false);
  updateMotivos();
});
document.getElementById('btnMotivosOk')?.addEventListener('click', () => {
  motivosWrap.removeAttribute('open');
});
motivosChecks?.forEach(c => c.addEventListener('change', updateMotivos));

function updateMotivos(){
  const sel = Array.from(motivosChecks || []).filter(c => c.checked).map(c => c.value);
  // hidden para required do HTML
  motivosHidden.value = sel.join('; ');
  // chips
  motivosChips.innerHTML = sel.length
    ? sel.map(v => `<span class="chip" data-v="${v}">${v}<button type="button" aria-label="Remover ${v}">×</button></span>`).join('')
    : `<span class="hint">Nenhum motivo selecionado.</span>`;
  motivosCount.textContent = `${sel.length} selecionado(s)`;
  // remover chip
  motivosChips.querySelectorAll('.chip button').forEach(btn => {
    btn.addEventListener('click', () => {
      const v = btn.parentElement.getAttribute('data-v');
      const el = Array.from(motivosChecks).find(c => c.value === v);
      if (el){ el.checked = false; updateMotivos(); }
    });
  });
}
updateMotivos();

// ====== Pop-up de sucesso ======
const successModal = document.getElementById('successModal');
const closeSuccess = document.getElementById('closeSuccess');
const okSuccess = document.getElementById('okSuccess');

// Contadores
function bindCounter(el){
  const span = document.querySelector(`[data-count="${el.id}"]`);
  if(!span) return;
  const update = () => span.textContent = el.value.length.toString();
  el.addEventListener('input', update);
  update();
}
[paciente, procedimento, executado, prescricao].forEach(bindCounter);

// Máscara CPF
cpfInput.addEventListener('input', () => {
  const pos = cpfInput.selectionStart;
  cpfInput.value = maskCPF(cpfInput.value);
  cpfInput.setSelectionRange(pos, pos);
});

// ====== rascunho (inclui motivos) ======
const DRAFT_KEY = 'ov-consulta-draft-v1';
function saveDraft(){
  const data = {
    paciente: paciente.value,
    cpf: cpfInput.value,
    procedimento: procedimento.value,
    executado: executado.value,
    prescricao: prescricao.value,
    motivos: motivosHidden.value,   // [MOTIVOS]
    updatedAt: new Date().toISOString()
  };
  localStorage.setItem(DRAFT_KEY, JSON.stringify(data));
}
function loadDraft(){
  const str = localStorage.getItem(DRAFT_KEY);
  if(!str) return;
  try{
    const data = JSON.parse(str);
    paciente.value = data.paciente || '';
    cpfInput.value = data.cpf || '';
    procedimento.value = data.procedimento || '';
    executado.value = data.executado || '';
    prescricao.value = data.prescricao || '';
    // [MOTIVOS] restaura
    if (data.motivos){
      const set = new Set(data.motivos.split(';').map(s => s.trim()).filter(Boolean));
      motivosChecks.forEach(c => c.checked = set.has(c.value));
      updateMotivos();
    }
    [paciente, procedimento, executado, prescricao].forEach(bindCounter);
  }catch(e){}
}
[paciente, cpfInput, procedimento, executado, prescricao].forEach(el => el.addEventListener('input', saveDraft));
motivosPanel?.addEventListener('change', saveDraft);
loadDraft();

// Submit com validações obrigatórias e CPF
form.addEventListener('submit', (e) => {
  e.preventDefault();
  const payload = buildPayload(true);
  if(!payload) return;

  // Aqui faria o POST real…
  // fetch('/api/consultas', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) })

  localStorage.removeItem(DRAFT_KEY);
  successModal.showModal();
});

// Limpar
btnLimpar.addEventListener('click', () => {
  form.reset();
  [paciente, procedimento, executado, prescricao].forEach(bindCounter);
  saveDraft();
  showToast('Formulário limpo.');
});

// Fechar modal
closeSuccess.addEventListener('click', () => successModal.close());
okSuccess.addEventListener('click', () => {
  successModal.close();
  form.reset();
  [paciente, procedimento, executado, prescricao].forEach(bindCounter);
});

// ====== payload (inclui motivos) ======
function buildPayload(validate){
  const motivosArr = motivosHidden.value
    .split(';')
    .map(s => s.trim())
    .filter(Boolean);

  const data = {
    paciente: paciente.value.trim(),
    cpf: cpfInput.value.trim(),
    procedimento: procedimento.value.trim(),
    executado: executado.value.trim(),
    prescricao: prescricao.value.trim(),
    motivos: motivosArr,          // [MOTIVOS] array padronizado
    empresa: 'OdontoVision',
    timestamp: new Date().toISOString()
  };

  if(!validate) return data;

  if(!data.paciente){ showToast('Informe o nome do paciente.'); paciente.focus(); return null; }
  if(!data.cpf || !isValidCPF(data.cpf)){ showToast('CPF inválido.'); cpfInput.focus(); return null; }
  if(!data.procedimento){ showToast('Informe o procedimento.'); procedimento.focus(); return null; }
  if(!data.executado){ showToast('Descreva o que foi executado.'); executado.focus(); return null; }
  if(data.motivos.length === 0){  // [MOTIVOS] obrigatório
    showToast('Selecione pelo menos um motivo da consulta.');
    motivosWrap.setAttribute('open','');
    return null;
  }

  return data;
}

// ============================
// Coleta Motivos (multiselect)
// ============================
function getMotivosSelecionados(){
  const wrap = document.getElementById('motivosWrap');
  const checks = wrap.querySelectorAll('input[type="checkbox"]');
  const out = [];
  checks.forEach(ch => { if(ch.checked) out.push(ch.value); });
  return out;
}
function renderMotivosSummary(){
  const sel = getMotivosSelecionados();
  const sum = document.getElementById('motivosSummary');
  if (!sel.length) { sum.innerHTML = 'Selecione um ou mais…'; return; }
  sum.innerHTML = sel.map(v => `<span class="chip">${v}</span>`).join('');
}
document.querySelectorAll('#motivosWrap input[type="checkbox"]').forEach(ch => {
  ch.addEventListener('change', renderMotivosSummary);
});
renderMotivosSummary();

// ============================
// Envio do formulário (POST)
// ============================
async function postConsulta(payload){
  const res = await fetch('http://127.0.0.1:5001/api/consultas', {
    method: 'POST',
    headers: { 'Content-Type':'application/json' },
    body: JSON.stringify(payload)
  });
  if(!res.ok){
    const txt = await res.text();
    throw new Error(txt || 'Falha ao enviar.');
  }
  return res.json();
}

// ===== API =====
const API_BASE = 'http://127.0.0.1:5001/api';

async function apiPost(path, body){
  const res = await fetch(API_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if(!res.ok) throw new Error(await res.text());
  return res.json();
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = buildPayload(true);
  if(!payload) return;

  // mapeia para o que a API espera
  const body = {
    paciente_nome: payload.paciente,
    cpf: payload.cpf.replace(/\D/g,''),
    procedimento: payload.procedimento,
    executado: payload.executado,
    prescricao: payload.prescricao,
    dentista: 'Dentista',                 // se quiser, trocamos pelo nome logado
    created_at: payload.timestamp,
    motivos: payload.motivos              // array
  };

  try{
    await apiPost('/consultas', body);
    localStorage.removeItem(DRAFT_KEY);
    showToast('Registro salvo.');
    // opcional: ir direto pro Admin
    // window.location.href = 'admin.html';
    successModal.showModal();
  }catch(err){
    console.error(err);
    showToast('Erro ao salvar registro.');
  }
});

// ao fechar o modal de sucesso, redireciona p/ Admin
okSuccess.addEventListener('click', () => {
  successModal.close();
  window.location.href = 'admin.html';
});

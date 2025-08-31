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

// ====== Elementos do formulário ======
const form = document.getElementById('consultaForm');
const paciente = document.getElementById('paciente');
const cpfInput = document.getElementById('cpf');
const procedimento = document.getElementById('procedimento');
const executado = document.getElementById('executado');
const prescricao = document.getElementById('prescricao');
const btnLimpar = document.getElementById('btnLimpar');

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

// Rascunho automático (localStorage)
const DRAFT_KEY = 'ov-consulta-draft-v1';
function saveDraft(){
  const data = {
    paciente: paciente.value,
    cpf: cpfInput.value,
    procedimento: procedimento.value,
    executado: executado.value,
    prescricao: prescricao.value,
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
    [paciente, procedimento, executado, prescricao].forEach(bindCounter);
  }catch(e){}
}
[paciente, cpfInput, procedimento, executado, prescricao].forEach(el => el.addEventListener('input', saveDraft));
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

// Helpers
function buildPayload(validate){
  const data = {
    paciente: paciente.value.trim(),
    cpf: cpfInput.value.trim(),
    procedimento: procedimento.value.trim(),
    executado: executado.value.trim(),
    prescricao: prescricao.value.trim(),
    empresa: 'OdontoVision',
    timestamp: new Date().toISOString()
  };

  if(!validate) return data;

  if(!data.paciente){ showToast('Informe o nome do paciente.'); paciente.focus(); return null; }
  if(!data.cpf || !isValidCPF(data.cpf)){ showToast('CPF inválido.'); cpfInput.focus(); return null; }
  if(!data.procedimento){ showToast('Informe o procedimento.'); procedimento.focus(); return null; }
  if(!data.executado){ showToast('Descreva o que foi executado.'); executado.focus(); return null; }

  return data;
}

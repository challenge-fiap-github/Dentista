// Tema persistente
(function initTheme(){
  const saved = localStorage.getItem('ov-theme');
  if (saved === 'dark') document.documentElement.classList.add('dark');
})();
const btnDarkMode = document.getElementById('btnDarkMode');
btnDarkMode.addEventListener('click', () => {
  document.documentElement.classList.toggle('dark');
  localStorage.setItem('ov-theme', document.documentElement.classList.contains('dark') ? 'dark' : 'light');
});

// Ano
document.getElementById('year').textContent = new Date().getFullYear();

// Toast
const toast = document.getElementById('toast');
function showToast(msg, ms=2400){
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(()=>toast.classList.remove('show'), ms);
}

// Login
const form = document.getElementById('loginForm');
form.addEventListener('submit', (e) => {
  e.preventDefault();
  const usuario = document.getElementById('usuario').value.trim();
  const senha = document.getElementById('senha').value;

  if(!usuario || !senha){
    showToast('Preencha usuário e senha.');
    return;
  }
  if(usuario !== 'dentista' || senha !== 'dentista123'){
    showToast('Usuário ou senha incorretos.');
    return;
  }

  // Marca login simples e segue para index
  localStorage.setItem('ov-auth', 'dentista'); // valor simbólico
  window.location.href = 'index.html';
});

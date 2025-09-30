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

// Login (ÚNICO submit listener)
const form = document.getElementById('loginForm');
form.addEventListener('submit', (e) => {
  e.preventDefault();
  const usuario = document.getElementById('usuario').value.trim();
  const senha = document.getElementById('senha').value;

  if(!usuario || !senha){
    showToast('Preencha usuário e senha.');
    return;
  }

  // dentista
  if(usuario === 'dentista' && senha === 'dentista123'){
    localStorage.setItem('ov-auth', 'dentista');
    localStorage.setItem('ov-auth-role', 'dentista');
    window.location.href = 'index.html'; // html/index.html
    return;
  }

  // admin
  if(usuario === 'admin' && senha === 'admin123'){
    localStorage.setItem('ov-auth', 'admin');
    localStorage.setItem('ov-auth-role', 'admin');
    window.location.href = 'admin.html'; // html/admin.html
    return;
  }

  showToast('Usuário ou senha incorretos.');
});

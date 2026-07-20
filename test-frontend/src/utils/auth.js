export function logout(reason) {
  localStorage.removeItem('token');
  localStorage.removeItem('username');
  if (reason) {
    sessionStorage.setItem('logoutReason', reason);
  }
  window.location.href = '/';
}

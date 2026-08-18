export function getSessionId() {
  const SESSION_KEY = 'pace_session_id'
  let sessionId = localStorage.getItem(SESSION_KEY)
  
  if (!sessionId) {
    sessionId = 'sess-' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15)
    localStorage.setItem(SESSION_KEY, sessionId)
  }
  
  return sessionId
}

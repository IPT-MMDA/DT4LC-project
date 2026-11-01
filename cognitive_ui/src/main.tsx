import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { useAppStore } from './store/useAppStore'
import { applyTheme } from './utils/theme'

// Before the first paint, so a dark theme does not flash light; the routes keep it applied.
applyTheme(useAppStore.getState().theme)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

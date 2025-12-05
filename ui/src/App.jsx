import { useState, useEffect } from 'react'
import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import {
  Play,
  BarChart3,
  History,
  Activity,
  Zap,
  Menu,
  X,
  ChevronLeft,
  ChevronRight,
  Github,
  ExternalLink,
  Sparkles,
  Sun,
  Moon,
  Key
} from 'lucide-react'

import RunManager from './components/RunManager'
import ProgressViewer from './components/ProgressViewer'
import ModelComparison from './components/ModelComparison'
import RunHistory from './components/RunHistory'
import RunDetail from './components/RunDetail'
import ApiKeysModal from './components/ApiKeysModal'

const API_BASE = '/api'

function App() {
  const [activeRun, setActiveRun] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [runningCount, setRunningCount] = useState(0)
  const [historyCount, setHistoryCount] = useState(0)
  const [showApiKeys, setShowApiKeys] = useState(false)
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode')
    return saved === null ? true : saved === 'true'
  })
  const location = useLocation()

  // Apply dark mode class to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem('darkMode', darkMode.toString())
  }, [darkMode])

  // Listen for API Keys modal open events
  useEffect(() => {
    const handleOpenApiKeys = () => setShowApiKeys(true)
    window.addEventListener('openApiKeys', handleOpenApiKeys)
    return () => window.removeEventListener('openApiKeys', handleOpenApiKeys)
  }, [])

  // Fetch counts for badges
  useEffect(() => {
    const fetchCounts = async () => {
      try {
        const queueRes = await fetch('/api/queue/status')
        const queueData = await queueRes.json()
        setRunningCount(queueData.running ? 1 : 0)

        const historyRes = await fetch('/api/runs')
        const historyData = await historyRes.json()
        setHistoryCount(historyData.length || 0)
      } catch (e) {
        // Silently fail
      }
    }
    fetchCounts()
    const interval = setInterval(fetchCounts, 5000)
    return () => clearInterval(interval)
  }, [])

  const navItems = [
    {
      path: '/',
      icon: Play,
      label: 'Run Manager',
      description: 'Configure & start evaluations'
    },
    {
      path: '/progress',
      icon: Activity,
      label: 'Progress',
      description: 'Live evaluation status',
      badge: runningCount > 0 ? 'running' : null
    },
    {
      path: '/compare',
      icon: BarChart3,
      label: 'Compare',
      description: 'Analyze & visualize results'
    },
    {
      path: '/history',
      icon: History,
      label: 'History',
      description: 'Past evaluation runs',
      badge: historyCount > 0 ? historyCount : null
    },
  ]

  return (
    <div className="min-h-screen bg-background flex transition-colors duration-300">
      {/* Sidebar */}
      <aside
        className={`
          ${sidebarOpen ? 'w-72' : 'w-20'} 
          bg-surface border-r border-border-default
          transition-all duration-300 
          flex flex-col
          relative
        `}
      >
        {/* Logo Header */}
        <div className="p-4 border-b border-border-default">
          <div className="flex items-center gap-3 justify-between">
            <div className="flex items-center gap-3">
              <div className="w-11 h-11 bg-gradient-to-br from-amber-500 to-orange-500 rounded-xl flex items-center justify-center shadow-lg shadow-amber-500/25">
                <Zap className="w-6 h-6 text-white" />
              </div>
              {sidebarOpen && (
                <div className="animate-fade-in">
                  <h1 className="text-lg font-bold text-primary tracking-tight">LLM Bench</h1>
                  <p className="text-xs text-tertiary font-medium">Dashboard v2.3</p>
                </div>
              )}
            </div>
            {/* Dark/Light Mode Toggle */}
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="p-2 rounded-lg bg-surface-hover hover:bg-surface-active text-secondary hover:text-primary transition-all"
              title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            >
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-3 space-y-1">
          {navItems.map(({ path, icon: Icon, label, description, badge }) => {
            const isActive = location.pathname === path ||
              (path === '/' && location.pathname === '/') ||
              (path !== '/' && location.pathname.startsWith(path))

            return (
              <NavLink
                key={path}
                to={path}
                className={`
                  group flex items-center gap-3 px-3 py-3 rounded-xl
                  transition-all duration-200
                  ${isActive
                    ? 'bg-interactive-bg text-interactive shadow-sm'
                    : 'text-secondary hover:bg-surface-hover hover:text-primary'
                  }
                `}
              >
                <div className={`
                  relative flex items-center justify-center w-9 h-9 rounded-lg
                  transition-all duration-200
                  ${isActive
                    ? 'bg-interactive/20'
                    : 'bg-surface-active group-hover:bg-surface-active'
                  }
                `}>
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  {badge === 'running' && (
                    <span className="absolute -top-1 -right-1 w-3 h-3 bg-interactive rounded-full animate-pulse shadow-lg shadow-interactive/50" />
                  )}
                </div>
                {sidebarOpen && (
                  <div className="flex-1 min-w-0 animate-fade-in">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">{label}</span>
                      {badge && badge !== 'running' && (
                        <span className="px-2 py-0.5 text-xs font-medium bg-surface-active text-secondary rounded-full">
                          {badge}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-tertiary truncate mt-0.5">{description}</p>
                  </div>
                )}
              </NavLink>
            )
          })}
        </nav>

        {/* Footer */}
        {sidebarOpen && (
          <div className="p-4 border-t border-border-default animate-fade-in space-y-3">
            <a
              href="https://github.com/NahuelGiudizi/llm-evaluation"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-xs text-tertiary hover:text-secondary transition-colors"
            >
              <Github className="w-4 h-4" />
              <span>View on GitHub</span>
              <ExternalLink className="w-3 h-3" />
            </a>

            <div className="flex items-center gap-2 text-xs text-secondary">
              <Sparkles className="w-3 h-3" />
              <span>llm-benchmark-toolkit</span>
            </div>
          </div>
        )}

        {/* Collapse Toggle */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="
            absolute -right-3 top-20
            w-6 h-6 bg-surface border border-border-default
            rounded-full flex items-center justify-center
            text-tertiary hover:text-primary hover:bg-surface-hover
            transition-all duration-200
            shadow-lg
          "
        >
          {sidebarOpen ? (
            <ChevronLeft className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
        </button>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto bg-background">
        <div className="p-6 max-w-7xl mx-auto">
          <Routes>
            <Route path="/" element={<RunManager onRunStart={setActiveRun} />} />
            <Route path="/progress" element={<ProgressViewer activeRun={activeRun} />} />
            <Route path="/compare" element={<ModelComparison />} />
            <Route path="/history" element={<RunHistory />} />
            <Route path="/run/:id" element={<RunDetail />} />
          </Routes>
        </div>
      </main>

      {/* API Keys Modal */}
      <ApiKeysModal isOpen={showApiKeys} onClose={() => setShowApiKeys(false)} />
    </div>
  )
}

export default App

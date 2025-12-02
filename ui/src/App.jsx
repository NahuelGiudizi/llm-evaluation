import { useState, useEffect } from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import {
  Play,
  BarChart3,
  History,
  Settings,
  Activity,
  Zap,
  Menu,
  X
} from 'lucide-react'

import RunManager from './components/RunManager'
import ProgressViewer from './components/ProgressViewer'
import ModelComparison from './components/ModelComparison'
import RunHistory from './components/RunHistory'
import RunDetail from './components/RunDetail'

const API_BASE = '/api'

function App() {
  const [activeRun, setActiveRun] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const navItems = [
    { path: '/', icon: Play, label: 'Run Manager' },
    { path: '/progress', icon: Activity, label: 'Progress' },
    { path: '/compare', icon: BarChart3, label: 'Compare Models' },
    { path: '/history', icon: History, label: 'Run History' },
  ]

  return (
    <div className="min-h-screen bg-slate-900 flex">
      {/* Sidebar */}
      <aside
        className={`${sidebarOpen ? 'w-64' : 'w-16'} bg-slate-800 border-r border-slate-700 transition-all duration-300 flex flex-col`}
      >
        {/* Logo */}
        <div className="p-4 border-b border-slate-700 flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-purple-500 rounded-lg flex items-center justify-center">
            <Zap className="w-6 h-6 text-white" />
          </div>
          {sidebarOpen && (
            <div>
              <h1 className="text-lg font-bold text-white">LLM Bench</h1>
              <p className="text-xs text-slate-400">Dashboard v1.0</p>
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-2">
          {navItems.map(({ path, icon: Icon, label }) => (
            <NavLink
              key={path}
              to={path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-colors ${isActive
                  ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
                  : 'text-slate-400 hover:bg-slate-700/50 hover:text-white'
                }`
              }
            >
              <Icon className="w-5 h-5 flex-shrink-0" />
              {sidebarOpen && <span className="text-sm font-medium">{label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Sidebar toggle */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-4 border-t border-slate-700 text-slate-400 hover:text-white flex items-center gap-2"
        >
          {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          {sidebarOpen && <span className="text-sm">Collapse</span>}
        </button>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <div className="p-6">
          <Routes>
            <Route path="/" element={<RunManager onRunStart={setActiveRun} />} />
            <Route path="/progress" element={<ProgressViewer activeRun={activeRun} />} />
            <Route path="/compare" element={<ModelComparison />} />
            <Route path="/history" element={<RunHistory />} />
            <Route path="/run/:id" element={<RunDetail />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

export default App

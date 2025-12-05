import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  ArrowLeft,
  Loader2,
  CheckCircle,
  XCircle,
  Clock,
  Download,
  Terminal,
  BarChart3,
  FileJson,
  Image,
  Cpu,
  HardDrive,
  Monitor,
  BookOpen,
  FileText,
  FileSpreadsheet,
  Quote,
  ChevronDown
} from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts'
import { fetchRun, fetchLogs } from '../api'
import ScenariosViewer from './ScenariosViewer'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

function RunDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [run, setRun] = useState(null)
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('results')
  const [showExportMenu, setShowExportMenu] = useState(false)
  const [exporting, setExporting] = useState(null)

  useEffect(() => {
    loadData()
  }, [id])

  // Close export menu when clicking outside
  useEffect(() => {
    function handleClickOutside(e) {
      if (showExportMenu && !e.target.closest('.export-menu-container')) {
        setShowExportMenu(false)
      }
    }
    document.addEventListener('click', handleClickOutside)
    return () => document.removeEventListener('click', handleClickOutside)
  }, [showExportMenu])

  async function loadData() {
    setLoading(true)
    try {
      const [runData, logsData] = await Promise.all([
        fetchRun(id),
        fetchLogs(id).catch(() => ({ logs: [] }))
      ])
      setRun(runData)
      setLogs(logsData.logs || [])
    } catch (err) {
      console.error('Failed to load run:', err)
    } finally {
      setLoading(false)
    }
  }

  async function handleExport(format) {
    setExporting(format)
    setShowExportMenu(false)
    try {
      const response = await fetch(`${API_BASE}/run/${id}/export/${format}`)
      if (!response.ok) throw new Error('Export failed')

      const blob = await response.blob()
      const contentDisposition = response.headers.get('Content-Disposition')
      const filenameMatch = contentDisposition?.match(/filename="(.+)"/)
      const filename = filenameMatch ? filenameMatch[1] : `${id}_export.${format}`

      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      console.error('Export failed:', err)
      alert('Export failed: ' + err.message)
    } finally {
      setExporting(null)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
      </div>
    )
  }

  if (!run) {
    return (
      <div className="text-center py-20">
        <XCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-primary mb-2">Run Not Found</h2>
        <p className="text-tertiary mb-6">The requested run could not be found</p>
        <button
          onClick={() => navigate('/history')}
          className="btn btn-primary"
        >
          Back to History
        </button>
      </div>
    )
  }

  // Helper to extract score from different benchmark result formats
  function extractScore(benchmarkName, data) {
    // Try different field names used by different benchmarks
    if (data.score !== undefined) return data.score
    if (data.mmlu_accuracy !== undefined) return data.mmlu_accuracy
    if (data.truthfulness_score !== undefined) return data.truthfulness_score
    if (data.hellaswag_accuracy !== undefined) return data.hellaswag_accuracy
    if (data.arc_accuracy !== undefined) return data.arc_accuracy
    if (data.winogrande_accuracy !== undefined) return data.winogrande_accuracy
    if (data.commonsenseqa_accuracy !== undefined) return data.commonsenseqa_accuracy
    if (data.boolq_accuracy !== undefined) return data.boolq_accuracy
    if (data.safetybench_accuracy !== undefined) return data.safetybench_accuracy
    if (data.donotanswer_refusal_rate !== undefined) return data.donotanswer_refusal_rate
    if (data.donotanswer_accuracy !== undefined) return data.donotanswer_accuracy
    if (data.gsm8k_accuracy !== undefined) return data.gsm8k_accuracy
    if (data.accuracy !== undefined) return data.accuracy
    // Fallback: calculate from correct/total if available
    if (data.correct !== undefined && data.questions_tested !== undefined && data.questions_tested > 0) {
      return data.correct / data.questions_tested
    }
    if (data.correct !== undefined && data.scenarios_tested !== undefined && data.scenarios_tested > 0) {
      return data.correct / data.scenarios_tested
    }
    return 0
  }

  // Helper to get total questions/scenarios tested
  function getTotal(data) {
    return data.questions_tested || data.scenarios_tested || data.total || 0
  }

  // Format benchmark name for display
  function formatBenchmarkName(name) {
    const nameMap = {
      'mmlu': 'MMLU',
      'truthfulqa': 'TruthfulQA',
      'hellaswag': 'HellaSwag',
      'arc': 'ARC',
      'winogrande': 'WinoGrande',
      'commonsenseqa': 'CommonsenseQA',
      'boolq': 'BoolQ',
      'safetybench': 'SafetyBench',
      'donotanswer': 'Do-Not-Answer',
      'gsm8k': 'GSM8K',
    }
    return nameMap[name.toLowerCase()] || name.toUpperCase()
  }

  // Calculate total inference time from benchmark elapsed_time values
  function getTotalInferenceTime() {
    if (!run.results) return null
    let total = 0
    let hasTime = false
    for (const data of Object.values(run.results)) {
      if (data.elapsed_time) {
        total += data.elapsed_time
        hasTime = true
      }
    }
    return hasTime ? total : null
  }

  const totalInferenceTime = getTotalInferenceTime()

  // Prepare chart data
  const chartData = run.results
    ? Object.entries(run.results).map(([name, data]) => {
      const score = extractScore(name, data)
      return {
        name: formatBenchmarkName(name),
        score: (score * 100).toFixed(1),
        color: score >= 0.7 ? '#10b981' : score >= 0.5 ? '#f59e0b' : '#ef4444'
      }
    })
    : []

  const avgScore = chartData.length > 0
    ? chartData.reduce((sum, d) => sum + parseFloat(d.score), 0) / chartData.length
    : 0

  // Normalize status - handle both "completed" and "RunStatus.COMPLETED" formats
  const normalizeStatus = (status) => {
    if (!status) return 'unknown'
    const s = String(status).toLowerCase()
    if (s.includes('completed') || s.includes('complete')) return 'completed'
    if (s.includes('running')) return 'running'
    if (s.includes('failed') || s.includes('error')) return 'failed'
    if (s.includes('cancelled') || s.includes('canceled')) return 'cancelled'
    if (s.includes('queued')) return 'queued'
    if (s.includes('pending')) return 'pending'
    return s
  }

  const status = normalizeStatus(run.status)
  const statusDisplay = status.charAt(0).toUpperCase() + status.slice(1)

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <button
          onClick={() => navigate('/history')}
          className="flex items-center gap-2 text-tertiary hover:text-primary mb-4 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to History
        </button>

        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold text-primary mb-2">{run.model}</h1>
            <div className="flex items-center gap-4 text-tertiary">
              <span className="px-2 py-1 bg-surface-active rounded text-sm">{run.provider}</span>
              <span>{new Date(run.started_at).toLocaleString()}</span>
              <span className={`flex items-center gap-1 ${status === 'completed' ? 'text-green-400' :
                status === 'failed' ? 'text-red-400' : 'text-primary-400'
                }`}>
                {status === 'completed' && <CheckCircle className="w-4 h-4" />}
                {status === 'failed' && <XCircle className="w-4 h-4" />}
                {status === 'running' && <Loader2 className="w-4 h-4 animate-spin" />}
                {statusDisplay}
              </span>
            </div>
          </div>

          {status === 'completed' && (
            <div className="flex items-start gap-6">
              {/* Export Menu */}
              <div className="relative export-menu-container">
                <button
                  onClick={() => setShowExportMenu(!showExportMenu)}
                  className="flex items-center gap-2 px-4 py-2 bg-surface-active hover:bg-surface-hover rounded-lg text-primary transition-colors"
                >
                  {exporting ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Download className="w-4 h-4" />
                  )}
                  Export
                  <ChevronDown className="w-4 h-4" />
                </button>

                {showExportMenu && (
                  <div className="absolute right-0 mt-2 w-56 bg-surface border border-default rounded-lg shadow-xl z-50 overflow-hidden">
                    <button
                      onClick={() => handleExport('json')}
                      className="w-full flex items-center gap-3 px-4 py-3 hover:bg-surface-active transition-colors text-left"
                    >
                      <FileJson className="w-5 h-5 text-blue-400" />
                      <div>
                        <div className="text-primary font-medium">JSON</div>
                        <div className="text-xs text-tertiary">Full results + manifest</div>
                      </div>
                    </button>
                    <button
                      onClick={() => handleExport('csv')}
                      className="w-full flex items-center gap-3 px-4 py-3 hover:bg-surface-active transition-colors text-left"
                    >
                      <FileSpreadsheet className="w-5 h-5 text-green-400" />
                      <div>
                        <div className="text-primary font-medium">CSV</div>
                        <div className="text-xs text-tertiary">Spreadsheet format</div>
                      </div>
                    </button>
                    <button
                      onClick={() => handleExport('latex')}
                      className="w-full flex items-center gap-3 px-4 py-3 hover:bg-surface-active transition-colors text-left"
                    >
                      <FileText className="w-5 h-5 text-amber-400" />
                      <div>
                        <div className="text-primary font-medium">LaTeX Table</div>
                        <div className="text-xs text-tertiary">Ready for papers</div>
                      </div>
                    </button>
                    <button
                      onClick={() => handleExport('bibtex')}
                      className="w-full flex items-center gap-3 px-4 py-3 hover:bg-surface-active transition-colors text-left"
                    >
                      <Quote className="w-5 h-5 text-purple-400" />
                      <div>
                        <div className="text-primary font-medium">BibTeX</div>
                        <div className="text-xs text-tertiary">Citations & references</div>
                      </div>
                    </button>
                  </div>
                )}
              </div>

              {/* Score */}
              <div className="text-right">
                <div className="text-4xl font-bold text-primary-400">
                  {avgScore.toFixed(1)}%
                </div>
                <div className="text-sm text-tertiary">Average Score</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="mb-6 border-b border-default">
        <div className="flex gap-1">
          {['results', 'scenarios', 'logs', 'config', 'system'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-3 text-sm font-medium capitalize border-b-2 transition-colors flex items-center gap-2 ${activeTab === tab
                ? 'border-primary-500 text-primary-400'
                : 'border-transparent text-tertiary hover:text-primary'
                }`}
            >
              {tab === 'scenarios' && <BookOpen className="w-4 h-4" />}
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Results Tab */}
      {activeTab === 'results' && (
        <div className="space-y-6">
          {/* Chart */}
          {chartData.length > 0 && (
            <div className="bg-surface rounded-xl border border-default p-6">
              <h3 className="text-lg font-semibold text-primary mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-tertiary" />
                Benchmark Scores
              </h3>
              <ResponsiveContainer width="100%" height={Math.max(300, chartData.length * 40)}>
                <BarChart data={chartData} layout="vertical" margin={{ left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-border-default" />
                  <XAxis type="number" domain={[0, 100]} tick={{ fill: 'currentColor' }} className="text-tertiary" />
                  <YAxis dataKey="name" type="category" width={150} tick={{ fill: 'currentColor', fontSize: 12 }} className="text-tertiary" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgb(var(--color-surface) / 1)',
                      border: '1px solid rgb(var(--color-border-default) / 1)',
                      borderRadius: '12px',
                      padding: '12px',
                      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                      color: 'rgb(var(--color-text-primary) / 1)'
                    }}
                    formatter={(value) => [`${value}%`, 'Score']}
                    labelStyle={{
                      color: 'rgb(var(--color-text-primary) / 1)',
                      fontWeight: '600',
                      marginBottom: '4px'
                    }}
                    wrapperStyle={{ zIndex: 1000, pointerEvents: 'none' }}
                    cursor={{ fill: 'rgba(0, 0, 0, 0.05)' }}
                  />
                  <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Detailed Results */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {run.results && Object.entries(run.results).map(([name, data]) => {
              const score = extractScore(name, data)
              const total = getTotal(data)
              return (
                <div
                  key={name}
                  className="bg-surface dark:bg-surface rounded-xl border border-subtle dark:border-default p-4"
                >
                  <h4 className="font-semibold text-slate-900 dark:text-primary mb-3 text-lg">{formatBenchmarkName(name)}</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-secondary dark:text-tertiary">Score</span>
                      <span className={`font-semibold ${score >= 0.7 ? 'text-green-600 dark:text-green-400' :
                        score >= 0.5 ? 'text-yellow-600 dark:text-yellow-400' : 'text-red-600 dark:text-red-400'
                        }`}>
                        {(score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-secondary dark:text-tertiary">Correct</span>
                      <span className="text-slate-900 dark:text-primary">{data.correct || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-secondary dark:text-tertiary">Total</span>
                      <span className="text-slate-900 dark:text-primary">{total}</span>
                    </div>
                    {data.elapsed_time && (
                      <div className="flex justify-between">
                        <span className="text-secondary dark:text-tertiary">Time</span>
                        <span className="text-slate-900 dark:text-primary">{data.elapsed_time.toFixed(2)}s</span>
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Scenarios Tab */}
      {activeTab === 'scenarios' && (
        <div className="bg-surface rounded-xl border border-default p-6">
          <div className="flex items-center gap-2 mb-4">
            <BookOpen className="w-5 h-5 text-tertiary" />
            <h3 className="text-lg font-semibold text-primary">Evaluated Scenarios</h3>
            <span className="text-sm text-tertiary">
              - View each question, model response, and correct answer
            </span>
          </div>
          <ScenariosViewer runId={run.run_id} benchmarks={run.benchmarks || []} />
        </div>
      )}

      {/* Logs Tab */}
      {activeTab === 'logs' && (
        <div className="bg-surface rounded-xl border border-default overflow-hidden">
          <div className="px-4 py-3 border-b border-default flex items-center gap-2">
            <Terminal className="w-4 h-4 text-tertiary" />
            <h3 className="font-medium text-primary">Evaluation Logs</h3>
          </div>
          <div className="p-4 max-h-96 overflow-auto font-mono text-sm">
            {logs.length === 0 ? (
              <p className="text-tertiary">No logs available</p>
            ) : (
              logs.map((log, i) => (
                <div key={i} className="text-secondary leading-relaxed">
                  {log}
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Config Tab */}
      {activeTab === 'config' && (
        <div className="space-y-6">
          {/* Run Settings */}
          <div className="bg-surface rounded-xl border border-default p-6">
            <h3 className="text-lg font-semibold text-primary mb-4 flex items-center gap-2">
              <FileJson className="w-5 h-5 text-tertiary" />
              Run Configuration
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="p-4 bg-surface-active/30 rounded-lg">
                <div className="text-sm text-tertiary mb-1">Model</div>
                <div className="text-primary font-medium">{run.model}</div>
              </div>
              <div className="p-4 bg-surface-active/30 rounded-lg">
                <div className="text-sm text-tertiary mb-1">Provider</div>
                <div className="text-primary font-medium capitalize">{run.provider}</div>
              </div>
              <div className="p-4 bg-surface-active/30 rounded-lg">
                <div className="text-sm text-tertiary mb-1">Preset</div>
                <div className="text-primary font-medium capitalize">{run.preset || 'custom'}</div>
              </div>
              <div className="p-4 bg-surface-active/30 rounded-lg">
                <div className="text-sm text-tertiary mb-1">Sample Size</div>
                <div className="text-primary-400 font-bold text-xl">{run.sample_size || '?'}</div>
                <div className="text-xs text-tertiary">questions per benchmark</div>
              </div>
              <div className="p-4 bg-surface-active/30 rounded-lg">
                <div className="text-sm text-tertiary mb-1">Benchmarks</div>
                <div className="flex flex-wrap gap-1 mt-1">
                  {run.benchmarks?.map(b => (
                    <span key={b} className="px-2 py-0.5 bg-slate-600 rounded text-xs text-primary">
                      {formatBenchmarkName(b)}
                    </span>
                  ))}
                </div>
              </div>
              <div className="p-4 bg-surface-active/30 rounded-lg">
                <div className="text-sm text-tertiary mb-1">Duration</div>
                <div className="text-primary font-medium">
                  {run.duration_seconds
                    ? `${Math.floor(run.duration_seconds / 60)}m ${Math.round(run.duration_seconds % 60)}s`
                    : 'N/A'
                  }
                </div>
              </div>
              {totalInferenceTime && (
                <div className="p-4 bg-surface-active/30 rounded-lg">
                  <div className="text-sm text-tertiary mb-1">Inference Time</div>
                  <div className="text-primary font-medium">
                    {totalInferenceTime >= 60
                      ? `${Math.floor(totalInferenceTime / 60)}m ${Math.round(totalInferenceTime % 60)}s`
                      : `${totalInferenceTime.toFixed(1)}s`
                    }
                  </div>
                  <div className="text-xs text-tertiary mt-1">
                    (sum of benchmark times)
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Inference Settings */}
          {run.inference_settings && Object.keys(run.inference_settings).length > 0 && (
            <div className="bg-surface rounded-xl border border-default p-6">
              <h3 className="text-lg font-semibold text-primary mb-4">Inference Settings</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {run.inference_settings.temperature !== undefined && (
                  <div className="p-3 bg-surface-active/30 rounded-lg text-center">
                    <div className="text-xs text-tertiary mb-1">Temperature</div>
                    <div className="text-primary font-mono">{run.inference_settings.temperature}</div>
                  </div>
                )}
                {run.inference_settings.top_p !== undefined && (
                  <div className="p-3 bg-surface-active/30 rounded-lg text-center">
                    <div className="text-xs text-tertiary mb-1">Top P</div>
                    <div className="text-primary font-mono">{run.inference_settings.top_p}</div>
                  </div>
                )}
                {run.inference_settings.top_k !== undefined && (
                  <div className="p-3 bg-surface-active/30 rounded-lg text-center">
                    <div className="text-xs text-tertiary mb-1">Top K</div>
                    <div className="text-primary font-mono">{run.inference_settings.top_k}</div>
                  </div>
                )}
                {run.inference_settings.max_tokens !== undefined && (
                  <div className="p-3 bg-surface-active/30 rounded-lg text-center">
                    <div className="text-xs text-tertiary mb-1">Max Tokens</div>
                    <div className="text-primary font-mono">{run.inference_settings.max_tokens}</div>
                  </div>
                )}
                {run.inference_settings.seed !== undefined && (
                  <div className="p-3 bg-surface-active/30 rounded-lg text-center">
                    <div className="text-xs text-tertiary mb-1">Seed</div>
                    <div className="text-primary font-mono">{run.inference_settings.seed ?? 'random'}</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Raw JSON */}
          <div className="bg-surface rounded-xl border border-default overflow-hidden">
            <div className="px-4 py-3 border-b border-default flex items-center gap-2">
              <FileJson className="w-4 h-4 text-tertiary" />
              <h3 className="font-medium text-primary">Raw Configuration</h3>
            </div>
            <pre className="p-4 text-sm text-secondary overflow-auto max-h-64">
              {JSON.stringify({
                run_id: run.run_id,
                provider: run.provider,
                model: run.model,
                preset: run.preset,
                sample_size: run.sample_size,
                benchmarks: run.benchmarks,
                inference_settings: run.inference_settings,
                started_at: run.started_at,
                completed_at: run.completed_at,
                status: run.status,
              }, null, 2)}
            </pre>
          </div>
        </div>
      )}

      {/* System Tab */}
      {activeTab === 'system' && (
        <div className="space-y-6">
          <div className="bg-surface rounded-xl border border-default p-6">
            <h3 className="text-lg font-semibold text-primary mb-4 flex items-center gap-2">
              <Monitor className="w-5 h-5 text-tertiary" />
              System Information
            </h3>

            {run.system_info && Object.keys(run.system_info).length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Platform Info */}
                <div className="space-y-3">
                  <h4 className="text-sm font-medium text-tertiary uppercase tracking-wide">Platform</h4>
                  <div className="space-y-2">
                    {run.system_info.platform && (
                      <div className="flex justify-between py-2 border-b border-default">
                        <span className="text-tertiary">Operating System</span>
                        <span className="text-primary font-medium">{run.system_info.platform}</span>
                      </div>
                    )}
                    {run.system_info.platform_version && (
                      <div className="flex justify-between py-2 border-b border-default">
                        <span className="text-tertiary">OS Version</span>
                        <span className="text-primary font-medium text-sm">{run.system_info.platform_version}</span>
                      </div>
                    )}
                    {run.system_info.python_version && (
                      <div className="flex justify-between py-2 border-b border-default">
                        <span className="text-tertiary">Python Version</span>
                        <span className="text-primary font-medium">{run.system_info.python_version}</span>
                      </div>
                    )}
                    {run.system_info.machine && (
                      <div className="flex justify-between py-2 border-b border-default">
                        <span className="text-tertiary">Architecture</span>
                        <span className="text-primary font-medium">{run.system_info.machine}</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Hardware Info */}
                <div className="space-y-3">
                  <h4 className="text-sm font-medium text-tertiary uppercase tracking-wide flex items-center gap-2">
                    <Cpu className="w-4 h-4" />
                    Hardware
                  </h4>
                  <div className="space-y-2">
                    {run.system_info.processor && (
                      <div className="flex justify-between py-2 border-b border-default">
                        <span className="text-tertiary">Processor</span>
                        <span className="text-primary font-medium text-sm">{run.system_info.processor}</span>
                      </div>
                    )}
                    {run.system_info.cpu_count && (
                      <div className="flex justify-between py-2 border-b border-default">
                        <span className="text-tertiary">CPU Cores</span>
                        <span className="text-primary font-medium">
                          {run.system_info.cpu_count} physical
                          {run.system_info.cpu_count_logical && ` / ${run.system_info.cpu_count_logical} logical`}
                        </span>
                      </div>
                    )}
                    {run.system_info.ram_total_gb && (
                      <div className="flex justify-between py-2 border-b border-default">
                        <span className="text-tertiary">RAM</span>
                        <span className="text-primary font-medium">{run.system_info.ram_total_gb} GB total</span>
                      </div>
                    )}
                    {run.system_info.gpu && (
                      <div className="flex justify-between py-2 border-b border-default">
                        <span className="text-tertiary">GPU</span>
                        <span className="text-primary font-medium text-sm">{run.system_info.gpu}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <HardDrive className="w-12 h-12 text-secondary mx-auto mb-3" />
                <p className="text-tertiary">No system information available for this run</p>
                <p className="text-tertiary text-sm mt-1">System info is captured for new runs only</p>
              </div>
            )}
          </div>

          {/* Raw System Info */}
          {run.system_info && Object.keys(run.system_info).length > 0 && (
            <div className="bg-surface rounded-xl border border-default overflow-hidden">
              <div className="px-4 py-3 border-b border-default flex items-center gap-2">
                <FileJson className="w-4 h-4 text-tertiary" />
                <h3 className="font-medium text-primary">Raw System Data</h3>
              </div>
              <pre className="p-4 text-sm text-secondary overflow-auto">
                {JSON.stringify(run.system_info, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default RunDetail

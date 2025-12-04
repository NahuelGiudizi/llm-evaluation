import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  History,
  Loader2,
  CheckCircle,
  XCircle,
  Clock,
  RefreshCw,
  Trash2,
  Hash,
  Eye,
  Square,
  CheckSquare,
  Minus,
  Copy,
  Check,
  FileJson
} from 'lucide-react'
import { fetchRuns, deleteRun, deleteRuns } from '../api'
import DeleteConfirmModal from './DeleteConfirmModal'

// Helper to extract score from different benchmark result formats
function extractScore(benchmarkName, data) {
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
  if (data.correct !== undefined && data.questions_tested !== undefined && data.questions_tested > 0) {
    return data.correct / data.questions_tested
  }
  if (data.correct !== undefined && data.scenarios_tested !== undefined && data.scenarios_tested > 0) {
    return data.correct / data.scenarios_tested
  }
  return null
}

// Format benchmark name
function formatBenchmarkName(name) {
  if (!name) return ''
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
    'do-not-answer': 'Do-Not-Answer',
  }
  const normalized = name.toLowerCase().replace(/[-_\s]/g, '')
  return nameMap[normalized] || name
}

// Format array of benchmark names
function formatBenchmarks(benchmarks) {
  if (!benchmarks || !Array.isArray(benchmarks)) return 'N/A'
  return benchmarks.map(formatBenchmarkName).join(', ')
}

function RunHistory() {
  const navigate = useNavigate()
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all')

  // Selection state
  const [selectedRuns, setSelectedRuns] = useState(new Set())

  // Delete modal state
  const [deleteModalOpen, setDeleteModalOpen] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState(null) // null = bulk, string = single runId
  const [isDeleting, setIsDeleting] = useState(false)

  // Copy feedback state
  const [copiedPath, setCopiedPath] = useState(null)

  useEffect(() => {
    loadRuns()
  }, [])

  async function loadRuns() {
    setLoading(true)
    try {
      const data = await fetchRuns()
      setRuns(data)
      // Clear selection when reloading
      setSelectedRuns(new Set())
    } catch (err) {
      console.error('Failed to load runs:', err)
    } finally {
      setLoading(false)
    }
  }

  const filteredRuns = runs.filter(run => {
    if (filter === 'all') return true
    return run.status === filter
  })

  // Selection handlers
  function toggleSelectRun(runId, event) {
    event.stopPropagation()
    const newSelected = new Set(selectedRuns)
    if (newSelected.has(runId)) {
      newSelected.delete(runId)
    } else {
      newSelected.add(runId)
    }
    setSelectedRuns(newSelected)
  }

  function toggleSelectAll() {
    if (selectedRuns.size === filteredRuns.length) {
      setSelectedRuns(new Set())
    } else {
      setSelectedRuns(new Set(filteredRuns.map(r => r.run_id)))
    }
  }

  // Delete handlers
  function handleDeleteClick(runId, event) {
    event.stopPropagation()
    setDeleteTarget(runId)
    setDeleteModalOpen(true)
  }

  function handleBulkDeleteClick() {
    setDeleteTarget(null)
    setDeleteModalOpen(true)
  }

  async function handleConfirmDelete() {
    setIsDeleting(true)
    try {
      if (deleteTarget) {
        // Single delete
        await deleteRun(deleteTarget)
      } else {
        // Bulk delete
        await deleteRuns([...selectedRuns])
      }

      // Reload runs
      await loadRuns()
      setDeleteModalOpen(false)
      setDeleteTarget(null)
    } catch (err) {
      console.error('Failed to delete:', err)
    } finally {
      setIsDeleting(false)
    }
  }

  // Checkbox state helpers
  const allSelected = filteredRuns.length > 0 && selectedRuns.size === filteredRuns.length
  const someSelected = selectedRuns.size > 0 && selectedRuns.size < filteredRuns.length
  const deleteCount = deleteTarget ? 1 : selectedRuns.size

  // Copy file path to clipboard
  async function copyFilePath(filePath, runId, event) {
    event.stopPropagation()
    if (!filePath) return

    try {
      await navigator.clipboard.writeText(filePath)
      setCopiedPath(runId)
      setTimeout(() => setCopiedPath(null), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  function getStatusIcon(status) {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-400" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-400" />
      case 'running':
        return <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />
      default:
        return <Clock className="w-5 h-5 text-slate-400" />
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Run History</h1>
          <p className="text-slate-400">View all past evaluation runs</p>
        </div>
        <div className="flex items-center gap-3">
          {selectedRuns.size > 0 && (
            <button
              onClick={handleBulkDeleteClick}
              className="px-4 py-2 rounded-lg bg-red-600/20 text-red-400 hover:bg-red-600/30 border border-red-600/50 flex items-center gap-2 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
              Delete Selected ({selectedRuns.size})
            </button>
          )}
          <button
            onClick={loadRuns}
            className="px-4 py-2 rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-800 flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="mb-6 flex gap-2">
        {['all', 'completed', 'running', 'failed'].map(status => (
          <button
            key={status}
            onClick={() => setFilter(status)}
            className={`px-4 py-2 rounded-lg text-sm capitalize transition-colors ${filter === status
              ? 'bg-primary-500 text-white'
              : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
          >
            {status}
          </button>
        ))}
      </div>

      {filteredRuns.length === 0 ? (
        <div className="text-center py-20">
          <History className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-white mb-2">No Runs Found</h2>
          <p className="text-slate-400">
            {filter === 'all'
              ? 'Start your first evaluation to see it here'
              : `No ${filter} runs found`}
          </p>
        </div>
      ) : (
        <>
          {/* Selection Header */}
          <div className="mb-4 flex items-center gap-4 px-2">
            <button
              onClick={toggleSelectAll}
              className="text-slate-400 hover:text-white transition-colors flex items-center gap-2"
            >
              {allSelected ? (
                <CheckSquare className="w-5 h-5 text-primary-400" />
              ) : someSelected ? (
                <Minus className="w-5 h-5 text-primary-400" />
              ) : (
                <Square className="w-5 h-5" />
              )}
              <span className="text-sm">
                {selectedRuns.size > 0
                  ? `${selectedRuns.size} selected`
                  : 'Select all'}
              </span>
            </button>
            <span className="text-sm text-slate-500">
              {filteredRuns.length} runs
            </span>
          </div>

          {/* Runs List - Cleaner Card Design */}
          <div className="space-y-3">
            {filteredRuns.map((run) => {
              // Calculate average score correctly
              const scores = run.results
                ? Object.entries(run.results)
                  .filter(([name]) => name.toLowerCase() !== 'average')
                  .map(([name, data]) => extractScore(name, data))
                  .filter(s => s !== null)
                : []
              const avgScore = scores.length > 0
                ? scores.reduce((sum, s) => sum + s, 0) / scores.length
                : null

              const isSelected = selectedRuns.has(run.run_id)
              const runIdShort = run.run_id?.slice(-6) || ''

              return (
                <div
                  key={run.run_id}
                  className={`bg-slate-800 rounded-xl border transition-all ${isSelected
                      ? 'border-primary-500/50 bg-slate-800/80'
                      : 'border-slate-700 hover:border-slate-600'
                    }`}
                >
                  {/* Main Row */}
                  <div className="p-4 flex items-center gap-4">
                    {/* Checkbox */}
                    <button
                      onClick={(e) => toggleSelectRun(run.run_id, e)}
                      className="text-slate-400 hover:text-white transition-colors flex-shrink-0"
                    >
                      {isSelected ? (
                        <CheckSquare className="w-5 h-5 text-primary-400" />
                      ) : (
                        <Square className="w-5 h-5" />
                      )}
                    </button>

                    {getStatusIcon(run.status)}

                    {/* Model & Meta Info */}
                    <div
                      className="flex-1 min-w-0 cursor-pointer"
                      onClick={() => navigate(`/run/${run.run_id}`)}
                    >
                      <div className="flex items-center gap-2 flex-wrap">
                        <h3 className="font-semibold text-white">{run.model}</h3>
                        <span className="text-xs text-slate-400 bg-slate-700 px-2 py-0.5 rounded">
                          {run.provider}
                        </span>
                        <span className="text-xs text-slate-500 font-mono">
                          #{runIdShort}
                        </span>
                      </div>
                      <div className="text-sm text-slate-400 mt-1">
                        {new Date(run.started_at).toLocaleString()}
                      </div>
                    </div>

                    {/* Sample Size & Preset */}
                    <div className="text-right flex-shrink-0 hidden sm:block">
                      <div className="text-sm text-primary-400">
                        {run.sample_size || '?'} samples
                      </div>
                      {run.preset && (
                        <div className="text-xs text-slate-500 capitalize">
                          ({run.preset})
                        </div>
                      )}
                    </div>

                    {/* Average Score */}
                    {avgScore !== null && (
                      <div
                        className="text-right cursor-pointer flex-shrink-0"
                        onClick={() => navigate(`/run/${run.run_id}`)}
                      >
                        <div className={`text-2xl font-bold ${avgScore >= 0.7 ? 'text-green-400' :
                            avgScore >= 0.4 ? 'text-yellow-400' : 'text-red-400'
                          }`}>
                          {(avgScore * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-slate-500">avg score</div>
                      </div>
                    )}

                    {/* Actions */}
                    <div className="flex items-center gap-1 flex-shrink-0">
                      <button
                        onClick={() => navigate(`/run/${run.run_id}`)}
                        className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
                        title="View details"
                      >
                        <Eye className="w-4 h-4" />
                      </button>
                      <button
                        onClick={(e) => handleDeleteClick(run.run_id, e)}
                        className="p-2 rounded-lg text-slate-400 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                        title="Delete run"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  {/* Benchmark Scores - Compact Grid */}
                  {run.results && Object.keys(run.results).length > 0 && (
                    <div
                      className="px-4 pb-4 pt-0 cursor-pointer"
                      onClick={() => navigate(`/run/${run.run_id}`)}
                    >
                      <div className="flex gap-2 flex-wrap">
                        {Object.entries(run.results)
                          .filter(([name]) => name.toLowerCase() !== 'average')
                          .map(([name, data]) => {
                            const score = extractScore(name, data)
                            const scoreColor = score === null ? 'text-slate-500' :
                              score >= 0.7 ? 'text-green-400' :
                                score >= 0.4 ? 'text-yellow-400' : 'text-red-400'
                            return (
                              <span
                                key={name}
                                className="inline-flex items-center gap-1 text-xs bg-slate-700/50 px-2 py-1 rounded"
                              >
                                <span className="text-slate-400">{formatBenchmarkName(name)}:</span>
                                <span className={`font-medium ${scoreColor}`}>
                                  {score !== null ? `${(score * 100).toFixed(0)}%` : 'N/A'}
                                </span>
                              </span>
                            )
                          })}
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </>
      )}

      {/* Delete Confirmation Modal */}
      <DeleteConfirmModal
        isOpen={deleteModalOpen}
        onClose={() => {
          setDeleteModalOpen(false)
          setDeleteTarget(null)
        }}
        onConfirm={handleConfirmDelete}
        count={deleteCount}
        isDeleting={isDeleting}
      />
    </div>
  )
}

export default RunHistory

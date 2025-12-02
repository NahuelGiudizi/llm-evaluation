import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Activity,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  AlertCircle,
  Terminal,
  RefreshCw,
  Square,
  Pause,
  ListOrdered
} from 'lucide-react'
import { fetchRun, cancelRun, fetchQueueStatus, cancelQueue } from '../api'

function ProgressViewer({ activeRun }) {
  const navigate = useNavigate()
  const [run, setRun] = useState(null)
  const [status, setStatus] = useState('connecting')
  const [error, setError] = useState(null)
  const [cancelling, setCancelling] = useState(false)
  const [queueStatus, setQueueStatus] = useState(null)
  const [cancellingQueue, setCancellingQueue] = useState(false)
  const [currentQueueRun, setCurrentQueueRun] = useState(null)
  const [showQueue, setShowQueue] = useState(false)
  const logsEndRef = useRef(null)
  const queueLogsEndRef = useRef(null)
  const pollIntervalRef = useRef(null)

  async function handleCancel() {
    if (!activeRun || cancelling) return

    setCancelling(true)
    try {
      await cancelRun(activeRun)
      setStatus('cancelled')
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    } catch (err) {
      setError('Failed to cancel run: ' + err.message)
    } finally {
      setCancelling(false)
    }
  }

  async function handleCancelQueue() {
    if (cancellingQueue) return
    setCancellingQueue(true)
    try {
      await cancelQueue()
      setQueueStatus(prev => prev ? { ...prev, status: 'cancelled' } : null)
    } catch (err) {
      console.error('Failed to cancel queue:', err)
    } finally {
      setCancellingQueue(false)
    }
  }

  // Check for active queue - ALWAYS poll, prioritize queue over stale activeRun
  useEffect(() => {
    let isMounted = true

    async function checkQueue() {
      if (!isMounted) return
      try {
        const status = await fetchQueueStatus()
        if (!isMounted) return
        if (status && (status.status === 'running' || status.status === 'pending')) {
          setQueueStatus(status)
          setShowQueue(true)
        } else {
          setQueueStatus(null)
          setShowQueue(false)
        }
      } catch (err) {
        // No active queue
        if (isMounted) {
          setQueueStatus(null)
          setShowQueue(false)
        }
      }
    }

    checkQueue()
    // Poll every 2 seconds
    const interval = setInterval(checkQueue, 2000)
    return () => {
      isMounted = false
      clearInterval(interval)
    }
  }, []) // No dependencies - always poll

  // Fetch logs for current running queue item
  useEffect(() => {
    const currentItem = queueStatus?.items?.find(i => i.status === 'running')
    const runId = currentItem?.run_id

    if (!runId) {
      setCurrentQueueRun(null)
      return
    }

    let isMounted = true

    async function fetchCurrentRun() {
      try {
        const data = await fetchRun(runId)
        if (isMounted) {
          setCurrentQueueRun(data)
        }
      } catch (err) {
        console.error('Failed to fetch current queue run:', err)
      }
    }

    fetchCurrentRun()
    const interval = setInterval(fetchCurrentRun, 2000)

    return () => {
      isMounted = false
      clearInterval(interval)
    }
  }, [queueStatus?.items])

  // Auto-scroll queue logs
  useEffect(() => {
    queueLogsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [currentQueueRun?.logs])

  useEffect(() => {
    if (!activeRun) {
      setStatus('no-run')
      return
    }

    setStatus('connecting')
    setRun(null)

    // Poll for updates every 2 seconds
    const pollRun = async () => {
      try {
        const data = await fetchRun(activeRun)
        setRun(data)

        if (data.status === 'running') {
          setStatus('running')
        } else if (data.status === 'completed') {
          setStatus('complete')
          // Stop polling when complete
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current)
          }
        } else if (data.status === 'failed') {
          setStatus('error')
          setError('Evaluation failed')
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current)
          }
        } else if (data.status === 'cancelled') {
          setStatus('cancelled')
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current)
          }
        } else {
          setStatus('running')
        }
      } catch (err) {
        console.error('Failed to fetch run:', err)
        setError('Failed to fetch run status')
      }
    }

    // Initial fetch
    pollRun()

    // Start polling
    pollIntervalRef.current = setInterval(pollRun, 2000)

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [activeRun])

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [run?.logs])

  // Extract logs from run data
  const logs = run?.logs?.map(l => typeof l === 'object' ? l.message : l) || []

  // Normalize benchmark names for comparison
  const normalizeBenchmark = (name) => {
    if (!name) return ''
    return name.toLowerCase().replace(/[-_\s]/g, '')
  }

  // Format benchmark name to proper case
  const formatBenchmarkName = (name) => {
    if (!name) return ''
    const benchmarkNames = {
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
    return benchmarkNames[normalized] || name
  }

  // Format array of benchmark names
  const formatBenchmarks = (benchmarks) => {
    if (!benchmarks || !Array.isArray(benchmarks)) return ''
    return benchmarks.map(formatBenchmarkName).join(', ')
  }

  // Calculate progress from logs
  const getProgressFromLogs = () => {
    if (!logs.length) return { percent: 0, benchmark: null, accuracy: null }

    const benchmarkPattern = /(MMLU|TruthfulQA|HellaSwag|ARC|WinoGrande|CommonsenseQA|BoolQ|SafetyBench|Do-Not-Answer|DONOTANSWER)/i

    // First, find the CURRENT benchmark by looking for "Running X..." from the end
    let currentBenchmark = null
    for (let i = logs.length - 1; i >= 0; i--) {
      const line = logs[i]
      if (line && (line.includes('Running ') || line.includes('Starting '))) {
        const match = line.match(benchmarkPattern)
        if (match) {
          currentBenchmark = match[1]
          break
        }
      }
    }

    // Now find the last progress line for this benchmark (or any if no current found)
    for (let i = logs.length - 1; i >= 0; i--) {
      const line = logs[i]
      if (line && line.includes('Progress:')) {
        const percentMatch = line.match(/(\d+)%/)
        const accMatch = line.match(/Acc:.*?(\d+\.?\d*)%/)
        const benchmarkMatch = line.match(benchmarkPattern)
        const logBenchmark = benchmarkMatch ? benchmarkMatch[1] : null

        // If we found a current benchmark from "Running X", check if this progress is for it
        if (currentBenchmark && logBenchmark &&
          normalizeBenchmark(logBenchmark) !== normalizeBenchmark(currentBenchmark)) {
          // This progress is for a previous benchmark, skip it
          continue
        }

        return {
          percent: percentMatch ? parseInt(percentMatch[1]) : 0,
          benchmark: currentBenchmark || logBenchmark,
          accuracy: accMatch ? parseFloat(accMatch[1]) : null
        }
      }
    }

    // If we found a current benchmark but no progress for it yet
    if (currentBenchmark) {
      return { percent: 0, benchmark: currentBenchmark, accuracy: null }
    }

    return { percent: 0, benchmark: null, accuracy: null }
  }

  const progressInfo = getProgressFromLogs()

  // Helper to get status icon for queue items
  function getQueueItemIcon(itemStatus) {
    switch (itemStatus) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-400" />
      case 'running':
        return <Loader2 className="w-4 h-4 text-primary-400 animate-spin" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-400" />
      case 'pending':
      default:
        return <Clock className="w-4 h-4 text-slate-400" />
    }
  }

  // If there's a queue running, ALWAYS show it (priority over stale activeRun)
  if (showQueue && queueStatus && (queueStatus.status === 'running' || queueStatus.status === 'pending')) {
    const completed = queueStatus.items?.filter(i => i.status === 'completed').length || 0
    const currentItem = queueStatus.items?.find(i => i.status === 'running')

    return (
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Queue Progress</h1>
          <p className="text-slate-400">Running {completed}/{queueStatus.total} evaluations</p>
        </div>

        {/* Queue Status Card */}
        <div className="mb-6 p-6 rounded-xl border bg-primary-500/10 border-primary-500/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-full bg-primary-500/20 flex items-center justify-center">
                <ListOrdered className="w-6 h-6 text-primary-400" />
              </div>
              <div>
                <h3 className="font-semibold text-white flex items-center gap-2">
                  <Loader2 className="w-4 h-4 text-primary-400 animate-spin" />
                  Queue Running
                </h3>
                {currentItem && (
                  <p className="text-sm text-slate-400">
                    Currently: {currentItem.model} ({currentItem.provider}) - {formatBenchmarks(currentItem.benchmarks)}
                  </p>
                )}
              </div>
            </div>
            <button
              onClick={handleCancelQueue}
              disabled={cancellingQueue}
              className="px-4 py-2 rounded-lg bg-red-600/20 text-red-400 hover:bg-red-600/30 border border-red-600/50 flex items-center gap-2 transition-colors disabled:opacity-50 whitespace-nowrap"
            >
              {cancellingQueue ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Stopping...</span>
                </>
              ) : (
                <>
                  <Square className="w-4 h-4" />
                  <span>Cancel All</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary-500 to-purple-500 transition-all duration-500"
              style={{ width: `${(completed / queueStatus.total) * 100}%` }}
            />
          </div>
        </div>

        {/* Queue Items */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-700">
            <h3 className="font-medium text-white flex items-center gap-2">
              <ListOrdered className="w-4 h-4 text-slate-400" />
              Queue Items
            </h3>
          </div>
          <div className="divide-y divide-slate-700/50">
            {queueStatus.items?.map((item, i) => (
              <div
                key={i}
                className={`p-4 flex items-center gap-4 ${item.status === 'running' ? 'bg-primary-500/5' : ''
                  }`}
              >
                {getQueueItemIcon(item.status)}
                <div className="flex-1">
                  <div className="font-medium text-white">
                    {item.model}
                    <span className="text-slate-400 font-normal ml-2">({item.provider})</span>
                  </div>
                  <div className="text-sm text-slate-400">
                    {formatBenchmarks(item.benchmarks)} • {item.sample_size} samples
                  </div>
                </div>
                {item.score !== null && item.score !== undefined && (
                  <div className="text-lg font-bold text-green-400">
                    {(item.score * 100).toFixed(1)}%
                  </div>
                )}
                {item.duration_seconds && (
                  <div className="text-sm text-slate-400">
                    {Math.floor(item.duration_seconds / 60)}m {Math.floor(item.duration_seconds % 60)}s
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {queueStatus.eta_seconds && (
          <div className="mt-4 text-sm text-slate-400 flex items-center gap-2">
            <Clock className="w-4 h-4" />
            ETA: ~{Math.ceil(queueStatus.eta_seconds / 60)} minutes remaining
          </div>
        )}

        {/* Live Logs for current running item */}
        {currentQueueRun && (
          <div className="mt-6 bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-700 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-slate-400" />
                <h3 className="font-medium text-white">Live Logs</h3>
                <span className="text-xs text-slate-500">
                  ({currentQueueRun.logs?.length || 0} entries)
                </span>
              </div>
              <span className="text-xs text-primary-400 animate-pulse">● Live</span>
            </div>
            <div className="p-4 h-64 overflow-auto font-mono text-xs">
              {(!currentQueueRun.logs || currentQueueRun.logs.length === 0) ? (
                <p className="text-slate-500">Waiting for logs...</p>
              ) : (
                currentQueueRun.logs.slice(-100).map((log, i) => {
                  const message = typeof log === 'object' ? log.message : log
                  return (
                    <div
                      key={i}
                      className={`leading-relaxed py-0.5 ${message?.includes('Progress:') ? 'text-primary-300' :
                        message?.includes('Accuracy:') || message?.includes('Score:') ? 'text-green-300' :
                          message?.includes('Error') || message?.includes('failed') ? 'text-red-300' :
                            'text-slate-400'
                        }`}
                    >
                      {message || ''}
                    </div>
                  )
                })
              )}
              <div ref={queueLogsEndRef} />
            </div>
          </div>
        )}
      </div>
    )
  }

  // Show "No Active Run" if there's no activeRun OR if the activeRun is already finished
  const isRunFinished = run && ['completed', 'failed', 'cancelled'].includes(run.status)
  
  if (!activeRun || (isRunFinished && !showQueue)) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="text-center py-20">
          <Activity className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-white mb-2">No Active Run</h2>
          <p className="text-slate-400 mb-6">
            {isRunFinished 
              ? 'The last evaluation has finished. Start a new one or view results in History.'
              : 'Start a new evaluation to see progress here'
            }
          </p>
          <div className="flex gap-4 justify-center">
            <button
              onClick={() => navigate('/')}
              className="px-6 py-3 rounded-lg bg-primary-500 text-white hover:bg-primary-600 transition-colors"
            >
              Start New Run
            </button>
            {isRunFinished && (
              <button
                onClick={() => navigate(`/run/${activeRun}`)}
                className="px-6 py-3 rounded-lg bg-slate-700 text-white hover:bg-slate-600 transition-colors"
              >
                View Last Results
              </button>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Evaluation Progress</h1>
            <p className="text-slate-400">Run ID: {activeRun}</p>
          </div>
          {status === 'running' && (
            <div className="flex items-center gap-2 text-slate-400">
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span className="text-sm">Auto-updating...</span>
            </div>
          )}
        </div>
      </div>

      {/* Status Card */}
      <div className={`mb-6 p-6 rounded-xl border ${status === 'running' ? 'bg-primary-500/10 border-primary-500/30' :
        status === 'complete' ? 'bg-green-500/10 border-green-500/30' :
          status === 'error' ? 'bg-red-500/10 border-red-500/30' :
            status === 'cancelled' ? 'bg-slate-700/50 border-slate-600' :
              'bg-slate-800 border-slate-700'
        }`}>
        <div className="flex items-center gap-4">
          {status === 'connecting' && (
            <>
              <Loader2 className="w-8 h-8 text-slate-400 animate-spin" />
              <div>
                <h3 className="font-semibold text-white">Connecting...</h3>
                <p className="text-sm text-slate-400">Fetching evaluation status...</p>
              </div>
            </>
          )}
          {status === 'running' && (
            <>
              <div className="w-8 h-8 rounded-full bg-primary-500/20 flex items-center justify-center pulse-glow">
                <Activity className="w-5 h-5 text-primary-400" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-white">Running - {run?.model || 'Loading...'}</h3>
                <p className="text-sm text-slate-400">
                  {progressInfo.benchmark ? `Running ${progressInfo.benchmark}...` : 'Initializing...'}
                  {progressInfo.accuracy !== null && ` (Current Accuracy: ${progressInfo.accuracy}%)`}
                </p>
              </div>
              <div className="text-right mr-4">
                <div className="text-2xl font-bold text-primary-400">{progressInfo.percent}%</div>
                <div className="text-sm text-slate-400">
                  {logs.length} log entries
                </div>
              </div>
              <button
                onClick={handleCancel}
                disabled={cancelling}
                className="px-4 py-2 rounded-lg bg-slate-700 border border-slate-600 text-slate-300 hover:bg-slate-600 hover:text-white transition-colors flex items-center gap-2 disabled:opacity-50"
              >
                {cancelling ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Stopping...
                  </>
                ) : (
                  <>
                    <Square className="w-4 h-4 fill-current" />
                    Stop
                  </>
                )}
              </button>
            </>
          )}
          {status === 'complete' && (
            <>
              <CheckCircle className="w-8 h-8 text-green-400" />
              <div className="flex-1">
                <h3 className="font-semibold text-white">Complete - {run?.model}</h3>
                <p className="text-sm text-slate-400">Evaluation finished successfully</p>
              </div>
              <button
                onClick={() => navigate(`/run/${activeRun}`)}
                className="px-4 py-2 rounded-lg bg-green-500 text-white hover:bg-green-600 transition-colors"
              >
                View Results
              </button>
            </>
          )}
          {status === 'cancelled' && (
            <>
              <Square className="w-8 h-8 text-slate-400" />
              <div className="flex-1">
                <h3 className="font-semibold text-white">Stopped - {run?.model}</h3>
                <p className="text-sm text-slate-400">Evaluation was stopped</p>
              </div>
              <button
                onClick={() => navigate('/')}
                className="px-4 py-2 rounded-lg bg-slate-600 text-white hover:bg-slate-500 transition-colors"
              >
                Start New Run
              </button>
            </>
          )}
          {status === 'error' && (
            <>
              <XCircle className="w-8 h-8 text-red-400" />
              <div className="flex-1">
                <h3 className="font-semibold text-white">Error</h3>
                <p className="text-sm text-red-400">{error}</p>
              </div>
              <button
                onClick={() => navigate(`/run/${activeRun}`)}
                className="px-4 py-2 rounded-lg bg-slate-600 text-white hover:bg-slate-500 transition-colors"
              >
                View Details
              </button>
            </>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      {(status === 'running' || status === 'complete') && (
        <div className="mb-6">
          <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary-500 to-purple-500 transition-all duration-500"
              style={{ width: `${status === 'complete' ? 100 : progressInfo.percent}%` }}
            />
          </div>
        </div>
      )}

      {/* Benchmarks info */}
      {run?.benchmarks && (
        <div className="mb-6 flex gap-2 flex-wrap">
          {run.benchmarks.map(b => (
            <span
              key={b}
              className={`px-3 py-1 rounded-lg text-sm ${normalizeBenchmark(progressInfo.benchmark) === normalizeBenchmark(b)
                ? 'bg-primary-500/30 text-primary-300 border border-primary-500/50'
                : 'bg-slate-700 text-slate-300'
                }`}
            >
              {b.toUpperCase()}
            </span>
          ))}
        </div>
      )}

      {/* Logs */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-slate-400" />
            <h3 className="font-medium text-white">Logs</h3>
            <span className="text-xs text-slate-500">({logs.length} entries)</span>
          </div>
          {status === 'running' && (
            <span className="text-xs text-primary-400 animate-pulse">● Live</span>
          )}
        </div>
        <div className="p-4 h-80 overflow-auto font-mono text-xs">
          {logs.length === 0 ? (
            <p className="text-slate-500">Waiting for logs...</p>
          ) : (
            logs.slice(-100).map((log, i) => (
              <div
                key={i}
                className={`leading-relaxed py-0.5 ${log?.includes('Progress:') ? 'text-primary-300' :
                  log?.includes('Accuracy:') || log?.includes('Score:') ? 'text-green-300' :
                    log?.includes('Error') || log?.includes('failed') ? 'text-red-300' :
                      'text-slate-400'
                  }`}
              >
                {log || ''}
              </div>
            ))
          )}
          <div ref={logsEndRef} />
        </div>
      </div>
    </div>
  )
}

export default ProgressViewer

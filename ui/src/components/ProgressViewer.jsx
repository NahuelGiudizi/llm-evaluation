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
  const logsContainerRef = useRef(null)
  const queueLogsContainerRef = useRef(null)
  const pollIntervalRef = useRef(null)
  const [autoScrollLogs, setAutoScrollLogs] = useState(true)
  const [autoScrollQueueLogs, setAutoScrollQueueLogs] = useState(true)

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

  // Auto-scroll queue logs only if user is near bottom
  useEffect(() => {
    if (autoScrollQueueLogs && queueLogsEndRef.current) {
      // Use a small delay to avoid jarring scroll during user interaction
      const timeoutId = setTimeout(() => {
        if (autoScrollQueueLogs) { // Check again after delay
          queueLogsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
        }
      }, 100)
      return () => clearTimeout(timeoutId)
    }
  }, [currentQueueRun?.logs, autoScrollQueueLogs])

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
    if (autoScrollLogs && logsEndRef.current) {
      // Use a small delay to avoid jarring scroll during user interaction
      const timeoutId = setTimeout(() => {
        if (autoScrollLogs) { // Check again after delay
          logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
        }
      }, 100)
      return () => clearTimeout(timeoutId)
    }
  }, [run?.logs, autoScrollLogs])

  // Detect manual scroll and disable auto-scroll if user scrolls up
  useEffect(() => {
    const handleLogsScroll = () => {
      if (!logsContainerRef.current) return
      const container = logsContainerRef.current
      // Check if user is near the bottom (within 150px)
      const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 150
      // Only update if state actually changed to avoid unnecessary re-renders
      setAutoScrollLogs(prev => {
        if (prev !== isNearBottom) {
          return isNearBottom
        }
        return prev
      })
    }

    const handleQueueLogsScroll = () => {
      if (!queueLogsContainerRef.current) return
      const container = queueLogsContainerRef.current
      // Check if user is near the bottom (within 150px)
      const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 150
      // Only update if state actually changed
      setAutoScrollQueueLogs(prev => {
        if (prev !== isNearBottom) {
          return isNearBottom
        }
        return prev
      })
    }

    const logsContainer = logsContainerRef.current
    const queueLogsContainer = queueLogsContainerRef.current

    if (logsContainer) {
      logsContainer.addEventListener('scroll', handleLogsScroll, { passive: true })
    }
    if (queueLogsContainer) {
      queueLogsContainer.addEventListener('scroll', handleQueueLogsScroll, { passive: true })
    }

    return () => {
      if (logsContainer) {
        logsContainer.removeEventListener('scroll', handleLogsScroll)
      }
      if (queueLogsContainer) {
        queueLogsContainer.removeEventListener('scroll', handleQueueLogsScroll)
      }
    }
  }, [])

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
        return <Clock className="w-4 h-4 text-tertiary" />
    }
  }

  // Helper to format inference settings compactly
  const formatInferenceSettings = (settings) => {
    if (!settings) return null
    return `T=${settings.temperature ?? 0}, P=${settings.top_p ?? 1}, K=${settings.top_k ?? -1}, Seed=${settings.seed ?? 42}`
  }

  // If there's a queue running, ALWAYS show it (priority over stale activeRun)
  if (showQueue && queueStatus && (queueStatus.status === 'running' || queueStatus.status === 'pending')) {
    const completed = queueStatus.items?.filter(i => i.status === 'completed').length || 0
    const currentItem = queueStatus.items?.find(i => i.status === 'running')

    // Calculate total progress based on samples
    const totalSamples = queueStatus.items?.reduce((sum, item) => sum + (item.sample_size || 100), 0) || 0
    const completedSamples = queueStatus.items?.reduce((sum, item) => {
      if (item.status === 'completed') return sum + (item.sample_size || 100)
      return sum
    }, 0) || 0

    // Get current run progress from logs
    const currentRunLogs = currentQueueRun?.logs || []
    let currentRunProgress = 0
    for (let i = currentRunLogs.length - 1; i >= 0; i--) {
      const line = typeof currentRunLogs[i] === 'object' ? currentRunLogs[i].message : currentRunLogs[i]
      if (line && line.includes('Progress:')) {
        const match = line.match(/(\d+)\/(\d+)/)
        if (match) {
          currentRunProgress = parseInt(match[1])
          break
        }
      }
    }

    const totalProgress = totalSamples > 0
      ? ((completedSamples + currentRunProgress) / totalSamples) * 100
      : 0

    return (
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-primary mb-2">Queue Progress</h1>
          <p className="text-tertiary">Running {completed}/{queueStatus.total} evaluations • {completedSamples + currentRunProgress}/{totalSamples} samples</p>
        </div>

        {/* Queue Status Card */}
        <div className="mb-6 p-6 rounded-xl border bg-primary-500/10 border-primary-500/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-full bg-primary-500/20 flex items-center justify-center">
                <ListOrdered className="w-6 h-6 text-primary-400" />
              </div>
              <div>
                <h3 className="font-semibold text-primary flex items-center gap-2">
                  <Loader2 className="w-4 h-4 text-primary-400 animate-spin" />
                  Queue Running
                </h3>
                {currentItem && (
                  <p className="text-sm text-tertiary">
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

        {/* Progress Bar - Total samples progress */}
        <div className="mb-6">
          <div className="flex justify-between text-xs text-tertiary mb-1">
            <span>{Math.round(totalProgress)}% complete</span>
            <span>{completedSamples + currentRunProgress} / {totalSamples} samples</span>
          </div>
          <div className="h-4 bg-surface-active rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary-500 to-purple-500 transition-all duration-500"
              style={{ width: `${totalProgress}%` }}
            />
          </div>
        </div>

        {/* Current Running Job Details */}
        {currentQueueRun && currentItem && (
          <div className="mb-6 p-4 rounded-xl border bg-surface border-default">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-8 h-8 rounded-full bg-primary-500/20 flex items-center justify-center pulse-glow">
                <Activity className="w-5 h-5 text-primary-400" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-primary">
                  Currently Running: {currentItem.model}
                  <span className="text-tertiary font-normal ml-2">({currentItem.provider})</span>
                </h3>
                <p className="text-sm text-tertiary">
                  {formatBenchmarks(currentItem.benchmarks)}
                </p>
              </div>
            </div>
            
            {/* Extract progress from current run logs */}
            {(() => {
              const logs = currentQueueRun.logs?.map(l => typeof l === 'object' ? l.message : l) || []
              let currentBenchmark = null
              let progressPercent = 0
              let accuracy = null
              
              const benchmarkPattern = /(MMLU|TruthfulQA|HellaSwag|ARC|WinoGrande|CommonsenseQA|BoolQ|SafetyBench|Do-Not-Answer|DONOTANSWER)/i
              
              // Find current benchmark
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
              
              // Find latest progress
              for (let i = logs.length - 1; i >= 0; i--) {
                const line = logs[i]
                if (line && line.includes('Progress:')) {
                  const percentMatch = line.match(/(\d+)%/)
                  const accMatch = line.match(/Acc:.*?(\d+\.?\d*)%/)
                  if (percentMatch) progressPercent = parseInt(percentMatch[1])
                  if (accMatch) accuracy = parseFloat(accMatch[1])
                  break
                }
              }
              
              return currentBenchmark && (
                <div className="mt-3">
                  <div className="flex justify-between text-xs text-tertiary mb-2">
                    <span>{formatBenchmarkName(currentBenchmark)} - {progressPercent}%</span>
                    {accuracy !== null && (
                      <span className="text-green-400">Current Accuracy: {accuracy}%</span>
                    )}
                  </div>
                  <div className="h-2 bg-surface-active rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-primary-500 to-purple-500 transition-all duration-500"
                      style={{ width: `${progressPercent}%` }}
                    />
                  </div>
                </div>
              )
            })()}
          </div>
        )}

        {/* Queue Items */}
        <div className="bg-surface rounded-xl border border-default overflow-hidden">
          <div className="px-4 py-3 border-b border-default">
            <h3 className="font-medium text-primary flex items-center gap-2">
              <ListOrdered className="w-4 h-4 text-tertiary" />
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
                  <div className="font-medium text-primary">
                    {item.model}
                    <span className="text-tertiary font-normal ml-2">({item.provider})</span>
                  </div>
                  <div className="text-sm text-tertiary">
                    {formatBenchmarks(item.benchmarks)} • {item.sample_size} samples
                  </div>
                  {item.inference_settings && (
                    <div className="text-xs text-tertiary mt-1 font-mono">
                      ⚙ {formatInferenceSettings(item.inference_settings)}
                    </div>
                  )}
                </div>
                {item.score !== null && item.score !== undefined && (
                  <div className="text-lg font-bold text-green-400">
                    {(item.score * 100).toFixed(1)}%
                  </div>
                )}
                {item.duration_seconds && (
                  <div className="text-sm text-tertiary">
                    {Math.floor(item.duration_seconds / 60)}m {Math.floor(item.duration_seconds % 60)}s
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {queueStatus.eta_seconds && (
          <div className="mt-4 text-sm text-tertiary flex items-center gap-2">
            <Clock className="w-4 h-4" />
            ETA: ~{Math.ceil(queueStatus.eta_seconds / 60)} minutes remaining
          </div>
        )}

        {/* Live Logs for current running item */}
        {currentQueueRun && (
          <div className="mt-6 bg-surface rounded-xl border border-default overflow-hidden">
            <div className="px-4 py-3 border-b border-default flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-tertiary" />
                <h3 className="font-medium text-primary">Live Logs</h3>
                <span className="text-xs text-tertiary">
                  ({currentQueueRun.logs?.length || 0} entries)
                </span>
              </div>
              <div className="flex items-center gap-3">
                {!autoScrollQueueLogs && (
                  <button
                    onClick={() => {
                      setAutoScrollQueueLogs(true)
                      queueLogsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
                    }}
                    className="text-xs px-2 py-1 rounded bg-surface-active text-tertiary hover:bg-surface-hover hover:text-primary transition-colors flex items-center gap-1"
                    title="Resume auto-scroll to bottom"
                  >
                    ↓ Scroll to bottom
                  </button>
                )}
                <span className="text-xs text-primary-400 animate-pulse">● Live</span>
              </div>
            </div>
            <div className="p-4 h-64 overflow-auto font-mono text-xs" ref={queueLogsContainerRef}>
              {(!currentQueueRun.logs || currentQueueRun.logs.length === 0) ? (
                <p className="text-tertiary">Waiting for logs...</p>
              ) : (
                currentQueueRun.logs.slice(-100).map((log, i) => {
                  const message = typeof log === 'object' ? log.message : log
                  return (
                    <div
                      key={i}
                      className={`leading-relaxed py-0.5 ${message?.includes('Progress:') ? 'text-primary-300' :
                        message?.includes('Accuracy:') || message?.includes('Score:') ? 'text-green-300' :
                          message?.includes('Error') || message?.includes('failed') ? 'text-red-300' :
                            'text-tertiary'
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
          <Activity className="w-16 h-16 text-secondary mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-primary mb-2">No Active Run</h2>
          <p className="text-tertiary mb-6">
            {isRunFinished
              ? 'The last evaluation has finished. Start a new one or view results in History.'
              : 'Start a new evaluation to see progress here'
            }
          </p>
          <div className="flex gap-4 justify-center">
            <button
              onClick={() => navigate('/')}
              className="btn btn-accent px-6"
            >
              Start New Run
            </button>
            {isRunFinished && (
              <button
                onClick={() => navigate(`/run/${activeRun}`)}
                className="btn btn-secondary px-6"
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
            <h1 className="text-3xl font-bold text-primary mb-2">Evaluation Progress</h1>
            <p className="text-tertiary">Run ID: {activeRun}</p>
          </div>
          {status === 'running' && (
            <div className="flex items-center gap-2 text-tertiary">
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
            status === 'cancelled' ? 'bg-surface-active/50 border-default' :
              'bg-surface border-default'
        }`}>
        <div className="flex items-center gap-4">
          {status === 'connecting' && (
            <>
              <Loader2 className="w-8 h-8 text-tertiary animate-spin" />
              <div>
                <h3 className="font-semibold text-primary">Connecting...</h3>
                <p className="text-sm text-tertiary">Fetching evaluation status...</p>
              </div>
            </>
          )}
          {status === 'running' && (
            <>
              <div className="w-8 h-8 rounded-full bg-primary-500/20 flex items-center justify-center pulse-glow">
                <Activity className="w-5 h-5 text-primary-400" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-primary">Running - {run?.model || 'Loading...'}</h3>
                <p className="text-sm text-tertiary">
                  {progressInfo.benchmark ? `Running ${progressInfo.benchmark}...` : 'Initializing...'}
                  {progressInfo.accuracy !== null && ` (Current Accuracy: ${progressInfo.accuracy}%)`}
                </p>
              </div>
              <div className="text-right mr-4">
                <div className="text-2xl font-bold text-primary-400">{progressInfo.percent}%</div>
                <div className="text-sm text-tertiary">
                  {logs.length} log entries
                </div>
              </div>
              <button
                onClick={handleCancel}
                disabled={cancelling}
                className="px-4 py-2 rounded-lg bg-surface-active border border-default text-secondary hover:bg-surface-hover hover:text-primary transition-colors flex items-center gap-2 disabled:opacity-50"
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
                <h3 className="font-semibold text-primary">Complete - {run?.model}</h3>
                <p className="text-sm text-tertiary">Evaluation finished successfully</p>
              </div>
              <button
                onClick={() => navigate(`/run/${activeRun}`)}
                className="px-4 py-2 rounded-lg bg-green-500 text-primary hover:bg-green-600 transition-colors"
              >
                View Results
              </button>
            </>
          )}
          {status === 'cancelled' && (
            <>
              <Square className="w-8 h-8 text-tertiary" />
              <div className="flex-1">
                <h3 className="font-semibold text-primary">Stopped - {run?.model}</h3>
                <p className="text-sm text-tertiary">Evaluation was stopped</p>
              </div>
              <button
                onClick={() => navigate('/')}
                className="btn btn-accent"
              >
                Start New Run
              </button>
            </>
          )}
          {status === 'error' && (
            <>
              <XCircle className="w-8 h-8 text-red-400" />
              <div className="flex-1">
                <h3 className="font-semibold text-primary">Error</h3>
                <p className="text-sm text-red-400">{error}</p>
              </div>
              <button
                onClick={() => navigate(`/run/${activeRun}`)}
                className="btn btn-secondary"
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
          <div className="h-4 bg-surface-active rounded-full overflow-hidden">
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
                : 'bg-surface-active text-secondary'
                }`}
            >
              {b.toUpperCase()}
            </span>
          ))}
        </div>
      )}

      {/* Logs */}
      <div className="bg-surface rounded-xl border border-default overflow-hidden">
        <div className="px-4 py-3 border-b border-default flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-tertiary" />
            <h3 className="font-medium text-primary">Logs</h3>
            <span className="text-xs text-tertiary">({logs.length} entries)</span>
          </div>
          <div className="flex items-center gap-3">
            {!autoScrollLogs && status === 'running' && (
              <button
                onClick={() => {
                  setAutoScrollLogs(true)
                  logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
                }}
                className="text-xs px-2 py-1 rounded bg-surface-active text-tertiary hover:bg-surface-hover hover:text-primary transition-colors flex items-center gap-1"
                title="Resume auto-scroll to bottom"
              >
                ↓ Scroll to bottom
              </button>
            )}
            {status === 'running' && (
              <span className="text-xs text-primary-400 animate-pulse">● Live</span>
            )}
          </div>
        </div>
        <div className="p-4 h-80 overflow-auto font-mono text-xs" ref={logsContainerRef}>
          {logs.length === 0 ? (
            <p className="text-tertiary">Waiting for logs...</p>
          ) : (
            logs.slice(-100).map((log, i) => (
              <div
                key={i}
                className={`leading-relaxed py-0.5 ${log?.includes('Progress:') ? 'text-primary-300' :
                  log?.includes('Accuracy:') || log?.includes('Score:') ? 'text-green-300' :
                    log?.includes('Error') || log?.includes('failed') ? 'text-red-300' :
                      'text-tertiary'
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

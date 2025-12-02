import { useState, useEffect } from 'react'
import {
    ListOrdered,
    X,
    ChevronUp,
    ChevronDown,
    Play,
    Trash2,
    Clock,
    CheckCircle,
    Loader2,
    XCircle,
    Pause
} from 'lucide-react'
import { startQueue, fetchQueueStatus, cancelQueue, subscribeToQueueProgress } from '../api'

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

function QueueBuilder({ queue, onQueueChange, onStartQueue }) {
    const [isRunning, setIsRunning] = useState(false)
    const [queueStatus, setQueueStatus] = useState(null)
    const [starting, setStarting] = useState(false)
    const [sseError, setSseError] = useState(false)

    useEffect(() => {
        // Only check for existing queue on mount
        let mounted = true

        async function checkOnMount() {
            try {
                const status = await fetchQueueStatus()
                if (!mounted) return
                if (status && (status.status === 'running' || status.status === 'pending')) {
                    setQueueStatus(status)
                    setIsRunning(true)
                }
            } catch (err) {
                // No active queue
            }
        }

        checkOnMount()
        return () => { mounted = false }
    }, [])

    // Separate effect for polling - only when running and SSE fails
    useEffect(() => {
        if (!isRunning) return
        if (!sseError) return  // SSE is working, no need to poll

        const interval = setInterval(() => {
            checkQueueStatus()
        }, 5000)  // Poll every 5 seconds as fallback

        return () => clearInterval(interval)
    }, [isRunning, sseError])

    useEffect(() => {
        if (isRunning && queueStatus?.queue_id) {
            setSseError(false)
            // Subscribe to progress
            const unsubscribe = subscribeToQueueProgress(
                (status) => {
                    setQueueStatus(status)
                    setSseError(false)
                    if (status.status === 'completed' || status.status === 'failed' || status.status === 'cancelled') {
                        setIsRunning(false)
                    }
                },
                (error) => {
                    console.error('Queue SSE error:', error)
                    setSseError(true)
                    // Don't set isRunning to false - poll will handle it
                }
            )
            return unsubscribe
        }
    }, [isRunning, queueStatus?.queue_id])

    async function checkQueueStatus() {
        try {
            const status = await fetchQueueStatus()
            if (status) {
                setQueueStatus(status)
                setIsRunning(status.status === 'running' || status.status === 'pending')
            } else {
                // No active queue
                if (isRunning) {
                    setIsRunning(false)
                }
            }
        } catch (err) {
            // No active queue or error
            console.log('Queue status check:', err.message)
        }
    }

    function removeFromQueue(index) {
        const newQueue = [...queue]
        newQueue.splice(index, 1)
        onQueueChange(newQueue)
    }

    function moveUp(index) {
        if (index === 0) return
        const newQueue = [...queue]
            ;[newQueue[index - 1], newQueue[index]] = [newQueue[index], newQueue[index - 1]]
        onQueueChange(newQueue)
    }

    function moveDown(index) {
        if (index === queue.length - 1) return
        const newQueue = [...queue]
            ;[newQueue[index], newQueue[index + 1]] = [newQueue[index + 1], newQueue[index]]
        onQueueChange(newQueue)
    }

    function clearQueue() {
        onQueueChange([])
    }

    async function handleStartQueue() {
        setStarting(true)
        try {
            const result = await startQueue(queue)
            setQueueStatus({
                queue_id: result.queue_id,
                status: 'running',
                items: queue.map((item, i) => ({
                    index: i,
                    model: item.model,
                    provider: item.provider,
                    benchmarks: item.benchmarks,
                    sample_size: item.sample_size,
                    status: i === 0 ? 'running' : 'pending',
                })),
                current_index: 0,
                total: queue.length,
            })
            setIsRunning(true)
            onStartQueue()
        } catch (err) {
            console.error('Failed to start queue:', err)
        } finally {
            setStarting(false)
        }
    }

    async function handleCancelQueue() {
        try {
            await cancelQueue()
            setIsRunning(false)
            setQueueStatus(null)
        } catch (err) {
            console.error('Failed to cancel queue:', err)
        }
    }

    // Estimate time (rough: 1 minute per 10 samples per benchmark)
    function estimateTime(item) {
        const benchmarkCount = item.benchmarks?.length || 3
        const samples = item.sample_size || 100
        return Math.ceil((samples * benchmarkCount) / 10)
    }

    const totalEstimatedMinutes = queue.reduce((sum, item) => sum + estimateTime(item), 0)

    function getStatusIcon(status) {
        switch (status) {
            case 'completed':
                return <CheckCircle className="w-4 h-4 text-green-400" />
            case 'running':
                return <Loader2 className="w-4 h-4 text-primary-400 animate-spin" />
            case 'failed':
                return <XCircle className="w-4 h-4 text-red-400" />
            case 'pending':
            default:
                return <Pause className="w-4 h-4 text-slate-400" />
        }
    }

    if (isRunning && queueStatus) {
        // Show running queue status
        const completed = queueStatus.items?.filter(i => i.status === 'completed').length || 0
        return (
            <div className="bg-slate-800 rounded-xl border border-primary-500/50 p-6 mb-6">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                        <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />
                        Queue Running ({completed}/{queueStatus.total})
                    </h2>
                    <button
                        onClick={handleCancelQueue}
                        className="px-4 py-2 rounded-lg bg-red-600/20 text-red-400 hover:bg-red-600/30 border border-red-600/50 flex items-center gap-2 transition-colors"
                    >
                        <X className="w-4 h-4" />
                        Cancel All
                    </button>
                </div>

                <div className="space-y-2">
                    {queueStatus.items?.map((item, index) => (
                        <div
                            key={index}
                            className={`flex items-center gap-4 p-3 rounded-lg ${item.status === 'running'
                                ? 'bg-primary-500/10 border border-primary-500/30'
                                : item.status === 'completed'
                                    ? 'bg-green-500/10'
                                    : 'bg-slate-700/50'
                                }`}
                        >
                            {getStatusIcon(item.status)}
                            <div className="flex-1">
                                <div className="font-medium text-white">{item.model}</div>
                                <div className="text-xs text-slate-400">
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

                {queueStatus.eta_seconds && (
                    <div className="mt-4 text-sm text-slate-400 flex items-center gap-2">
                        <Clock className="w-4 h-4" />
                        ETA: ~{Math.ceil(queueStatus.eta_seconds / 60)} minutes remaining
                    </div>
                )}
            </div>
        )
    }

    if (queue.length === 0) {
        return null
    }

    return (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                    <ListOrdered className="w-5 h-5 text-primary-400" />
                    Queue ({queue.length} runs)
                </h2>
                <div className="flex items-center gap-2">
                    <button
                        onClick={clearQueue}
                        className="px-3 py-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition-colors text-sm"
                    >
                        Clear
                    </button>
                    <button
                        onClick={handleStartQueue}
                        disabled={starting || queue.length === 0}
                        className="px-4 py-2 rounded-lg bg-gradient-to-r from-primary-500 to-purple-500 text-white font-medium hover:from-primary-600 hover:to-purple-600 disabled:opacity-50 flex items-center gap-2 transition-all"
                    >
                        {starting ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                            <Play className="w-4 h-4" />
                        )}
                        Start Queue
                    </button>
                </div>
            </div>

            <div className="space-y-2">
                {queue.map((item, index) => (
                    <div
                        key={index}
                        className="flex items-center gap-3 p-3 bg-slate-700/50 rounded-lg"
                    >
                        <span className="w-6 h-6 rounded-full bg-slate-600 flex items-center justify-center text-sm font-medium text-white">
                            {index + 1}
                        </span>
                        <div className="flex-1">
                            <div className="font-medium text-white">{item.model}</div>
                            <div className="text-xs text-slate-400">
                                {formatBenchmarks(item.benchmarks)} • {item.sample_size} samples
                            </div>
                        </div>
                        <div className="text-sm text-slate-400 flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            ~{estimateTime(item)}min
                        </div>
                        <div className="flex items-center gap-1">
                            <button
                                onClick={() => moveUp(index)}
                                disabled={index === 0}
                                className="p-1.5 rounded text-slate-400 hover:text-white hover:bg-slate-600 disabled:opacity-30 transition-colors"
                            >
                                <ChevronUp className="w-4 h-4" />
                            </button>
                            <button
                                onClick={() => moveDown(index)}
                                disabled={index === queue.length - 1}
                                className="p-1.5 rounded text-slate-400 hover:text-white hover:bg-slate-600 disabled:opacity-30 transition-colors"
                            >
                                <ChevronDown className="w-4 h-4" />
                            </button>
                            <button
                                onClick={() => removeFromQueue(index)}
                                className="p-1.5 rounded text-slate-400 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                            >
                                <Trash2 className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                ))}
            </div>

            <div className="mt-4 pt-4 border-t border-slate-700 flex items-center justify-between text-sm">
                <div className="text-slate-400 flex items-center gap-2">
                    <Clock className="w-4 h-4" />
                    Total estimated: ~{totalEstimatedMinutes} minutes
                </div>
            </div>
        </div>
    )
}

export default QueueBuilder

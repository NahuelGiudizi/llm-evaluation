import { useState, useEffect } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  LineChart,
  Line
} from 'recharts'
import { BarChart3, Loader2, RefreshCw, TrendingUp, Settings2, Hash } from 'lucide-react'
import { fetchRuns } from '../api'

// Format benchmark name for display
function formatBenchmarkName(name) {
  const nameMap = {
    'mmlu': 'MMLU',
    'truthfulqa': 'TruthfulQA',
    'hellaswag': 'HellaSwag',
    'arc': 'ARC-Challenge',
    'winogrande': 'WinoGrande',
    'commonsenseqa': 'CommonsenseQA',
    'boolq': 'BoolQ',
    'safetybench': 'SafetyBench',
    'donotanswer': 'Do-Not-Answer',
    'gsm8k': 'GSM8K',
    'average': 'Average',
  }
  return nameMap[name.toLowerCase()] || name.toUpperCase()
}

// Benchmark descriptions for tooltips
const benchmarkDescriptions = {
  'MMLU': 'Massive Multitask Language Understanding - Tests knowledge across 57 subjects including STEM, humanities, and social sciences',
  'TruthfulQA': 'Measures model truthfulness - Evaluates tendency to generate false but plausible-sounding answers',
  'HellaSwag': 'Commonsense reasoning - Tests ability to complete sentences with the most plausible ending',
  'ARC-Challenge': 'AI2 Reasoning Challenge - Science questions requiring multi-step logical reasoning',
  'WinoGrande': 'Pronoun resolution - Tests commonsense reasoning through fill-in-the-blank challenges',
  'CommonsenseQA': 'Commonsense knowledge - Questions requiring real-world background knowledge',
  'BoolQ': 'Boolean Questions - Yes/no reading comprehension from real Google queries',
  'SafetyBench': 'Safety evaluation - Tests model behavior on safety and ethics-related scenarios',
  'Do-Not-Answer': 'Refusal detection - Measures rate of refusing to answer harmful prompts',
  'GSM8K': 'Grade School Math - Mathematical reasoning with multi-step word problems',
}

// Custom tooltip component for charts
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null

  const description = benchmarkDescriptions[label] || ''

  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 shadow-xl max-w-xs">
      <p className="font-semibold text-white mb-1">{label}</p>
      {description && (
        <p className="text-xs text-slate-400 mb-2">{description}</p>
      )}
      <div className="space-y-1">
        {payload.map((entry, index) => (
          <div key={index} className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-sm text-slate-300">{entry.name}:</span>
            <span className="text-sm font-medium text-white">{entry.value}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

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
  if (data.accuracy !== undefined) return data.accuracy
  if (data.correct !== undefined && data.questions_tested !== undefined && data.questions_tested > 0) {
    return data.correct / data.questions_tested
  }
  if (data.correct !== undefined && data.scenarios_tested !== undefined && data.scenarios_tested > 0) {
    return data.correct / data.scenarios_tested
  }
  return 0
}

function ModelComparison() {
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedRuns, setSelectedRuns] = useState([])
  const [chartType, setChartType] = useState('bar')

  useEffect(() => {
    loadRuns()
  }, [])

  async function loadRuns() {
    setLoading(true)
    try {
      const data = await fetchRuns()
      // Filter to only completed runs with results and known models (not legacy)
      const completedRuns = data.filter(r =>
        r.status === 'completed' &&
        r.results &&
        Object.keys(r.results).length > 0 &&
        !r.model?.includes('unknown')
      )
      setRuns(completedRuns)
      // Auto-select last 3 runs for comparison
      setSelectedRuns(completedRuns.slice(0, 3).map(r => r.run_id))
    } catch (err) {
      console.error('Failed to load runs:', err)
    } finally {
      setLoading(false)
    }
  }

  function toggleRunSelection(runId) {
    if (selectedRuns.includes(runId)) {
      setSelectedRuns(selectedRuns.filter(id => id !== runId))
    } else {
      setSelectedRuns([...selectedRuns, runId])
    }
  }

  // Prepare chart data
  const chartData = []
  const benchmarkNames = new Set()

  const selectedRunsData = runs.filter(r => selectedRuns.includes(r.run_id))
  selectedRunsData.forEach(run => {
    if (run.results) {
      Object.keys(run.results).forEach(name => {
        // Skip "average" as it's not a real benchmark
        if (name.toLowerCase() !== 'average') {
          benchmarkNames.add(name)
        }
      })
    }
  })

  // Generate unique keys for each run that include sample size info
  const getRunKey = (run) => {
    const samples = run.sample_size ? `n=${run.sample_size}` : ''
    return `${run.model} ${samples}`.trim()
  }

  benchmarkNames.forEach(benchmark => {
    const dataPoint = { benchmark: formatBenchmarkName(benchmark) }
    selectedRunsData.forEach(run => {
      const data = run.results?.[benchmark]
      if (data) {
        const score = extractScore(benchmark, data)
        dataPoint[getRunKey(run)] = parseFloat((score * 100).toFixed(1))
      }
    })
    chartData.push(dataPoint)
  })

  // Radar chart data - needs numeric values, not strings
  const radarData = Array.from(benchmarkNames).map(benchmark => {
    const dataPoint = { benchmark: formatBenchmarkName(benchmark) }
    selectedRunsData.forEach(run => {
      const data = run.results?.[benchmark]
      if (data) {
        const score = extractScore(benchmark, data)
        dataPoint[getRunKey(run)] = parseFloat((score * 100).toFixed(1))
      }
    })
    return dataPoint
  })

  const colors = ['#0ea5e9', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444']

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
          <h1 className="text-3xl font-bold text-white mb-2">Model Comparison</h1>
          <p className="text-slate-400">Compare benchmark results across different models</p>
        </div>
        <button
          onClick={loadRuns}
          className="px-4 py-2 rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-800 flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {runs.length === 0 ? (
        <div className="text-center py-20">
          <BarChart3 className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-white mb-2">No Completed Runs</h2>
          <p className="text-slate-400">Complete some evaluations to see comparisons here</p>
        </div>
      ) : (
        <>
          {/* Run selection */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Select runs to compare:</h3>
            <div className="flex flex-wrap gap-2">
              {runs.map(run => {
                const runTime = new Date(run.started_at)
                const timeStr = runTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                const dateStr = runTime.toLocaleDateString()
                const runIdShort = run.run_id?.slice(-6) || ''
                const sampleInfo = run.sample_size ? `${run.sample_size} samples` : run.preset || 'custom'

                return (
                  <button
                    key={run.run_id}
                    onClick={() => toggleRunSelection(run.run_id)}
                    className={`px-3 py-2 rounded-lg border text-sm transition-colors flex flex-col items-start ${selectedRuns.includes(run.run_id)
                      ? 'bg-primary-500/20 border-primary-500/50 text-primary-400'
                      : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-600'
                      }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{run.model}</span>
                      <span className="text-xs opacity-60 font-mono">#{runIdShort}</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs opacity-60">
                      <span>{dateStr} {timeStr}</span>
                      <span className="text-primary-400/80">• {sampleInfo}</span>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Chart type selector */}
          <div className="mb-6 flex gap-2">
            <button
              onClick={() => setChartType('bar')}
              className={`px-4 py-2 rounded-lg text-sm transition-colors ${chartType === 'bar'
                ? 'bg-primary-500 text-white'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                }`}
            >
              Bar Chart
            </button>
            <button
              onClick={() => setChartType('radar')}
              className={`px-4 py-2 rounded-lg text-sm transition-colors ${chartType === 'radar'
                ? 'bg-primary-500 text-white'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                }`}
            >
              Radar Chart
            </button>
          </div>

          {selectedRuns.length === 0 ? (
            <div className="text-center py-20 bg-slate-800 rounded-xl border border-slate-700">
              <TrendingUp className="w-12 h-12 text-slate-600 mx-auto mb-4" />
              <p className="text-slate-400">Select at least one run to see the chart</p>
            </div>
          ) : (
            <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
              {chartType === 'bar' ? (
                <ResponsiveContainer width="100%" height={450}>
                  <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis 
                      dataKey="benchmark" 
                      tick={{ fill: '#94a3b8', fontSize: 12 }} 
                      angle={-35}
                      textAnchor="end"
                      height={80}
                      interval={0}
                    />
                    <YAxis tick={{ fill: '#94a3b8' }} domain={[0, 100]} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ paddingTop: 20 }} />
                    {selectedRunsData.map((run, i) => (
                      <Bar
                        key={run.run_id}
                        dataKey={getRunKey(run)}
                        fill={colors[i % colors.length]}
                      />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              ) : radarData.length >= 3 ? (
                <ResponsiveContainer width="100%" height={400}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#334155" />
                    <PolarAngleAxis dataKey="benchmark" tick={{ fill: '#94a3b8' }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#94a3b8' }} />
                    {selectedRunsData.map((run, i) => (
                      <Radar
                        key={run.run_id}
                        name={getRunKey(run)}
                        dataKey={getRunKey(run)}
                        stroke={colors[i % colors.length]}
                        fill={colors[i % colors.length]}
                        fillOpacity={0.3}
                      />
                    ))}
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />
                  </RadarChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-96 flex items-center justify-center text-slate-400">
                  <p>Radar chart requires at least 3 benchmarks to display properly.</p>
                </div>
              )}
            </div>
          )}

          {/* Summary Table */}
          {selectedRuns.length > 0 && (
            <div className="mt-6 bg-slate-800 rounded-xl border border-slate-700 overflow-x-auto">
              <table className="w-full min-w-max">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="px-4 py-3 text-left text-sm font-medium text-slate-400">Model</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-slate-400">
                      <div className="flex items-center gap-1">
                        <Settings2 className="w-3 h-3" />
                        Config
                      </div>
                    </th>
                    {Array.from(benchmarkNames).map(name => (
                      <th key={name} className="px-4 py-3 text-left text-sm font-medium text-slate-400">
                        {formatBenchmarkName(name)}
                      </th>
                    ))}
                    <th className="px-4 py-3 text-left text-sm font-medium text-slate-400">Average</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedRunsData.map(run => {
                    // Calculate average only from real benchmarks, not "average" key
                    const scores = Object.entries(run.results || {})
                      .filter(([name]) => name.toLowerCase() !== 'average')
                      .map(([name, data]) => extractScore(name, data))
                    const avg = scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0

                    // Get config info
                    const sampleSize = run.sample_size || '-'
                    const preset = run.preset || 'custom'

                    return (
                      <tr key={run.run_id} className="border-b border-slate-700/50">
                        <td className="px-4 py-3">
                          <div className="text-white font-medium">{run.model}</div>
                          <div className="text-xs text-slate-500 font-mono">#{run.run_id?.slice(-6)}</div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-xs space-y-1">
                            <div className="flex items-center gap-1 text-slate-300">
                              <Hash className="w-3 h-3 text-slate-500" />
                              <span>{sampleSize} samples</span>
                            </div>
                            <div className="text-slate-500 capitalize">{preset}</div>
                          </div>
                        </td>
                        {Array.from(benchmarkNames).map(name => {
                          const data = run.results?.[name]
                          const score = data ? extractScore(name, data) : null
                          return (
                            <td key={name} className="px-4 py-3 text-slate-300">
                              {score !== null ? `${(score * 100).toFixed(1)}%` : '-'}
                            </td>
                          )
                        })}
                        <td className="px-4 py-3 text-primary-400 font-semibold">
                          {(avg * 100).toFixed(1)}%
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default ModelComparison

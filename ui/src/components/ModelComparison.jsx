import { useState, useEffect, useRef } from 'react'
import ReactDOM from 'react-dom'
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
  Line,
  ScatterChart,
  Scatter,
  ZAxis,
  Cell
} from 'recharts'
import { BarChart3, Loader2, RefreshCw, TrendingUp, Settings2, Hash, Activity, Grid3x3, Zap } from 'lucide-react'
import { fetchRuns } from '../api'

// Helper to get CSS custom property color
function getCSSColor(varName) {
  const rgb = getComputedStyle(document.documentElement)
    .getPropertyValue(varName)
    .trim()
  return rgb ? `rgb(${rgb})` : '#64748b'
}

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

// Custom tick component for XAxis with tooltip
function CustomXAxisTick({ x, y, payload }) {
  const [showTooltip, setShowTooltip] = useState(false)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })
  const textRef = useRef(null)
  const description = benchmarkDescriptions[payload.value] || ''

  const handleMouseEnter = (e) => {
    if (textRef.current) {
      const rect = textRef.current.getBoundingClientRect()
      const scrollX = window.pageXOffset || document.documentElement.scrollLeft
      const scrollY = window.pageYOffset || document.documentElement.scrollTop
      setTooltipPos({
        x: rect.left + scrollX,
        y: rect.top + scrollY
      })
    }
    setShowTooltip(true)
  }

  return (
    <>
      <g transform={`translate(${x},${y})`}>
        <text
          ref={textRef}
          x={0}
          y={0}
          dy={16}
          textAnchor="end"
          fill="#94a3b8"
          fontSize={12}
          transform="rotate(-35)"
          style={{ cursor: 'help' }}
          onMouseEnter={handleMouseEnter}
          onMouseLeave={() => setShowTooltip(false)}
        >
          {payload.value}
        </text>
      </g>
      {showTooltip && description && ReactDOM.createPortal(
        <div
          className="fixed bg-surface dark:bg-slate-800 border border-default rounded-xl p-3 shadow-2xl text-xs text-secondary leading-relaxed max-w-xs"
          style={{
            left: `${tooltipPos.x}px`,
            top: `${tooltipPos.y}px`,
            transform: 'translate(-50%, -120%)',
            zIndex: 9999,
            pointerEvents: 'none'
          }}
        >
          <p className="font-semibold text-primary mb-1.5">{payload.value}</p>
          <p>{description}</p>
        </div>,
        document.body
      )}
    </>
  )
}

// Custom tick component for Radar chart labels with tooltip
function CustomRadarTick({ payload, x, y, cx, cy, ...rest }) {
  const [showTooltip, setShowTooltip] = useState(false)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })
  const textRef = useRef(null)
  const description = benchmarkDescriptions[payload.value] || ''

  // Calculate text anchor based on position
  const textAnchor = x > cx ? 'start' : x < cx ? 'end' : 'middle'
  const dy = y > cy ? 10 : y < cy ? -5 : 0

  const handleMouseEnter = (e) => {
    if (textRef.current) {
      const rect = textRef.current.getBoundingClientRect()
      const scrollX = window.pageXOffset || document.documentElement.scrollLeft
      const scrollY = window.pageYOffset || document.documentElement.scrollTop
      setTooltipPos({
        x: rect.left + scrollX + rect.width / 2,
        y: rect.top + scrollY - 10
      })
    }
    setShowTooltip(true)
  }

  return (
    <>
      <text
        ref={textRef}
        {...rest}
        x={x}
        y={y}
        dy={dy}
        textAnchor={textAnchor}
        className="fill-slate-700 dark:fill-slate-200"
        fontSize={14}
        fontWeight={700}
        style={{ cursor: 'help' }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={() => setShowTooltip(false)}
      >
        {payload.value}
      </text>
      {showTooltip && description && ReactDOM.createPortal(
        <div
          className="fixed bg-surface dark:bg-slate-800 border border-default rounded-xl p-3 shadow-2xl text-xs text-secondary leading-relaxed max-w-xs"
          style={{
            left: `${tooltipPos.x}px`,
            top: `${tooltipPos.y}px`,
            transform: 'translate(-50%, -100%)',
            zIndex: 9999,
            pointerEvents: 'none'
          }}
        >
          <p className="font-semibold text-primary mb-1.5">{payload.value}</p>
          <p>{description}</p>
        </div>,
        document.body
      )}
    </>
  )
}

// Custom tooltip component for charts
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null

  const description = benchmarkDescriptions[label] || ''

  return (
    <div className="bg-surface dark:bg-slate-800 border border-default rounded-xl p-4 shadow-2xl max-w-xs" style={{ zIndex: 9999 }}>
      <p className="font-semibold text-primary mb-1">{label}</p>
      {description && (
        <p className="text-xs text-tertiary mb-2">{description}</p>
      )}
      <div className="space-y-1">
        {payload.map((entry, index) => (
          <div key={index} className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-sm text-secondary">{entry.name}:</span>
            <span className="text-sm font-medium text-primary">{entry.value}%</span>
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
  if (data.donotanswer_accuracy !== undefined) return data.donotanswer_accuracy
  if (data.gsm8k_accuracy !== undefined) return data.gsm8k_accuracy
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
  const [sortBy, setSortBy] = useState('date-desc') // date-desc, date-asc, model-asc, model-desc, score-desc, score-asc, samples-desc, samples-asc, temp-asc, temp-desc
  const [highlightedRun, setHighlightedRun] = useState(null) // Track which run is highlighted
  const [heatmapSortBy, setHeatmapSortBy] = useState(null) // Track which benchmark column to sort by in heatmap
  const [heatmapSortOrder, setHeatmapSortOrder] = useState('desc') // 'asc' or 'desc'

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

  // Sort runs based on selected criteria
  const sortedRuns = [...runs].sort((a, b) => {
    switch (sortBy) {
      case 'date-desc':
        return new Date(b.started_at) - new Date(a.started_at)
      case 'date-asc':
        return new Date(a.started_at) - new Date(b.started_at)
      case 'model-asc':
        return a.model.localeCompare(b.model)
      case 'model-desc':
        return b.model.localeCompare(a.model)
      case 'score-desc': {
        const avgA = calculateAverage(a.results)
        const avgB = calculateAverage(b.results)
        return avgB - avgA
      }
      case 'score-asc': {
        const avgA = calculateAverage(a.results)
        const avgB = calculateAverage(b.results)
        return avgA - avgB
      }
      case 'samples-desc':
        return (b.sample_size || 0) - (a.sample_size || 0)
      case 'samples-asc':
        return (a.sample_size || 0) - (b.sample_size || 0)
      case 'temp-asc':
        return (a.inference_settings?.temperature ?? 0) - (b.inference_settings?.temperature ?? 0)
      case 'temp-desc':
        return (b.inference_settings?.temperature ?? 0) - (a.inference_settings?.temperature ?? 0)
      default:
        return 0
    }
  })

  // Helper to calculate average score
  function calculateAverage(results) {
    if (!results) return 0
    const scores = Object.entries(results)
      .filter(([name]) => name.toLowerCase() !== 'average')
      .map(([name, data]) => extractScore(name, data))
    return scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0
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

  // Generate unique keys for each run that include sample size and run_id for uniqueness
  const getRunKey = (run) => {
    const samples = run.sample_size ? `n=${run.sample_size}` : ''
    const runIdShort = run.run_id ? `#${run.run_id.slice(-6)}` : ''
    // Include inference settings if different from defaults
    let inferenceInfo = ''
    if (run.inference_settings) {
      const settings = run.inference_settings
      const parts = []
      if (settings.temperature !== undefined && settings.temperature !== 0) {
        parts.push(`T=${settings.temperature}`)
      }
      if (settings.seed !== undefined && settings.seed !== 42) {
        parts.push(`S=${settings.seed}`)
      }
      if (parts.length > 0) {
        inferenceInfo = ` (${parts.join(', ')})`
      }
    }
    return `${run.model} ${samples} ${runIdShort}${inferenceInfo}`.trim()
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

  // Radar chart data - needs numeric values for all runs on each benchmark
  // Use 0 as default when a model doesn't have data for a benchmark
  const radarData = Array.from(benchmarkNames).map(benchmark => {
    const dataPoint = { benchmark: formatBenchmarkName(benchmark) }
    selectedRunsData.forEach(run => {
      const data = run.results?.[benchmark]
      if (data) {
        const score = extractScore(benchmark, data)
        dataPoint[getRunKey(run)] = parseFloat((score * 100).toFixed(1))
      } else {
        // Set 0 for missing benchmarks so the area can be drawn
        dataPoint[getRunKey(run)] = 0
      }
    })
    return dataPoint
  })

  // More vibrant and distinguishable colors for charts
  const colors = [
    '#3b82f6', // Blue
    '#8b5cf6', // Purple  
    '#10b981', // Emerald
    '#f59e0b', // Amber
    '#ef4444', // Red
    '#06b6d4', // Cyan
    '#ec4899', // Pink
    '#84cc16', // Lime
  ]

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
          <h1 className="text-3xl font-bold text-slate-900 dark:text-primary mb-2">Model Comparison</h1>
          <p className="text-secondary dark:text-tertiary">Compare benchmark results across different models</p>
        </div>
        <button
          onClick={loadRuns}
          className="px-4 py-2 rounded-lg border border-default text-secondary hover:bg-surface flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {runs.length === 0 ? (
        <div className="text-center py-20">
          <BarChart3 className="w-16 h-16 text-secondary mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-primary mb-2">No Completed Runs</h2>
          <p className="text-tertiary">Complete some evaluations to see comparisons here</p>
        </div>
      ) : (
        <>
          {/* Run selection */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-secondary">Select runs to compare:</h3>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="px-3 py-1.5 rounded-lg bg-surface border border-default text-secondary text-sm focus:outline-none focus:border-primary-500"
              >
                <option value="date-desc">Date (Newest)</option>
                <option value="date-asc">Date (Oldest)</option>
                <option value="model-asc">Model (A-Z)</option>
                <option value="model-desc">Model (Z-A)</option>
                <option value="score-desc">Score (Highest)</option>
                <option value="score-asc">Score (Lowest)</option>
                <option value="samples-desc">Samples (Most)</option>
                <option value="samples-asc">Samples (Least)</option>
                <option value="temp-asc">Temperature (Low-High)</option>
                <option value="temp-desc">Temperature (High-Low)</option>
              </select>
            </div>
            <div className="flex flex-wrap gap-2">
              {sortedRuns.map(run => {
                const runTime = new Date(run.started_at)
                const timeStr = runTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                const dateStr = runTime.toLocaleDateString()
                const runIdShort = run.run_id?.slice(-6) || ''
                const sampleInfo = run.sample_size ? `${run.sample_size} samples` : run.preset || 'custom'

                // Show inference settings if different from defaults
                let inferenceTag = ''
                if (run.inference_settings) {
                  const s = run.inference_settings
                  if (s.temperature !== undefined && s.temperature !== 0) {
                    inferenceTag = `T=${s.temperature}`
                  }
                }

                return (
                  <button
                    key={run.run_id}
                    onClick={() => toggleRunSelection(run.run_id)}
                    className={`px-3 py-2 rounded-lg text-sm transition-all duration-200 flex flex-col items-start ${selectedRuns.includes(run.run_id)
                      ? 'border-[2.5px] border-primary-500 bg-primary-500/20 text-primary-400 shadow-md shadow-primary-500/20 ring-2 ring-primary-500/10 scale-[1.02]'
                      : 'border-2 border-default bg-surface text-tertiary hover:border-slate-400 dark:hover:border-slate-500 hover:bg-surface-hover'
                      }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className={`font-medium transition-colors ${selectedRuns.includes(run.run_id) ? 'text-primary-600 dark:text-primary-300' : ''}`}>{run.model}</span>
                      <span className="text-xs opacity-60 font-mono">#{runIdShort}</span>
                      {inferenceTag && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400">{inferenceTag}</span>
                      )}
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
          <div className="mb-6 flex gap-2 flex-wrap">
            <button
              onClick={() => setChartType('bar')}
              className={`px-4 py-2 rounded-lg text-sm transition-colors flex items-center gap-2 ${chartType === 'bar'
                ? 'bg-interactive text-text-inverse'
                : 'bg-surface text-tertiary hover:bg-surface-hover'
                }`}
            >
              <BarChart3 className="w-4 h-4" />
              Bar Chart
            </button>
            <button
              onClick={() => setChartType('radar')}
              className={`px-4 py-2 rounded-lg text-sm transition-colors flex items-center gap-2 ${chartType === 'radar'
                ? 'bg-interactive text-text-inverse'
                : 'bg-surface text-tertiary hover:bg-surface-hover'
                }`}
            >
              <Activity className="w-4 h-4" />
              Radar Chart
            </button>
            <button
              onClick={() => setChartType('heatmap')}
              className={`px-4 py-2 rounded-lg text-sm transition-colors flex items-center gap-2 ${chartType === 'heatmap'
                ? 'bg-interactive text-text-inverse'
                : 'bg-surface text-tertiary hover:bg-surface-hover'
                }`}
            >
              <Grid3x3 className="w-4 h-4" />
              Heatmap
            </button>
            <button
              onClick={() => setChartType('line')}
              className={`px-4 py-2 rounded-lg text-sm transition-colors flex items-center gap-2 ${chartType === 'line'
                ? 'bg-interactive text-text-inverse'
                : 'bg-surface text-tertiary hover:bg-surface-hover'
                }`}
            >
              <TrendingUp className="w-4 h-4" />
              Timeline
            </button>
            <button
              onClick={() => setChartType('scatter')}
              className={`px-4 py-2 rounded-lg text-sm transition-colors flex items-center gap-2 ${chartType === 'scatter'
                ? 'bg-interactive text-text-inverse'
                : 'bg-surface text-tertiary hover:bg-surface-hover'
                }`}
            >
              <Zap className="w-4 h-4" />
              Trade-offs
            </button>
          </div>

          {selectedRuns.length === 0 ? (
            <div className="text-center py-20 bg-surface rounded-xl border border-default">
              <TrendingUp className="w-12 h-12 text-secondary mx-auto mb-4" />
              <p className="text-tertiary">Select at least one run to see the chart</p>
            </div>
          ) : (
            <div className="bg-surface rounded-xl border border-default p-6">
              {/* Bar Chart */}
              {chartType === 'bar' && (
                <ResponsiveContainer width="100%" height={450}>
                  <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={getCSSColor('--chart-grid')} />
                    <XAxis
                      dataKey="benchmark"
                      tick={<CustomXAxisTick />}
                      height={80}
                      interval={0}
                    />
                    <YAxis tick={{ fill: getCSSColor('--chart-text') }} domain={[0, 100]} />
                    <Tooltip
                      content={<CustomTooltip />}
                      wrapperStyle={{ zIndex: 1000, pointerEvents: 'none' }}
                      cursor={{ fill: 'rgba(0, 0, 0, 0.05)' }}
                      allowEscapeViewBox={{ x: true, y: true }}
                    />
                    <Legend
                      wrapperStyle={{ paddingTop: 20, cursor: 'pointer' }}
                      onClick={(e) => {
                        const runKey = e.value
                        setHighlightedRun(highlightedRun === runKey ? null : runKey)
                      }}
                    />
                    {selectedRunsData.map((run, i) => {
                      const runKey = getRunKey(run)
                      const isHighlighted = highlightedRun === null || highlightedRun === runKey
                      return (
                        <Bar
                          key={run.run_id}
                          dataKey={runKey}
                          fill={colors[i % colors.length]}
                          fillOpacity={isHighlighted ? 1 : 0.2}
                        />
                      )
                    })}
                  </BarChart>
                </ResponsiveContainer>
              )}

              {/* Radar Chart */}
              {chartType === 'radar' && (
                radarData.length >= 3 ? (
                  <ResponsiveContainer width="100%" height={500}>
                    <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
                      <PolarGrid
                        stroke={getCSSColor('--chart-grid')}
                        strokeWidth={1}
                        gridType="polygon"
                      />
                      <PolarAngleAxis
                        dataKey="benchmark"
                        tick={<CustomRadarTick />}
                        tickLine={false}
                      />
                      <PolarRadiusAxis
                        angle={90}
                        domain={[0, 100]}
                        tick={{ fill: getCSSColor('--chart-text'), fontSize: 10 }}
                        tickCount={5}
                        axisLine={false}
                      />
                      {selectedRunsData.map((run, i) => {
                        const runKey = getRunKey(run)
                        const isHighlighted = highlightedRun === null || highlightedRun === runKey
                        return (
                          <Radar
                            key={run.run_id}
                            name={runKey}
                            dataKey={runKey}
                            stroke={colors[i % colors.length]}
                            strokeWidth={isHighlighted ? 3 : 1}
                            fill={colors[i % colors.length]}
                            fillOpacity={isHighlighted ? 0.5 : 0.08}
                            dot={false}
                            activeDot={false}
                            isAnimationActive={false}
                          />
                        )
                      })}
                      <Tooltip
                        content={<CustomTooltip />}
                        wrapperStyle={{ zIndex: 1000, pointerEvents: 'none' }}
                        cursor={false}
                        allowEscapeViewBox={{ x: true, y: true }}
                      />
                      <Legend
                        wrapperStyle={{
                          paddingTop: 20,
                          fontSize: 13,
                          cursor: 'pointer',
                        }}
                        iconType="circle"
                        onClick={(e) => {
                          const runKey = e.value
                          setHighlightedRun(highlightedRun === runKey ? null : runKey)
                        }}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-96 flex items-center justify-center text-tertiary">
                    <p>Radar chart requires at least 3 benchmarks to display properly.</p>
                  </div>
                )
              )}

              {/* Heatmap */}
              {chartType === 'heatmap' && (
                <div className="space-y-4">
                  <p className="text-sm text-tertiary">Heatmap visualization of model performance across benchmarks. Darker green = better performance. Click column headers to sort.</p>
                  <div className="overflow-x-auto">
                    <div className="min-w-max">
                      {/* Header row */}
                      <div className="flex gap-1 mb-1">
                        <div className="w-48 flex-shrink-0" />
                        {Array.from(benchmarkNames).map(name => {
                          const isActiveSortColumn = heatmapSortBy === name
                          return (
                            <div
                              key={name}
                              className={`w-24 text-xs text-center font-medium cursor-pointer hover:bg-surface-active rounded px-2 py-1 transition-colors ${isActiveSortColumn ? 'bg-surface-active text-primary-400' : 'text-tertiary'
                                }`}
                              title={`${benchmarkDescriptions[formatBenchmarkName(name)]}\n\nClick to sort by this benchmark`}
                              onClick={() => {
                                if (heatmapSortBy === name) {
                                  // Toggle sort order if already sorting by this column
                                  setHeatmapSortOrder(heatmapSortOrder === 'desc' ? 'asc' : 'desc')
                                } else {
                                  // Set new sort column, default to descending
                                  setHeatmapSortBy(name)
                                  setHeatmapSortOrder('desc')
                                }
                              }}
                            >
                              <div className="flex items-center justify-center gap-1">
                                {formatBenchmarkName(name)}
                                {isActiveSortColumn && (
                                  <span className="text-[10px]">{heatmapSortOrder === 'desc' ? '▼' : '▲'}</span>
                                )}
                              </div>
                            </div>
                          )
                        })}
                        <div
                          className={`w-24 text-xs text-center font-medium cursor-pointer hover:bg-surface-active rounded px-2 py-1 transition-colors ${heatmapSortBy === 'avg' ? 'bg-surface-active text-primary-400' : 'text-tertiary'
                            }`}
                          onClick={() => {
                            if (heatmapSortBy === 'avg') {
                              setHeatmapSortOrder(heatmapSortOrder === 'desc' ? 'asc' : 'desc')
                            } else {
                              setHeatmapSortBy('avg')
                              setHeatmapSortOrder('desc')
                            }
                          }}
                        >
                          <div className="flex items-center justify-center gap-1">
                            Avg
                            {heatmapSortBy === 'avg' && (
                              <span className="text-[10px]">{heatmapSortOrder === 'desc' ? '▼' : '▲'}</span>
                            )}
                          </div>
                        </div>
                      </div>
                      {/* Data rows */}
                      {(() => {
                        // Sort runs based on selected column
                        let sortedRuns = [...selectedRunsData]
                        if (heatmapSortBy) {
                          sortedRuns.sort((a, b) => {
                            let scoreA, scoreB
                            if (heatmapSortBy === 'avg') {
                              scoreA = calculateAverage(a.results) * 100
                              scoreB = calculateAverage(b.results) * 100
                            } else {
                              const dataA = a.results?.[heatmapSortBy]
                              const dataB = b.results?.[heatmapSortBy]
                              scoreA = dataA ? extractScore(heatmapSortBy, dataA) * 100 : 0
                              scoreB = dataB ? extractScore(heatmapSortBy, dataB) * 100 : 0
                            }
                            return heatmapSortOrder === 'desc' ? scoreB - scoreA : scoreA - scoreB
                          })
                        }

                        return sortedRuns.map((run, i) => {
                          const runKey = getRunKey(run)
                          const avg = calculateAverage(run.results) * 100
                          const isHighlighted = highlightedRun === null || highlightedRun === runKey

                          return (
                            <div
                              key={run.run_id}
                              className={`flex gap-1 mb-1 transition-opacity ${isHighlighted ? 'opacity-100' : 'opacity-30'}`}
                            >
                              <div
                                className="w-48 flex-shrink-0 text-xs text-primary px-2 py-2 bg-surface-active rounded flex items-center justify-between cursor-pointer hover:bg-surface-hover"
                                onClick={() => setHighlightedRun(highlightedRun === runKey ? null : runKey)}
                              >
                                <span className="truncate">{run.model}</span>
                                <span className="text-tertiary font-mono">#{run.run_id?.slice(-6)}</span>
                              </div>
                              {Array.from(benchmarkNames).map(name => {
                                const data = run.results?.[name]
                                const score = data ? extractScore(name, data) * 100 : 0
                                const intensity = Math.round(score)
                                const bgColor = score === 0
                                  ? 'bg-surface-active'
                                  : `bg-green-${Math.min(9, Math.floor(intensity / 11) + 1)}00`
                                const textColor = intensity > 60 ? 'text-primary' : 'text-secondary'

                                return (
                                  <div
                                    key={name}
                                    className={`w-24 h-10 rounded flex items-center justify-center text-xs font-medium ${bgColor} ${textColor} transition-transform hover:scale-105 cursor-help`}
                                    style={{
                                      backgroundColor: score > 0 ? `rgba(34, 197, 94, ${intensity / 100 * 0.8})` : undefined
                                    }}
                                    title={`${formatBenchmarkName(name)}: ${score.toFixed(1)}%`}
                                  >
                                    {score > 0 ? `${score.toFixed(0)}%` : '-'}
                                  </div>
                                )
                              })}
                              <div
                                className="w-24 h-10 rounded flex items-center justify-center text-xs font-bold bg-primary-500/20 text-primary-400"
                                title={`Average: ${avg.toFixed(1)}%`}
                              >
                                {avg.toFixed(0)}%
                              </div>
                            </div>
                          )
                        })
                      })()}
                    </div>
                  </div>
                </div>
              )}

              {/* Timeline Chart */}
              {chartType === 'line' && (
                <div className="space-y-4">
                  <p className="text-sm text-tertiary">Evolution of benchmark scores over time. Shows how performance changes across different runs.</p>
                  <ResponsiveContainer width="100%" height={450}>
                    <LineChart
                      data={selectedRunsData.map((run, idx) => {
                        const date = new Date(run.started_at)
                        const scores = {}
                        Array.from(benchmarkNames).forEach(name => {
                          const data = run.results?.[name]
                          scores[name] = data ? extractScore(name, data) * 100 : null
                        })
                        return {
                          timestamp: date.getTime(),
                          date: date.toLocaleDateString(),
                          time: date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                          model: run.model,
                          runId: run.run_id?.slice(-6),
                          avg: calculateAverage(run.results) * 100,
                          ...scores,
                        }
                      }).sort((a, b) => a.timestamp - b.timestamp)}
                      margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis
                        dataKey="date"
                        tick={{ fill: '#94a3b8', fontSize: 11 }}
                        angle={-45}
                        textAnchor="end"
                        height={80}
                      />
                      <YAxis tick={{ fill: '#94a3b8' }} domain={[0, 100]} label={{ value: 'Score (%)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1e293b',
                          border: '1px solid #475569',
                          borderRadius: '0.5rem',
                          padding: '0.75rem',
                        }}
                        labelStyle={{ color: '#f1f5f9', marginBottom: '0.5rem', fontWeight: 'bold' }}
                        formatter={(value, name) => [
                          value ? `${Number(value).toFixed(1)}%` : '-',
                          formatBenchmarkName(name)
                        ]}
                      />
                      <Legend
                        wrapperStyle={{ paddingTop: 20, cursor: 'pointer' }}
                        onClick={(e) => {
                          setHighlightedRun(highlightedRun === e.value ? null : e.value)
                        }}
                      />
                      {Array.from(benchmarkNames).map((name, i) => {
                        const isHighlighted = highlightedRun === null || highlightedRun === name
                        return (
                          <Line
                            key={name}
                            type="monotone"
                            dataKey={name}
                            stroke={colors[i % colors.length]}
                            strokeWidth={isHighlighted ? 3 : 1}
                            dot={{ r: isHighlighted ? 5 : 3 }}
                            name={formatBenchmarkName(name)}
                            connectNulls
                          />
                        )
                      })}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Scatter Plot - Trade-offs */}
              {chartType === 'scatter' && (
                <div className="space-y-4">
                  <p className="text-sm text-tertiary">Performance trade-offs visualization. Each point represents a run, showing the relationship between average score and sample size.</p>
                  <ResponsiveContainer width="100%" height={450}>
                    <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke={getCSSColor('--chart-grid')} />
                      <XAxis
                        type="number"
                        dataKey="sampleSize"
                        name="Samples"
                        tick={{ fill: getCSSColor('--chart-text') }}
                        label={{ value: 'Sample Size', position: 'insideBottom', offset: -5, fill: getCSSColor('--chart-text') }}
                      />
                      <YAxis
                        type="number"
                        dataKey="avgScore"
                        name="Avg Score"
                        tick={{ fill: getCSSColor('--chart-text') }}
                        domain={[0, 100]}
                        label={{ value: 'Average Score (%)', angle: -90, position: 'insideLeft', fill: getCSSColor('--chart-text') }}
                      />
                      <ZAxis range={[100, 400]} />
                      <Tooltip
                        cursor={{ strokeDasharray: '3 3' }}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload
                            return (
                              <div className="bg-background border border-default rounded-lg p-3 shadow-xl">
                                <p className="font-semibold text-primary mb-2">{data.model}</p>
                                <div className="space-y-1 text-sm">
                                  <p className="text-secondary">
                                    <span className="text-tertiary">ID:</span> #{data.runIdShort}
                                  </p>
                                  <p className="text-green-400">
                                    <span className="text-tertiary">Avg Score:</span> {data.avgScore.toFixed(1)}%
                                  </p>
                                  <p className="text-blue-400">
                                    <span className="text-tertiary">Sample Size:</span> {data.sampleSize}
                                  </p>
                                  <p className="text-amber-400">
                                    <span className="text-tertiary">Temperature:</span> {data.temperature}
                                  </p>
                                  <p className="text-tertiary">
                                    <span className="text-tertiary">Date:</span> {data.dateStr}
                                  </p>
                                </div>
                              </div>
                            )
                          }
                          return null
                        }}
                      />
                      <Scatter
                        name="Runs"
                        data={selectedRunsData.map((run, i) => ({
                          model: run.model,
                          runIdShort: run.run_id?.slice(-6),
                          avgScore: calculateAverage(run.results) * 100,
                          sampleSize: run.sample_size || 0,
                          temperature: run.inference_settings?.temperature ?? 0,
                          dateStr: new Date(run.started_at).toLocaleDateString(),
                          color: colors[i % colors.length],
                        }))}
                        fill="#8884d8"
                      >
                        {selectedRunsData.map((run, i) => (
                          <Cell
                            key={run.run_id}
                            fill={colors[i % colors.length]}
                            opacity={highlightedRun === null || highlightedRun === getRunKey(run) ? 1 : 0.3}
                            onMouseEnter={() => setHighlightedRun(getRunKey(run))}
                            onMouseLeave={() => setHighlightedRun(null)}
                          />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}

          {/* Summary Table */}
          {selectedRuns.length > 0 && (
            <div className="mt-6 bg-surface rounded-xl border border-default overflow-x-auto">
              <table className="w-full min-w-max">
                <thead>
                  <tr className="border-b border-default">
                    <th className="px-4 py-3 text-left text-sm font-medium text-tertiary">Model</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-tertiary">
                      <div className="flex items-center gap-1">
                        <Settings2 className="w-3 h-3" />
                        Config
                      </div>
                    </th>
                    {Array.from(benchmarkNames).map(name => (
                      <th key={name} className="px-4 py-3 text-left text-sm font-medium text-tertiary">
                        {formatBenchmarkName(name)}
                      </th>
                    ))}
                    <th className="px-4 py-3 text-left text-sm font-medium text-tertiary">Average</th>
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

                    // Get inference settings info
                    const inferenceInfo = run.inference_settings
                      ? `T=${run.inference_settings.temperature ?? 0} S=${run.inference_settings.seed ?? 42}`
                      : 'Default'

                    return (
                      <tr key={run.run_id} className="border-b border-default/50">
                        <td className="px-4 py-3">
                          <div className="text-primary font-medium">{run.model}</div>
                          <div className="text-xs text-tertiary font-mono">#{run.run_id?.slice(-6)}</div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-xs space-y-1">
                            <div className="flex items-center gap-1 text-secondary">
                              <Hash className="w-3 h-3 text-tertiary" />
                              <span>{sampleSize} samples</span>
                            </div>
                            <div className="text-tertiary capitalize">{preset}</div>
                            <div className="text-tertiary font-mono text-[10px]">{inferenceInfo}</div>
                          </div>
                        </td>
                        {Array.from(benchmarkNames).map(name => {
                          const data = run.results?.[name]
                          const score = data ? extractScore(name, data) : null
                          return (
                            <td key={name} className="px-4 py-3 text-secondary">
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

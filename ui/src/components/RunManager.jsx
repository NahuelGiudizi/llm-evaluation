import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Play,
  RefreshCw,
  CheckCircle,
  XCircle,
  Loader2,
  Settings,
  Zap,
  Server,
  Database,
  Sliders,
  ChevronDown,
  ChevronUp,
  Info,
  Plus,
  Key,
  Eye,
  EyeOff,
  Brain,
  Shield,
  BookOpen,
  Sparkles,
  AlertTriangle,
  Hash
} from 'lucide-react'
import { fetchModels, fetchBenchmarks, fetchPresets, startRun, fetchModelInfo } from '../api'
import QueueBuilder from './QueueBuilder'

// Tooltip component for inference parameters
function ParamTooltip({ children, tooltip }) {
  const [show, setShow] = useState(false)

  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="cursor-help"
      >
        {children}
      </div>
      {show && (
        <div className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 p-3 bg-surface border border-default rounded-lg shadow-xl text-xs">
          <div className="text-secondary leading-relaxed whitespace-pre-line">{tooltip}</div>
          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1">
            <div className="border-4 border-transparent border-t-slate-600"></div>
          </div>
        </div>
      )}
    </div>
  )
}

// Inference parameter descriptions
const PARAM_TOOLTIPS = {
  temperature: `🎲 Controls randomness/creativity

• 0 = Deterministic (always picks most probable)
• 0.1-0.5 = Conservative responses
• 0.7-1.0 = More creative and varied
• >1.0 = Very random, may produce incoherence

✅ For benchmarks: 0 is ideal for consistent, reproducible results.`,

  top_p: `🎯 Nucleus Sampling

Limits tokens to those whose cumulative probability sums to P.

• 1.0 = Consider all tokens (no filter)
• 0.9 = Only tokens summing 90% probability
• 0.5 = Only most probable tokens

✅ For benchmarks: 1.0 (temperature=0 handles control).`,

  top_k: `📊 Top-K Sampling

Limits to the K most probable tokens.

• -1 = Disabled (no limit)
• 50 = Only top 50 tokens
• 10 = Very restrictive

✅ For benchmarks: -1 (disabled) since we use temperature=0.`,

  max_tokens: `📝 Response Limit

Maximum tokens the model can generate.

• 128 = Very short responses
• 512 = Medium responses
• 1024 = Long responses (good for GSM8K)
• 2048+ = Extended text

✅ For benchmarks: 1024 is sufficient for most.`,

  seed: `🌱 Reproducibility Seed

Controls the random number generator.

• Same seed + same params = same response
• 42 = Classic value in data science
  (reference to "The Hitchhiker's Guide to the Galaxy")

✅ Crucial for replicating scientific results.`
}

function RunManager({ onRunStart }) {
  const navigate = useNavigate()
  const [models, setModels] = useState([])
  const [benchmarks, setBenchmarks] = useState([])
  const [presets, setPresets] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [starting, setStarting] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [modelInfo, setModelInfo] = useState(null)
  const [loadingModelInfo, setLoadingModelInfo] = useState(false)

  // Form state
  const [selectedProvider, setSelectedProvider] = useState('')
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedBenchmarks, setSelectedBenchmarks] = useState([])
  const [questionsPerDataset, setQuestionsPerDataset] = useState(10)
  const [useAllSamples, setUseAllSamples] = useState(false)
  const [selectedPreset, setSelectedPreset] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [showApiKey, setShowApiKey] = useState(false)

  // Inference settings
  const [inferenceSettings, setInferenceSettings] = useState({
    temperature: 0.0,
    top_p: 1.0,
    top_k: -1,
    max_tokens: 2048,
    seed: 42,
  })

  // Queue state
  const [queue, setQueue] = useState([])
  const [isQueueRunning, setIsQueueRunning] = useState(false)

  useEffect(() => {
    loadData()
  }, [])

  // Load model info when model changes
  useEffect(() => {
    if (selectedProvider && selectedModel) {
      loadModelInfo(selectedProvider, selectedModel)
    }
  }, [selectedProvider, selectedModel])

  async function loadModelInfo(provider, model) {
    setLoadingModelInfo(true)
    try {
      const info = await fetchModelInfo(provider, model)
      setModelInfo(info)
      // Update inference settings with model defaults
      if (info?.inference_defaults) {
        setInferenceSettings(prev => ({
          ...prev,
          ...info.inference_defaults
        }))
      }
    } catch (err) {
      console.log('Could not load model info:', err)
      setModelInfo(null)
    } finally {
      setLoadingModelInfo(false)
    }
  }

  async function loadData() {
    setLoading(true)
    setError(null)
    try {
      const [modelsData, benchmarksData, presetsData] = await Promise.all([
        fetchModels(),
        fetchBenchmarks(),
        fetchPresets(),
      ])
      setModels(modelsData)
      setBenchmarks(benchmarksData)
      setPresets(presetsData)

      // Set defaults
      if (modelsData.length > 0) {
        setSelectedProvider(modelsData[0].provider)
        if (modelsData[0].models?.length > 0) {
          setSelectedModel(modelsData[0].models[0])
        }
      }
      if (benchmarksData.length > 0) {
        setSelectedBenchmarks([benchmarksData[0].id])
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Listen for API keys updates and reload models
  useEffect(() => {
    const handleApiKeysUpdate = () => {
      console.log('API keys updated, reloading providers...')
      loadData()
    }
    window.addEventListener('apiKeysUpdated', handleApiKeysUpdate)
    return () => window.removeEventListener('apiKeysUpdated', handleApiKeysUpdate)
  }, [])

  function handlePresetChange(presetId) {
    setSelectedPreset(presetId)
    const preset = presets.find(p => p.id === presetId)
    if (preset) {
      setSelectedBenchmarks(preset.benchmarks)
      if (preset.questions_per_dataset === null) {
        setUseAllSamples(true)
      } else {
        setUseAllSamples(false)
        setQuestionsPerDataset(preset.questions_per_dataset)
      }
    }
  }

  async function handleStartRun() {
    if (!selectedProvider || !selectedModel || selectedBenchmarks.length === 0) {
      setError('Please select a provider, model, and at least one benchmark')
      return
    }

    // If queue is already running, add to queue instead
    if (isQueueRunning) {
      handleAddToQueue()
      return
    }

    // If there are items in the queue, start the queue with this as first item
    if (queue.length > 0) {
      const newItem = {
        model: selectedModel,
        provider: selectedProvider,
        benchmarks: selectedBenchmarks,
        sample_size: useAllSamples ? null : questionsPerDataset,
        inference_settings: { ...inferenceSettings },
        base_url: baseUrl || undefined,
        api_key: apiKey || undefined,
      }
      setQueue([newItem, ...queue])
      // Start queue will be triggered by user clicking Start Queue
      return
    }

    setStarting(true)
    setError(null)
    try {
      const config = {
        provider: selectedProvider,
        model: selectedModel,
        benchmarks: selectedBenchmarks,
        questions_per_dataset: useAllSamples ? null : questionsPerDataset,
        inference_settings: inferenceSettings,
        model_config: modelInfo,
        base_url: baseUrl || undefined,
        api_key: apiKey || undefined,
      }
      const result = await startRun(config)
      onRunStart(result.run_id)
      navigate('/progress')
    } catch (err) {
      setError(err.message)
    } finally {
      setStarting(false)
    }
  }

  function handleAddToQueue() {
    if (!selectedProvider || !selectedModel || selectedBenchmarks.length === 0) {
      setError('Please select a provider, model, and at least one benchmark')
      return
    }

    const queueItem = {
      model: selectedModel,
      provider: selectedProvider,
      benchmarks: [...selectedBenchmarks],
      sample_size: useAllSamples ? null : questionsPerDataset,
      inference_settings: { ...inferenceSettings },
      base_url: baseUrl || undefined,
      api_key: apiKey || undefined,
    }

    setQueue([...queue, queueItem])
    setError(null)
  }

  function handleQueueStart() {
    // Queue started - QueueBuilder handles its own running state
    // Clear the queue items (they're now in the server queue)
    setQueue([])
    // Don't navigate - QueueBuilder shows progress inline
  }

  const selectedProviderModels = models.find(m => m.provider === selectedProvider)?.models || []

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 bg-gradient-to-br from-amber-500 to-orange-500 rounded-xl flex items-center justify-center shadow-lg shadow-amber-500/25">
            <Play className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-primary">Run Evaluation</h1>
            <p className="text-sm text-secondary">Configure and benchmark your LLM models</p>
          </div>
        </div>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-error-500/10 border border-error-500/30 rounded-xl flex items-center gap-3 animate-fade-in">
          <XCircle className="w-5 h-5 text-error-400" />
          <span className="text-error-400 flex-1">{error}</span>
          <button onClick={() => setError(null)} className="text-error-400 hover:text-error-300 transition-colors">
            <XCircle className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Quick Presets */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-primary mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-400" />
          Evaluation Presets
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {presets.map(preset => {
            // Icon based on category
            const PresetIcon = preset.category === 'Speed' ? Zap :
              preset.category === 'Knowledge' ? BookOpen :
                preset.category === 'Reasoning' ? Brain :
                  preset.category === 'Safety' ? Shield :
                    preset.category === 'Complete' ? Sparkles : Zap

            const iconColor = preset.category === 'Speed' ? 'text-yellow-400' :
              preset.category === 'Knowledge' ? 'text-blue-400' :
                preset.category === 'Reasoning' ? 'text-purple-400' :
                  preset.category === 'Safety' ? 'text-green-400' :
                    preset.category === 'Complete' ? 'text-primary-400' : 'text-tertiary'

            return (
              <button
                key={preset.id}
                onClick={() => handlePresetChange(preset.id)}
                className={`p-4 rounded-lg border text-left transition-all card-hover ${selectedPreset === preset.id
                  ? 'bg-primary-500/20 border-primary-500/50'
                  : 'bg-surface border-default hover:border-interactive'
                  }`}
              >
                <div className="flex items-center gap-2 mb-2">
                  <PresetIcon className={`w-5 h-5 ${iconColor}`} />
                  <h3 className="font-semibold text-primary">{preset.name}</h3>
                  <span className="ml-auto text-xs text-tertiary bg-surface-active px-2 py-0.5 rounded">
                    {preset.questions_per_dataset || 'All'} samples
                  </span>
                </div>
                <p className="text-sm text-tertiary mb-3">{preset.description}</p>
                <div className="flex gap-1.5 flex-wrap">
                  {preset.benchmarks.map(b => (
                    <span key={b} className="px-2 py-0.5 bg-surface-hover rounded text-xs text-secondary capitalize">{b === 'mmlu' ? 'MMLU' :
                      b === 'truthfulqa' ? 'TruthfulQA' :
                        b === 'hellaswag' ? 'HellaSwag' :
                          b === 'arc' ? 'ARC' :
                            b === 'winogrande' ? 'WinoGrande' :
                              b === 'commonsenseqa' ? 'CommonsenseQA' :
                                b === 'boolq' ? 'BoolQ' :
                                  b === 'safetybench' ? 'SafetyBench' :
                                    b === 'donotanswer' ? 'Do-Not-Answer' : b}
                    </span>
                  ))}
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {/* Configuration */}
      <div className="bg-surface rounded-xl border border-default p-6 mb-6">
        <h2 className="text-lg font-semibold text-primary mb-6 flex items-center gap-2">
          <Settings className="w-5 h-5 text-tertiary" />
          Configuration
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Provider */}
          <div>
            <label className="block text-sm font-medium text-secondary mb-2">
              <Server className="w-4 h-4 inline mr-2" />
              Provider
            </label>
            <select
              value={selectedProvider}
              onChange={(e) => {
                setSelectedProvider(e.target.value)
                const providerModels = models.find(m => m.provider === e.target.value)?.models || []
                setSelectedModel(providerModels[0] || '')
              }}
              className="w-full bg-background border border-default rounded-lg px-4 py-2.5 text-primary focus:outline-none focus:ring-2 focus:ring-interactive"
            >
              {models.map(m => (
                <option key={m.provider} value={m.provider}>{m.provider}</option>
              ))}
            </select>
          </div>

          {/* Model */}
          <div>
            <label className="block text-sm font-medium text-secondary mb-2">
              <Database className="w-4 h-4 inline mr-2" />
              Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-background border border-default rounded-lg px-4 py-2.5 text-primary focus:outline-none focus:ring-2 focus:ring-interactive"
            >
              {selectedProviderModels.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>

          {/* Base URL for custom providers */}
          {(selectedProvider === 'openai' || selectedProvider === 'ollama') && (
            <div>
              <label className="block text-sm font-medium text-secondary dark:text-secondary mb-2">
                <Server className="w-4 h-4 inline mr-2" />
                Custom Base URL
                <span className="ml-2 text-xs text-tertiary">(optional)</span>
              </label>
              <input
                type="text"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder={selectedProvider === 'ollama' ? 'http://localhost:11434' : 'https://api.openai.com/v1'}
                className="w-full bg-surface dark:bg-background border border-default dark:border-default rounded-lg px-4 py-2.5 text-slate-900 dark:text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 placeholder-slate-400 dark:placeholder-slate-500"
              />
              <p className="mt-1 text-xs text-tertiary">
                For OpenAI-compatible APIs: vLLM, LM Studio, Together.ai, Azure OpenAI, etc.
              </p>
            </div>
          )}

          {/* API Keys Management Button */}
          <div className={selectedProvider === 'openai' || selectedProvider === 'ollama' ? '' : 'md:col-span-2'}>
            <label className="block text-sm font-medium text-secondary dark:text-secondary mb-2">
              <Key className="w-4 h-4 inline mr-2" />
              API Keys
            </label>
            <button
              type="button"
              onClick={() => window.dispatchEvent(new CustomEvent('openApiKeys'))}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 text-sm text-secondary hover:text-primary bg-surface hover:bg-surface-hover border border-default rounded-lg transition-colors"
            >
              <Key className="w-4 h-4" />
              <span>Manage API Keys</span>
              <span className="ml-auto text-xs text-tertiary">Configure provider API keys</span>
            </button>
          </div>

          {/* API Key for cloud providers */}
          {(selectedProvider === 'openai' || selectedProvider === 'anthropic' || selectedProvider === 'deepseek') && (
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-secondary dark:text-secondary mb-2">
                <Key className="w-4 h-4 inline mr-2" />
                API Key
                <span className="ml-2 text-xs text-tertiary">(or set via environment variable)</span>
              </label>
              <div className="relative">
                <input
                  type={showApiKey ? 'text' : 'password'}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder={
                    selectedProvider === 'openai' ? 'sk-... or leave empty to use OPENAI_API_KEY env var' :
                      selectedProvider === 'anthropic' ? 'sk-ant-... or leave empty to use ANTHROPIC_API_KEY env var' :
                        'Leave empty to use DEEPSEEK_API_KEY env var'
                  }
                  className="w-full bg-surface dark:bg-background border border-default dark:border-default rounded-lg px-4 py-2.5 pr-12 text-slate-900 dark:text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 placeholder-slate-400 dark:placeholder-slate-500"
                />
                <button
                  type="button"
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-tertiary hover:text-primary transition-colors"
                >
                  {showApiKey ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
              <p className="mt-1 text-xs text-tertiary">
                {selectedProvider === 'openai' && '💡 Tip: Set OPENAI_API_KEY environment variable before starting the dashboard'}
                {selectedProvider === 'anthropic' && '💡 Tip: Set ANTHROPIC_API_KEY environment variable before starting the dashboard'}
                {selectedProvider === 'deepseek' && '💡 Tip: Set DEEPSEEK_API_KEY environment variable before starting the dashboard'}
              </p>
            </div>
          )}

          {/* Questions per dataset */}
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-secondary mb-2">
              Questions per Dataset
            </label>
            <div className="flex gap-4 items-start">
              <div className="flex-1">
                <input
                  type="number"
                  min="1"
                  max="15000"
                  value={questionsPerDataset}
                  onChange={(e) => setQuestionsPerDataset(parseInt(e.target.value) || 10)}
                  disabled={useAllSamples}
                  className={`w-full bg-background border border-default rounded-lg px-4 py-2.5 text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 ${useAllSamples ? 'opacity-50 cursor-not-allowed' : ''
                    }`}
                />
              </div>
              <label className="flex items-center gap-2 bg-surface dark:bg-background border border-default dark:border-default rounded-lg px-4 py-2.5 cursor-pointer hover:border-slate-400 dark:hover:border-slate-500 transition-colors">
                <input
                  type="checkbox"
                  checked={useAllSamples}
                  onChange={(e) => setUseAllSamples(e.target.checked)}
                  className="w-4 h-4 rounded border-default dark:border-default bg-surface dark:bg-background text-primary-500 focus:ring-primary-500"
                />
                <span className="text-sm text-secondary dark:text-secondary whitespace-nowrap">Use all available</span>
              </label>
            </div>
            {useAllSamples && (
              <div className="mt-2 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
                <p className="text-xs text-amber-300">
                  <strong>Warning:</strong> Using all samples can take several hours per benchmark.
                  MMLU has 14K questions, HellaSwag 10K, etc. Consider using a subset for faster results.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Benchmarks */}
        <div className="mt-6">
          <label className="block text-sm font-medium text-secondary dark:text-secondary mb-4 flex items-center gap-2">
            <BookOpen className="w-4 h-4" />
            Select Benchmarks
          </label>

          {/* Knowledge Benchmarks */}
          <div className="mb-6">
            <div className="flex items-center gap-2 mb-3">
              <Brain className="w-4 h-4 text-primary-400" />
              <span className="text-sm font-medium text-secondary dark:text-tertiary">Knowledge & Reasoning</span>
              <span className="text-xs text-tertiary">
                ({benchmarks.filter(b => !['safetybench', 'donotanswer'].includes(b.id)).length} benchmarks)
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {benchmarks.filter(b => !['safetybench', 'donotanswer'].includes(b.id)).map(benchmark => {
                const icons = {
                  mmlu: '📚',
                  truthfulqa: '✓',
                  hellaswag: '💭',
                  arc: '🔬',
                  winogrande: '🧩',
                  commonsenseqa: '💡',
                  boolq: '❓',
                  gsm8k: '🔢',
                }
                const descriptions = {
                  mmlu: '57 subjects knowledge',
                  truthfulqa: 'Truthfulness evaluation',
                  hellaswag: 'Common sense reasoning',
                  arc: 'Science reasoning',
                  winogrande: 'Pronoun resolution',
                  commonsenseqa: 'Commonsense knowledge',
                  boolq: 'Yes/No comprehension',
                  gsm8k: 'Grade school math',
                }
                const isSelected = selectedBenchmarks.includes(benchmark.id)

                return (
                  <button
                    key={benchmark.id}
                    onClick={() => {
                      if (isSelected) {
                        setSelectedBenchmarks(selectedBenchmarks.filter(b => b !== benchmark.id))
                      } else {
                        setSelectedBenchmarks([...selectedBenchmarks, benchmark.id])
                      }
                    }}
                    className={`
                      p-4 rounded-xl text-left transition-all duration-200
                      ${isSelected
                        ? 'border-[3px] border-primary-500 bg-primary-500/15 shadow-lg shadow-primary-500/20 scale-[1.02] ring-2 ring-primary-500/20'
                        : 'border-2 border-default hover:border-slate-400 dark:hover:border-slate-500 bg-surface/50 hover:bg-surface'
                      }
                    `}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-lg">{icons[benchmark.id] || '📋'}</span>
                        <span className={`font-medium transition-colors ${isSelected ? 'text-primary-600 dark:text-primary-400' : 'text-primary'}`}>{benchmark.name}</span>
                      </div>
                      {isSelected && (
                        <div className="w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all duration-200 border-primary-500 bg-primary-500 shadow-md shadow-primary-500/30">
                          <CheckCircle className="w-4 h-4 text-white" />
                        </div>
                      )}
                    </div>
                    <p className="text-xs text-tertiary mt-1.5">
                      {descriptions[benchmark.id] || benchmark.description}
                    </p>
                    <div className="flex items-center gap-2 mt-2">
                      <span className="text-xs text-tertiary font-mono flex items-center gap-1">
                        <Hash className="w-3 h-3" />
                        {benchmark.questions_count >= 1000
                          ? `${(benchmark.questions_count / 1000).toFixed(1)}K`
                          : benchmark.questions_count || '?'} questions
                      </span>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Safety Benchmarks */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Shield className="w-4 h-4 text-emerald-400" />
              <span className="text-sm font-medium text-tertiary">Safety & Security</span>
              <span className="text-xs text-tertiary">
                ({benchmarks.filter(b => ['safetybench', 'donotanswer'].includes(b.id)).length} benchmarks)
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {benchmarks.filter(b => ['safetybench', 'donotanswer'].includes(b.id)).map(benchmark => {
                const icons = {
                  safetybench: '🛡️',
                  donotanswer: '🚫',
                }
                const descriptions = {
                  safetybench: 'Safety scenario evaluation',
                  donotanswer: 'Harmful prompt rejection',
                }
                const isSelected = selectedBenchmarks.includes(benchmark.id)

                return (
                  <button
                    key={benchmark.id}
                    onClick={() => {
                      if (isSelected) {
                        setSelectedBenchmarks(selectedBenchmarks.filter(b => b !== benchmark.id))
                      } else {
                        setSelectedBenchmarks([...selectedBenchmarks, benchmark.id])
                      }
                    }}
                    className={`
                      p-4 rounded-xl border-2 text-left transition-all
                      ${isSelected
                        ? 'border-emerald-500 bg-emerald-500/10 shadow-lg shadow-emerald-500/10'
                        : 'border-default hover:border-slate-500 bg-surface/50'
                      }
                    `}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-lg">{icons[benchmark.id] || '📋'}</span>
                        <span className="font-medium text-primary">{benchmark.name}</span>
                      </div>
                      <div className={`
                        w-5 h-5 rounded-full border-2 flex items-center justify-center transition-all
                        ${isSelected
                          ? 'border-emerald-500 bg-emerald-500'
                          : 'border-default'
                        }
                      `}>
                        {isSelected && <CheckCircle className="w-3 h-3 text-primary" />}
                      </div>
                    </div>
                    <p className="text-xs text-tertiary mt-1.5">
                      {descriptions[benchmark.id] || benchmark.description}
                    </p>
                    <div className="flex items-center gap-2 mt-2">
                      <span className="text-xs text-tertiary font-mono flex items-center gap-1">
                        <Hash className="w-3 h-3" />
                        {benchmark.questions_count >= 1000
                          ? `${(benchmark.questions_count / 1000).toFixed(1)}K`
                          : benchmark.questions_count || '?'} questions
                      </span>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Quick select buttons */}
          <div className="mt-4 flex gap-2">
            <button
              onClick={() => setSelectedBenchmarks(benchmarks.map(b => b.id))}
              className="text-xs px-3 py-1.5 rounded-lg bg-surface-active text-secondary hover:bg-surface-hover transition-colors"
            >
              Select All
            </button>
            <button
              onClick={() => setSelectedBenchmarks([])}
              className="text-xs px-3 py-1.5 rounded-lg bg-surface-active text-secondary hover:bg-surface-hover transition-colors"
            >
              Clear All
            </button>
            <button
              onClick={() => setSelectedBenchmarks(benchmarks.filter(b => !['safetybench', 'donotanswer'].includes(b.id)).map(b => b.id))}
              className="text-xs px-3 py-1.5 rounded-lg bg-primary-500/20 text-primary-400 hover:bg-primary-500/30 transition-colors"
            >
              Knowledge Only
            </button>
            <button
              onClick={() => setSelectedBenchmarks(['safetybench', 'donotanswer'])}
              className="text-xs px-3 py-1.5 rounded-lg bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 transition-colors"
            >
              Safety Only
            </button>
          </div>
        </div>
      </div>

      {/* Model Info Panel */}
      {modelInfo && (
        <div className="bg-surface rounded-xl border border-default p-6 mb-6">
          <h2 className="text-lg font-semibold text-primary mb-4 flex items-center gap-2">
            <Info className="w-5 h-5 text-info" />
            Model Information
            {loadingModelInfo && <Loader2 className="w-4 h-4 animate-spin text-tertiary" />}
          </h2>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {modelInfo.architecture?.parameter_count && (
              <div className="bg-surface-active rounded-lg p-3">
                <div className="text-xs text-tertiary mb-1">Parameters</div>
                <div className="text-lg font-semibold text-primary">{modelInfo.architecture.parameter_count}</div>
              </div>
            )}
            {modelInfo.context?.context_length && (
              <div className="bg-surface-active rounded-lg p-3">
                <div className="text-xs text-tertiary mb-1">Context Window</div>
                <div className="text-lg font-semibold text-primary">{modelInfo.context.context_length.toLocaleString()}</div>
              </div>
            )}
            {modelInfo.architecture?.quantization && (
              <div className="bg-surface-active rounded-lg p-3">
                <div className="text-xs text-tertiary mb-1">Quantization</div>
                <div className="text-lg font-semibold text-primary">{modelInfo.architecture.quantization}</div>
              </div>
            )}
            {modelInfo.architecture?.family && (
              <div className="bg-surface-active rounded-lg p-3">
                <div className="text-xs text-tertiary mb-1">Family</div>
                <div className="text-lg font-semibold text-primary">{modelInfo.architecture.family}</div>
              </div>
            )}
            {modelInfo.architecture?.format && (
              <div className="bg-surface-active rounded-lg p-3">
                <div className="text-xs text-tertiary mb-1">Format</div>
                <div className="text-lg font-semibold text-primary">{modelInfo.architecture.format}</div>
              </div>
            )}
            {modelInfo.architecture?.embedding_length && (
              <div className="bg-surface-active rounded-lg p-3">
                <div className="text-xs text-tertiary mb-1">Embedding Dim</div>
                <div className="text-lg font-semibold text-primary">{modelInfo.architecture.embedding_length.toLocaleString()}</div>
              </div>
            )}
            {modelInfo.architecture?.attention_heads && (
              <div className="bg-surface-active rounded-lg p-3">
                <div className="text-xs text-tertiary mb-1">Attention Heads</div>
                <div className="text-lg font-semibold text-primary">{modelInfo.architecture.attention_heads}</div>
              </div>
            )}
            {modelInfo.architecture?.layers && (
              <div className="bg-surface-active rounded-lg p-3">
                <div className="text-xs text-tertiary mb-1">Layers</div>
                <div className="text-lg font-semibold text-primary">{modelInfo.architecture.layers}</div>
              </div>
            )}
            {modelInfo.architecture?.vocab_size && (
              <div className="bg-surface-active rounded-lg p-3">
                <div className="text-xs text-tertiary mb-1">Vocab Size</div>
                <div className="text-lg font-semibold text-primary">{modelInfo.architecture.vocab_size.toLocaleString()}</div>
              </div>
            )}
          </div>

          {/* Show message if no architecture info available */}
          {!modelInfo.architecture && !modelInfo.error && (
            <p className="text-tertiary text-sm">No detailed model information available</p>
          )}
          {modelInfo.error && (
            <p className="text-error text-sm">Could not load model info: {modelInfo.error}</p>
          )}
        </div>
      )}

      {/* Advanced Settings (Inference Parameters) */}
      <div className="bg-surface rounded-xl border border-default mb-6">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-surface-hover rounded-xl transition-colors"
        >
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-accent" />
            <span className="font-semibold text-primary">Inference Parameters</span>
            <span className="text-xs text-tertiary">(for reproducibility)</span>
          </div>
          {showAdvanced ? (
            <ChevronUp className="w-5 h-5 text-tertiary" />
          ) : (
            <ChevronDown className="w-5 h-5 text-tertiary" />
          )}
        </button>

        {showAdvanced && (
          <div className="p-6 pt-2 border-t border-default">
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div>
                <label className="flex items-center gap-1.5 text-xs text-tertiary mb-1">
                  Temperature
                  <ParamTooltip tooltip={PARAM_TOOLTIPS.temperature}>
                    <Info className="w-3.5 h-3.5 text-tertiary hover:text-interactive transition-colors" />
                  </ParamTooltip>
                </label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="2"
                  value={inferenceSettings.temperature}
                  onChange={(e) => setInferenceSettings({ ...inferenceSettings, temperature: parseFloat(e.target.value) })}
                  className="w-full bg-background border border-default rounded-lg px-3 py-2 text-primary text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <div className="text-xs text-tertiary mt-1">0 = deterministic</div>
              </div>

              <div>
                <label className="flex items-center gap-1.5 text-xs text-tertiary mb-1">
                  Top P
                  <ParamTooltip tooltip={PARAM_TOOLTIPS.top_p}>
                    <Info className="w-3.5 h-3.5 text-tertiary hover:text-primary-400 transition-colors" />
                  </ParamTooltip>
                </label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  value={inferenceSettings.top_p}
                  onChange={(e) => setInferenceSettings({ ...inferenceSettings, top_p: parseFloat(e.target.value) })}
                  className="w-full bg-background border border-default rounded-lg px-3 py-2 text-primary text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <div className="text-xs text-tertiary mt-1">Nucleus sampling</div>
              </div>

              <div>
                <label className="flex items-center gap-1.5 text-xs text-tertiary mb-1">
                  Top K
                  <ParamTooltip tooltip={PARAM_TOOLTIPS.top_k}>
                    <Info className="w-3.5 h-3.5 text-tertiary hover:text-primary-400 transition-colors" />
                  </ParamTooltip>
                </label>
                <input
                  type="number"
                  min="-1"
                  max="100"
                  value={inferenceSettings.top_k}
                  onChange={(e) => setInferenceSettings({ ...inferenceSettings, top_k: parseInt(e.target.value) })}
                  className="w-full bg-background border border-default rounded-lg px-3 py-2 text-primary text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <div className="text-xs text-tertiary mt-1">-1 = disabled</div>
              </div>

              <div>
                <label className="flex items-center gap-1.5 text-xs text-tertiary mb-1">
                  Max Tokens
                  <ParamTooltip tooltip={PARAM_TOOLTIPS.max_tokens}>
                    <Info className="w-3.5 h-3.5 text-tertiary hover:text-primary-400 transition-colors" />
                  </ParamTooltip>
                </label>
                <input
                  type="number"
                  min="1"
                  max="32768"
                  value={inferenceSettings.max_tokens}
                  onChange={(e) => setInferenceSettings({ ...inferenceSettings, max_tokens: parseInt(e.target.value) })}
                  className="w-full bg-background border border-default rounded-lg px-3 py-2 text-primary text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <div className="text-xs text-tertiary mt-1">Response limit</div>
              </div>

              <div>
                <label className="flex items-center gap-1.5 text-xs text-tertiary mb-1">
                  Seed
                  <ParamTooltip tooltip={PARAM_TOOLTIPS.seed}>
                    <Info className="w-3.5 h-3.5 text-tertiary hover:text-primary-400 transition-colors" />
                  </ParamTooltip>
                </label>
                <input
                  type="number"
                  min="0"
                  value={inferenceSettings.seed}
                  onChange={(e) => setInferenceSettings({ ...inferenceSettings, seed: parseInt(e.target.value) })}
                  className="w-full bg-background border border-default rounded-lg px-3 py-2 text-primary text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <div className="text-xs text-tertiary mt-1">Reproducibility</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Queue Builder */}
      <QueueBuilder
        queue={queue}
        onQueueChange={setQueue}
        onStartQueue={handleQueueStart}
        onRunningChange={setIsQueueRunning}
      />

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row justify-between items-center gap-4 mt-8 pt-6 border-t border-surface-700">
        <div className="text-sm text-tertiary">
          {selectedBenchmarks.length > 0 ? (
            <span className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-primary-400" />
              {selectedBenchmarks.length} benchmark{selectedBenchmarks.length > 1 ? 's' : ''} selected
              {!useAllSamples && ` • ${questionsPerDataset} samples each`}
            </span>
          ) : (
            <span className="text-tertiary">Select at least one benchmark to continue</span>
          )}
        </div>

        <div className="flex gap-3">
          <button
            onClick={loadData}
            className="btn btn-secondary"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
          <button
            onClick={handleAddToQueue}
            disabled={!selectedModel || selectedBenchmarks.length === 0}
            className="btn btn-ghost border-2 border-interactive text-interactive hover:bg-interactive-bg"
          >
            <Plus className="w-5 h-5" />
            Add to Queue
          </button>
          {!isQueueRunning && (
            <button
              onClick={handleStartRun}
              disabled={starting || !selectedModel || selectedBenchmarks.length === 0}
              className="btn btn-accent px-6"
            >
              {starting ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Starting...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Run Now
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default RunManager

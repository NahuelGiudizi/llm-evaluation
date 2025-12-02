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
      benchmarks: selectedBenchmarks,
      sample_size: useAllSamples ? null : questionsPerDataset,
      inference_settings: inferenceSettings,
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
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Run Evaluation</h1>
        <p className="text-slate-400">Configure and start a new LLM benchmark evaluation</p>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-3">
          <XCircle className="w-5 h-5 text-red-400" />
          <span className="text-red-400">{error}</span>
          <button onClick={() => setError(null)} className="ml-auto text-red-400 hover:text-red-300">
            <XCircle className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Quick Presets */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
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
                             preset.category === 'Complete' ? 'text-primary-400' : 'text-slate-400'

            return (
              <button
                key={preset.id}
                onClick={() => handlePresetChange(preset.id)}
                className={`p-4 rounded-lg border text-left transition-all card-hover ${selectedPreset === preset.id
                  ? 'bg-primary-500/20 border-primary-500/50'
                  : 'bg-slate-800 border-slate-700 hover:border-slate-600'
                  }`}
              >
                <div className="flex items-center gap-2 mb-2">
                  <PresetIcon className={`w-5 h-5 ${iconColor}`} />
                  <h3 className="font-semibold text-white">{preset.name}</h3>
                  <span className="ml-auto text-xs text-slate-500 bg-slate-700 px-2 py-0.5 rounded">
                    {preset.questions_per_dataset || 'All'} samples
                  </span>
                </div>
                <p className="text-sm text-slate-400 mb-3">{preset.description}</p>
                <div className="flex gap-1.5 flex-wrap">
                  {preset.benchmarks.map(b => (
                    <span key={b} className="px-2 py-0.5 bg-slate-700/50 rounded text-xs text-slate-300 capitalize">
                      {b === 'mmlu' ? 'MMLU' : 
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
      <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 mb-6">
        <h2 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
          <Settings className="w-5 h-5 text-slate-400" />
          Configuration
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Provider */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
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
              className="w-full bg-slate-900 border border-slate-600 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              {models.map(m => (
                <option key={m.provider} value={m.provider}>{m.provider}</option>
              ))}
            </select>
          </div>

          {/* Model */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              <Database className="w-4 h-4 inline mr-2" />
              Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-slate-900 border border-slate-600 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              {selectedProviderModels.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>

          {/* Base URL for custom providers */}
          {(selectedProvider === 'openai' || selectedProvider === 'ollama') && (
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-slate-300 mb-2">
                <Server className="w-4 h-4 inline mr-2" />
                Custom Base URL
                <span className="ml-2 text-xs text-slate-500">(optional)</span>
              </label>
              <input
                type="text"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder={selectedProvider === 'ollama' ? 'http://localhost:11434' : 'https://api.openai.com/v1'}
                className="w-full bg-slate-900 border border-slate-600 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-primary-500 placeholder-slate-500"
              />
              <p className="mt-1 text-xs text-slate-500">
                For OpenAI-compatible APIs: vLLM, LM Studio, Together.ai, Azure OpenAI, etc.
              </p>
            </div>
          )}

          {/* API Key for cloud providers */}
          {(selectedProvider === 'openai' || selectedProvider === 'anthropic' || selectedProvider === 'deepseek') && (
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-slate-300 mb-2">
                <Key className="w-4 h-4 inline mr-2" />
                API Key
                <span className="ml-2 text-xs text-slate-500">(or set via environment variable)</span>
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
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg px-4 py-2.5 pr-12 text-white focus:outline-none focus:ring-2 focus:ring-primary-500 placeholder-slate-500"
                />
                <button
                  type="button"
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white transition-colors"
                >
                  {showApiKey ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
              <p className="mt-1 text-xs text-slate-500">
                {selectedProvider === 'openai' && '💡 Tip: Set OPENAI_API_KEY environment variable before starting the dashboard'}
                {selectedProvider === 'anthropic' && '💡 Tip: Set ANTHROPIC_API_KEY environment variable before starting the dashboard'}
                {selectedProvider === 'deepseek' && '💡 Tip: Set DEEPSEEK_API_KEY environment variable before starting the dashboard'}
              </p>
            </div>
          )}

          {/* Questions per dataset */}
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-slate-300 mb-2">
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
                  className={`w-full bg-slate-900 border border-slate-600 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                    useAllSamples ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                />
              </div>
              <label className="flex items-center gap-2 bg-slate-900 border border-slate-600 rounded-lg px-4 py-2.5 cursor-pointer hover:border-slate-500 transition-colors">
                <input
                  type="checkbox"
                  checked={useAllSamples}
                  onChange={(e) => setUseAllSamples(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-600 bg-slate-900 text-primary-500 focus:ring-primary-500"
                />
                <span className="text-sm text-slate-300 whitespace-nowrap">Use all available</span>
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
          <label className="block text-sm font-medium text-slate-300 mb-3">
            Benchmarks
          </label>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {benchmarks.map(benchmark => (
              <label
                key={benchmark.id}
                className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${selectedBenchmarks.includes(benchmark.id)
                  ? 'bg-primary-500/20 border-primary-500/50'
                  : 'bg-slate-900 border-slate-700 hover:border-slate-600'
                  }`}
              >
                <input
                  type="checkbox"
                  checked={selectedBenchmarks.includes(benchmark.id)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedBenchmarks([...selectedBenchmarks, benchmark.id])
                    } else {
                      setSelectedBenchmarks(selectedBenchmarks.filter(b => b !== benchmark.id))
                    }
                  }}
                  className="w-4 h-4 mt-1 rounded border-slate-600 bg-slate-900 text-primary-500 focus:ring-primary-500"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium text-white">{benchmark.name}</span>
                    {benchmark.questions_count > 0 && (
                      <span className="text-xs text-slate-500 bg-slate-800 px-2 py-0.5 rounded flex items-center gap-1">
                        <Hash className="w-3 h-3" />
                        {benchmark.questions_count >= 1000 
                          ? `${(benchmark.questions_count / 1000).toFixed(1)}K` 
                          : benchmark.questions_count}
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-slate-400 mt-0.5">{benchmark.description}</div>
                </div>
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Model Info Panel */}
      {modelInfo && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 mb-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Info className="w-5 h-5 text-blue-400" />
            Model Information
            {loadingModelInfo && <Loader2 className="w-4 h-4 animate-spin text-slate-400" />}
          </h2>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {modelInfo.architecture?.parameter_count && (
              <div className="bg-slate-900 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Parameters</div>
                <div className="text-lg font-semibold text-white">{modelInfo.architecture.parameter_count}</div>
              </div>
            )}
            {modelInfo.context?.context_length && (
              <div className="bg-slate-900 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Context Window</div>
                <div className="text-lg font-semibold text-white">{modelInfo.context.context_length.toLocaleString()}</div>
              </div>
            )}
            {modelInfo.architecture?.quantization && (
              <div className="bg-slate-900 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Quantization</div>
                <div className="text-lg font-semibold text-white">{modelInfo.architecture.quantization}</div>
              </div>
            )}
            {modelInfo.architecture?.family && (
              <div className="bg-slate-900 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Family</div>
                <div className="text-lg font-semibold text-white">{modelInfo.architecture.family}</div>
              </div>
            )}
            {modelInfo.architecture?.format && (
              <div className="bg-slate-900 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Format</div>
                <div className="text-lg font-semibold text-white">{modelInfo.architecture.format}</div>
              </div>
            )}
            {modelInfo.architecture?.embedding_length && (
              <div className="bg-slate-900 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Embedding Dim</div>
                <div className="text-lg font-semibold text-white">{modelInfo.architecture.embedding_length.toLocaleString()}</div>
              </div>
            )}
            {modelInfo.architecture?.attention_heads && (
              <div className="bg-slate-900 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Attention Heads</div>
                <div className="text-lg font-semibold text-white">{modelInfo.architecture.attention_heads}</div>
              </div>
            )}
            {modelInfo.architecture?.layers && (
              <div className="bg-slate-900 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Layers</div>
                <div className="text-lg font-semibold text-white">{modelInfo.architecture.layers}</div>
              </div>
            )}
            {modelInfo.architecture?.vocab_size && (
              <div className="bg-slate-900 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Vocab Size</div>
                <div className="text-lg font-semibold text-white">{modelInfo.architecture.vocab_size.toLocaleString()}</div>
              </div>
            )}
          </div>

          {/* Show message if no architecture info available */}
          {!modelInfo.architecture && !modelInfo.error && (
            <p className="text-slate-400 text-sm">No detailed model information available</p>
          )}
          {modelInfo.error && (
            <p className="text-red-400 text-sm">Could not load model info: {modelInfo.error}</p>
          )}
        </div>
      )}

      {/* Advanced Settings (Inference Parameters) */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 mb-6">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-slate-700/50 rounded-xl transition-colors"
        >
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-orange-400" />
            <span className="font-semibold text-white">Inference Parameters</span>
            <span className="text-xs text-slate-400">(for reproducibility)</span>
          </div>
          {showAdvanced ? (
            <ChevronUp className="w-5 h-5 text-slate-400" />
          ) : (
            <ChevronDown className="w-5 h-5 text-slate-400" />
          )}
        </button>

        {showAdvanced && (
          <div className="p-6 pt-2 border-t border-slate-700">
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div>
                <label className="block text-xs text-slate-400 mb-1">Temperature</label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="2"
                  value={inferenceSettings.temperature}
                  onChange={(e) => setInferenceSettings({ ...inferenceSettings, temperature: parseFloat(e.target.value) })}
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <div className="text-xs text-slate-500 mt-1">0 = deterministic</div>
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-1">Top P</label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  value={inferenceSettings.top_p}
                  onChange={(e) => setInferenceSettings({ ...inferenceSettings, top_p: parseFloat(e.target.value) })}
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <div className="text-xs text-slate-500 mt-1">Nucleus sampling</div>
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-1">Top K</label>
                <input
                  type="number"
                  min="-1"
                  max="100"
                  value={inferenceSettings.top_k}
                  onChange={(e) => setInferenceSettings({ ...inferenceSettings, top_k: parseInt(e.target.value) })}
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <div className="text-xs text-slate-500 mt-1">-1 = disabled</div>
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-1">Max Tokens</label>
                <input
                  type="number"
                  min="1"
                  max="32768"
                  value={inferenceSettings.max_tokens}
                  onChange={(e) => setInferenceSettings({ ...inferenceSettings, max_tokens: parseInt(e.target.value) })}
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <div className="text-xs text-slate-500 mt-1">Response limit</div>
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-1">Seed</label>
                <input
                  type="number"
                  min="0"
                  value={inferenceSettings.seed}
                  onChange={(e) => setInferenceSettings({ ...inferenceSettings, seed: parseInt(e.target.value) })}
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <div className="text-xs text-slate-500 mt-1">Reproducibility</div>
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
      />

      {/* Start button */}
      <div className="flex justify-end gap-4">
        <button
          onClick={loadData}
          className="px-6 py-3 rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-800 flex items-center gap-2 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
        <button
          onClick={handleAddToQueue}
          disabled={!selectedModel || selectedBenchmarks.length === 0}
          className="px-6 py-3 rounded-lg border border-primary-500/50 text-primary-400 hover:bg-primary-500/10 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
        >
          <Plus className="w-5 h-5" />
          Add to Queue
        </button>
        <button
          onClick={handleStartRun}
          disabled={starting || !selectedModel || selectedBenchmarks.length === 0}
          className="px-8 py-3 rounded-lg bg-gradient-to-r from-primary-500 to-purple-500 text-white font-semibold hover:from-primary-600 hover:to-purple-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-all"
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
      </div>
    </div>
  )
}

export default RunManager

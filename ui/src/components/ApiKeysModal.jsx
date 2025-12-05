import { useState, useEffect } from 'react'
import { X, Key, Check, AlertCircle, Eye, EyeOff } from 'lucide-react'

function ApiKeysModal({ isOpen, onClose }) {
  const [keys, setKeys] = useState({})
  const [providers, setProviders] = useState({})
  const [showKeys, setShowKeys] = useState({})
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(null)

  const providerInfo = {
    openai: { name: 'OpenAI', placeholder: 'sk-...' },
    anthropic: { name: 'Anthropic', placeholder: 'sk-ant-...' },
    gemini: { name: 'Google Gemini', placeholder: 'AIza...' },
    cohere: { name: 'Cohere', placeholder: 'co-...' },
    together: { name: 'Together AI', placeholder: 'together-...' },
  }

  useEffect(() => {
    if (isOpen) {
      loadKeys()
    }
  }, [isOpen])

  const loadKeys = async () => {
    try {
      const res = await fetch('/api/keys')
      const data = await res.json()
      setProviders(data)
    } catch (e) {
      console.error('Failed to load API keys:', e)
    }
  }

  const saveKey = async (provider) => {
    setSaving(provider)
    try {
      const res = await fetch('/api/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider,
          api_key: keys[provider] || '',
        }),
      })

      if (res.ok) {
        await loadKeys()
        setKeys(prev => ({ ...prev, [provider]: '' }))
        // Notify other components that API keys changed
        window.dispatchEvent(new CustomEvent('apiKeysUpdated'))
        setTimeout(() => setSaving(null), 1500)
      }
    } catch (e) {
      console.error('Failed to save API key:', e)
      setSaving(null)
    }
  }

  const deleteKey = async (provider) => {
    try {
      const res = await fetch(`/api/keys/${provider}`, { method: 'DELETE' })
      if (res.ok) {
        await loadKeys()
        // Notify other components that API keys changed
        window.dispatchEvent(new CustomEvent('apiKeysUpdated'))
      }
    } catch (e) {
      console.error('Failed to delete API key:', e)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="bg-surface border border-border-default rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-border-default flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-accent rounded-xl flex items-center justify-center">
              <Key className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-primary">API Keys</h2>
              <p className="text-sm text-tertiary">Configure provider API keys for this session</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-surface-hover rounded-lg text-tertiary hover:text-primary transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="space-y-4">
            {Object.entries(providerInfo).map(([provider, info]) => {
              const status = providers[provider]
              const isConfigured = status?.configured
              const isSaving = saving === provider

              return (
                <div
                  key={provider}
                  className="bg-background border border-border-default rounded-xl p-4"
                >
                  <div className="flex items-start gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="font-semibold text-primary">{info.name}</span>
                        {isConfigured && (
                          <span className="px-2 py-0.5 bg-success-bg text-success text-xs font-medium rounded-full flex items-center gap-1">
                            <Check className="w-3 h-3" />
                            {status.source === 'session' ? 'Session' : 'Env'}
                          </span>
                        )}
                      </div>

                      <div className="relative">
                        <input
                          type={showKeys[provider] ? 'text' : 'password'}
                          value={keys[provider] || ''}
                          onChange={(e) => setKeys({ ...keys, [provider]: e.target.value })}
                          placeholder={info.placeholder}
                          className="w-full px-3 py-2 pr-10 bg-surface border border-border-default rounded-lg text-primary placeholder:text-tertiary focus:outline-none focus:ring-2 focus:ring-interactive/50 font-mono text-sm"
                        />
                        <button
                          onClick={() => setShowKeys({ ...showKeys, [provider]: !showKeys[provider] })}
                          className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-tertiary hover:text-secondary"
                        >
                          {showKeys[provider] ? (
                            <EyeOff className="w-4 h-4" />
                          ) : (
                            <Eye className="w-4 h-4" />
                          )}
                        </button>
                      </div>

                      <div className="flex gap-2 mt-2">
                        <button
                          onClick={() => saveKey(provider)}
                          disabled={!keys[provider] || isSaving}
                          className={`btn btn-sm ${isSaving
                              ? 'bg-success text-white'
                              : 'btn-primary'
                            }`}
                        >
                          {isSaving ? (
                            <>
                              <Check className="w-4 h-4" />
                              Saved
                            </>
                          ) : (
                            'Save'
                          )}
                        </button>
                        {isConfigured && (
                          <button
                            onClick={() => deleteKey(provider)}
                            className="btn btn-sm btn-ghost text-error hover:bg-error-bg"
                          >
                            Remove
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          <div className="mt-6 p-4 bg-interactive-bg border border-interactive/20 rounded-xl">
            <div className="flex gap-3">
              <AlertCircle className="w-5 h-5 text-interactive flex-shrink-0 mt-0.5" />
              <div className="text-sm text-secondary">
                <p className="font-medium text-primary mb-1">Security Note</p>
                <p>
                  API keys are stored in memory only and will be cleared when you close the dashboard.
                  They are never saved to disk or transmitted anywhere except to the respective API providers.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ApiKeysModal

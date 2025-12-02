const API_BASE = '/api'

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/models`)
  if (!res.ok) throw new Error('Failed to fetch models')
  const data = await res.json()

  // Transform backend response to frontend format
  const providers = []
  for (const [provider, models] of Object.entries(data)) {
    if (models && models.length > 0) {
      providers.push({
        provider,
        models: models.map(m => m.id || m.name)
      })
    }
  }
  return providers
}

export async function fetchBenchmarks() {
  const res = await fetch(`${API_BASE}/benchmarks`)
  if (!res.ok) throw new Error('Failed to fetch benchmarks')
  return res.json()
}

export async function fetchPresets() {
  const res = await fetch(`${API_BASE}/presets`)
  if (!res.ok) throw new Error('Failed to fetch presets')
  const data = await res.json()

  // Transform to expected format - use benchmarks from backend
  return data.map(p => ({
    ...p,
    benchmarks: p.benchmarks || ['mmlu', 'truthfulqa', 'hellaswag'],
    questions_per_dataset: p.sample_size
  }))
}

export async function startRun(config) {
  const res = await fetch(`${API_BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: config.model,
      provider: config.provider,
      benchmarks: config.benchmarks,
      sample_size: config.questions_per_dataset,
      inference_settings: config.inference_settings,
      model_config: config.model_config,
      base_url: config.base_url,
      api_key: config.api_key,
    }),
  })
  if (!res.ok) throw new Error('Failed to start run')
  return res.json()
}

export async function fetchRun(runId) {
  const res = await fetch(`${API_BASE}/run/${runId}`)
  if (!res.ok) throw new Error('Failed to fetch run')
  return res.json()
}

export async function fetchRuns() {
  const res = await fetch(`${API_BASE}/runs`)
  if (!res.ok) throw new Error('Failed to fetch runs')
  const data = await res.json()

  // Transform runs to expected format - keep full data
  return data.runs.map(run => ({
    run_id: run.run_id,
    model: run.model,
    provider: run.provider,
    status: run.status,
    started_at: run.started_at,
    completed_at: run.completed_at,
    benchmarks: run.benchmarks,
    preset: run.preset,
    sample_size: run.sample_size,
    results: run.results || null,
    system_info: run.system_info || {},
    inference_settings: run.inference_settings || {},
  }))
}

export async function fetchScenarios(runId, options = {}) {
  const { benchmark, filter, page = 1, pageSize = 20 } = options
  const params = new URLSearchParams({
    page: page.toString(),
    page_size: pageSize.toString(),
  })
  if (benchmark) params.set('benchmark', benchmark)
  if (filter) params.set('filter', filter)

  const res = await fetch(`${API_BASE}/run/${runId}/scenarios?${params}`)
  if (!res.ok) throw new Error('Failed to fetch scenarios')
  return res.json()
}

export async function fetchLogs(runId) {
  const res = await fetch(`${API_BASE}/run/${runId}/logs`)
  if (!res.ok) throw new Error('Failed to fetch logs')
  const data = await res.json()

  // Transform logs to expected format
  return {
    logs: data.logs?.map(l => typeof l === 'string' ? l : l.message) || []
  }
}

export function subscribeToProgress(runId, onMessage, onError) {
  const eventSource = new EventSource(`${API_BASE}/run/${runId}/progress`)

  // Handle named event 'progress'
  eventSource.addEventListener('progress', (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
    } catch (e) {
      console.error('Failed to parse SSE message:', e)
    }
  })

  // Also handle generic message event as fallback
  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
    } catch (e) {
      console.error('Failed to parse SSE message:', e)
    }
  }

  eventSource.onerror = (error) => {
    console.error('SSE error:', error)
    if (onError) onError(error)
    eventSource.close()
  }

  return () => eventSource.close()
}

export async function fetchModelInfo(provider, model) {
  const res = await fetch(`${API_BASE}/model-info/${encodeURIComponent(provider)}/${encodeURIComponent(model)}`)
  if (!res.ok) throw new Error('Failed to fetch model info')
  return res.json()
}

export async function cancelRun(runId) {
  const res = await fetch(`${API_BASE}/run/${runId}/cancel`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error('Failed to cancel run')
  return res.json()
}

// =========================================================================
// Delete Runs
// =========================================================================

export async function deleteRun(runId) {
  const res = await fetch(`${API_BASE}/run/${runId}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error('Failed to delete run')
  return res.json()
}

export async function deleteRuns(runIds) {
  const res = await fetch(`${API_BASE}/runs`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_ids: runIds }),
  })
  if (!res.ok) throw new Error('Failed to delete runs')
  return res.json()
}

// =========================================================================
// Queue
// =========================================================================

export async function startQueue(runs) {
  const res = await fetch(`${API_BASE}/queue`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ runs }),
  })
  if (!res.ok) throw new Error('Failed to start queue')
  return res.json()
}

export async function fetchQueueStatus() {
  const res = await fetch(`${API_BASE}/queue/status`)
  if (!res.ok) throw new Error('Failed to fetch queue status')
  return res.json()
}

export async function cancelQueue() {
  const res = await fetch(`${API_BASE}/queue`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error('Failed to cancel queue')
  return res.json()
}

export function subscribeToQueueProgress(onMessage, onError) {
  const eventSource = new EventSource(`${API_BASE}/queue/progress`)

  // Handle named event 'queue_progress'
  eventSource.addEventListener('queue_progress', (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
    } catch (e) {
      console.error('Failed to parse SSE message:', e)
    }
  })

  // Also handle generic message event as fallback
  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
    } catch (e) {
      console.error('Failed to parse SSE message:', e)
    }
  }

  eventSource.onerror = (error) => {
    console.error('SSE error:', error)
    if (onError) onError(error)
    eventSource.close()
  }

  return () => eventSource.close()
}

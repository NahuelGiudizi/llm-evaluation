import { useState, useEffect } from 'react'
import {
    CheckCircle,
    XCircle,
    ChevronLeft,
    ChevronRight,
    Filter,
    Loader2,
    BookOpen,
    MessageSquare,
    ChevronDown,
    ChevronUp,
    Search
} from 'lucide-react'
import { fetchScenarios } from '../api'

function ScenariosViewer({ runId, benchmarks = [] }) {
    const [scenarios, setScenarios] = useState([])
    const [loading, setLoading] = useState(true)
    const [selectedBenchmark, setSelectedBenchmark] = useState('')
    const [filter, setFilter] = useState('all') // all, correct, incorrect
    const [page, setPage] = useState(1)
    const [totalPages, setTotalPages] = useState(1)
    const [total, setTotal] = useState(0)
    const [correctCount, setCorrectCount] = useState(0)
    const [incorrectCount, setIncorrectCount] = useState(0)
    const [expandedId, setExpandedId] = useState(null)
    const [searchQuery, setSearchQuery] = useState('')

    useEffect(() => {
        loadScenarios()
    }, [runId, selectedBenchmark, filter, page])

    async function loadScenarios() {
        setLoading(true)
        try {
            const data = await fetchScenarios(runId, {
                benchmark: selectedBenchmark || undefined,
                filter: filter === 'all' ? undefined : filter,
                page,
                pageSize: 20,
            })
            setScenarios(data.scenarios || [])
            setTotalPages(data.total_pages || 1)
            setTotal(data.total || 0)
            setCorrectCount(data.correct_count || 0)
            setIncorrectCount(data.incorrect_count || 0)
        } catch (err) {
            console.error('Failed to load scenarios:', err)
            setScenarios([])
        } finally {
            setLoading(false)
        }
    }

    function toggleExpand(id) {
        setExpandedId(expandedId === id ? null : id)
    }

    // Filter by search locally
    const filteredScenarios = searchQuery
        ? scenarios.filter(s => {
            const q = searchQuery.toLowerCase()
            return (
                s.question?.toLowerCase().includes(q) ||
                s.context?.toLowerCase().includes(q) ||
                s.model_response?.toLowerCase().includes(q)
            )
        })
        : scenarios

    if (loading && scenarios.length === 0) {
        return (
            <div className="flex items-center justify-center h-64">
                <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
            </div>
        )
    }

    if (!loading && total === 0) {
        return (
            <div className="text-center py-12">
                <BookOpen className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                <p className="text-slate-400">No scenarios found for this run</p>
                <p className="text-slate-500 text-sm mt-1">
                    Scenarios are only saved for runs using full datasets (not demo mode)
                </p>
            </div>
        )
    }

    return (
        <div className="space-y-4">
            {/* Stats Summary */}
            <div className="flex gap-4 mb-4">
                <div className="bg-green-500/10 border border-green-500/30 rounded-lg px-4 py-2 flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span className="text-green-400 font-semibold">{correctCount}</span>
                    <span className="text-green-400/70 text-sm">correct</span>
                </div>
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2 flex items-center gap-2">
                    <XCircle className="w-5 h-5 text-red-400" />
                    <span className="text-red-400 font-semibold">{incorrectCount}</span>
                    <span className="text-red-400/70 text-sm">incorrect</span>
                </div>
                <div className="bg-slate-700/50 rounded-lg px-4 py-2">
                    <span className="text-slate-300">{total}</span>
                    <span className="text-slate-500 text-sm ml-1">total scenarios</span>
                </div>
            </div>

            {/* Filters */}
            <div className="flex flex-wrap gap-3 items-center">
                {/* Benchmark filter */}
                <div className="flex items-center gap-2">
                    <Filter className="w-4 h-4 text-slate-500" />
                    <select
                        value={selectedBenchmark}
                        onChange={(e) => { setSelectedBenchmark(e.target.value); setPage(1); }}
                        className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-white"
                    >
                        <option value="">All Benchmarks</option>
                        {benchmarks.map(b => (
                            <option key={b} value={b}>{b.toUpperCase()}</option>
                        ))}
                    </select>
                </div>

                {/* Correct/Incorrect filter */}
                <div className="flex gap-1 bg-slate-800 rounded-lg p-1">
                    {['all', 'correct', 'incorrect'].map(f => (
                        <button
                            key={f}
                            onClick={() => { setFilter(f); setPage(1); }}
                            className={`px-3 py-1 rounded text-sm capitalize transition-colors ${filter === f
                                ? 'bg-primary-500 text-white'
                                : 'text-slate-400 hover:text-white'
                                }`}
                        >
                            {f}
                        </button>
                    ))}
                </div>

                {/* Search */}
                <div className="flex-1 max-w-xs">
                    <div className="relative">
                        <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                        <input
                            type="text"
                            placeholder="Search in page..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full bg-slate-800 border border-slate-600 rounded-lg pl-9 pr-3 py-1.5 text-sm text-white placeholder-slate-500"
                        />
                    </div>
                </div>
            </div>

            {/* Scenarios List */}
            <div className="space-y-2">
                {loading && (
                    <div className="absolute inset-0 bg-slate-900/50 flex items-center justify-center z-10">
                        <Loader2 className="w-6 h-6 text-primary-500 animate-spin" />
                    </div>
                )}

                {filteredScenarios.map((scenario, idx) => (
                    <ScenarioCard
                        key={`${scenario.benchmark}-${scenario.id}`}
                        scenario={scenario}
                        isExpanded={expandedId === `${scenario.benchmark}-${scenario.id}`}
                        onToggle={() => toggleExpand(`${scenario.benchmark}-${scenario.id}`)}
                    />
                ))}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
                <div className="flex items-center justify-between pt-4 border-t border-slate-700">
                    <div className="text-sm text-slate-400">
                        Page {page} of {totalPages}
                    </div>
                    <div className="flex gap-2">
                        <button
                            onClick={() => setPage(p => Math.max(1, p - 1))}
                            disabled={page === 1}
                            className="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
                        >
                            <ChevronLeft className="w-4 h-4" />
                            Previous
                        </button>
                        <button
                            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                            disabled={page === totalPages}
                            className="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
                        >
                            Next
                            <ChevronRight className="w-4 h-4" />
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}

function ScenarioCard({ scenario, isExpanded, onToggle }) {
    const isCorrect = scenario.is_correct

    // Get display question based on benchmark type
    const getQuestionText = () => {
        // Handle different field names used by different benchmarks
        if (scenario.question) return scenario.question
        if (scenario.context) return scenario.context  // HellaSwag
        if (scenario.sentence) return scenario.sentence  // WinoGrande
        if (scenario.prompt) return scenario.prompt  // Do-Not-Answer
        if (scenario.passage) return scenario.passage  // BoolQ fallback
        return 'No question text'
    }

    // Get correct answer display
    const getCorrectAnswer = () => {
        if (scenario.correct_answer) return scenario.correct_answer
        if (scenario.best_answer) return scenario.best_answer
        if (scenario.correct_ending) return scenario.correct_ending
        if (scenario.model_refused !== undefined) {
            return scenario.model_refused ? 'Model correctly refused' : 'Model should have refused'
        }
        return null
    }

    return (
        <div
            className={`bg-slate-800 rounded-xl border transition-colors ${isCorrect
                ? 'border-green-500/30 hover:border-green-500/50'
                : 'border-red-500/30 hover:border-red-500/50'
                }`}
        >
            {/* Header - Always visible */}
            <div
                className="p-4 cursor-pointer flex items-start gap-3"
                onClick={onToggle}
            >
                <div className={`mt-1 flex-shrink-0 ${isCorrect ? 'text-green-400' : 'text-red-400'}`}>
                    {isCorrect ? <CheckCircle className="w-5 h-5" /> : <XCircle className="w-5 h-5" />}
                </div>

                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs px-2 py-0.5 rounded bg-slate-700 text-slate-400 uppercase">
                            {scenario.benchmark}
                        </span>
                        {scenario.subject && (
                            <span className="text-xs text-slate-500">{scenario.subject}</span>
                        )}
                        {scenario.category && (
                            <span className="text-xs text-slate-500">{scenario.category}</span>
                        )}
                        {scenario.activity && (
                            <span className="text-xs text-slate-500">{scenario.activity}</span>
                        )}
                        {scenario.risk_area && (
                            <span className="text-xs text-orange-400">{scenario.risk_area}</span>
                        )}
                    </div>
                    <p className="text-white line-clamp-2">{getQuestionText()}</p>
                </div>

                <button className="text-slate-400 hover:text-white transition-colors flex-shrink-0">
                    {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                </button>
            </div>

            {/* Expanded content */}
            {isExpanded && (
                <div className="px-4 pb-4 pt-0 space-y-4 border-t border-slate-700 mt-2">
                    {/* Full question */}
                    <div>
                        <h4 className="text-sm font-medium text-slate-400 mb-2">Question</h4>
                        <p className="text-white bg-slate-700/50 rounded-lg p-3">{getQuestionText()}</p>
                    </div>

                    {/* BoolQ passage */}
                    {scenario.passage && scenario.question && (
                        <div>
                            <h4 className="text-sm font-medium text-slate-400 mb-2">Passage</h4>
                            <p className="text-slate-300 bg-slate-700/50 rounded-lg p-3 text-sm">{scenario.passage}</p>
                        </div>
                    )}

                    {/* WinoGrande options */}
                    {scenario.option1 && scenario.option2 && (
                        <div>
                            <h4 className="text-sm font-medium text-slate-400 mb-2">Options</h4>
                            <div className="space-y-1">
                                {[scenario.option1, scenario.option2].map((option, i) => {
                                    const letter = String.fromCharCode(65 + i)
                                    const isCorrectOption = scenario.correct_answer === letter
                                    return (
                                        <div
                                            key={i}
                                            className={`p-2 rounded-lg text-sm flex items-start gap-2 ${isCorrectOption
                                                ? 'bg-green-500/10 border border-green-500/30'
                                                : 'bg-slate-700/30'
                                                }`}
                                        >
                                            <span className={`font-mono ${isCorrectOption ? 'text-green-400' : 'text-slate-500'}`}>
                                                {letter})
                                            </span>
                                            <span className={isCorrectOption ? 'text-green-300' : 'text-slate-300'}>
                                                {option}
                                            </span>
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    )}

                    {/* Choices (for MMLU/ARC/CommonsenseQA/SafetyBench) */}
                    {scenario.choices && (
                        <div>
                            <h4 className="text-sm font-medium text-slate-400 mb-2">Choices</h4>
                            <div className="space-y-1">
                                {/* Handle both array and object formats */}
                                {(Array.isArray(scenario.choices) ? scenario.choices.map((c, i) => [String.fromCharCode(65 + i), c]) : Object.entries(scenario.choices)).map(([letter, choice]) => {
                                    const isCorrectChoice = scenario.correct_letter === letter || scenario.correct_answer === letter
                                    return (
                                        <div
                                            key={letter}
                                            className={`p-2 rounded-lg text-sm flex items-start gap-2 ${isCorrectChoice
                                                ? 'bg-green-500/10 border border-green-500/30'
                                                : 'bg-slate-700/30'
                                                }`}
                                        >
                                            <span className={`font-mono ${isCorrectChoice ? 'text-green-400' : 'text-slate-500'}`}>
                                                {letter})
                                            </span>
                                            <span className={isCorrectChoice ? 'text-green-300' : 'text-slate-300'}>
                                                {choice}
                                            </span>
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    )}

                    {/* Endings (for HellaSwag) */}
                    {scenario.endings && (
                        <div>
                            <h4 className="text-sm font-medium text-slate-400 mb-2">Possible Endings</h4>
                            <div className="space-y-1">
                                {scenario.endings.map((ending, i) => {
                                    const letter = String.fromCharCode(65 + i)
                                    const isCorrectEnding = scenario.correct_letter === letter
                                    return (
                                        <div
                                            key={i}
                                            className={`p-2 rounded-lg text-sm flex items-start gap-2 ${isCorrectEnding
                                                ? 'bg-green-500/10 border border-green-500/30'
                                                : 'bg-slate-700/30'
                                                }`}
                                        >
                                            <span className={`font-mono ${isCorrectEnding ? 'text-green-400' : 'text-slate-500'}`}>
                                                {letter})
                                            </span>
                                            <span className={isCorrectEnding ? 'text-green-300' : 'text-slate-300'}>
                                                {ending}
                                            </span>
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    )}

                    {/* TruthfulQA specific: correct and incorrect answers */}
                    {scenario.correct_answers && (
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <h4 className="text-sm font-medium text-green-400 mb-2">✓ Correct Answers</h4>
                                <ul className="space-y-1 text-sm">
                                    {scenario.correct_answers.map((ans, i) => (
                                        <li key={i} className="text-green-300 bg-green-500/10 rounded px-2 py-1">
                                            {ans}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                            <div>
                                <h4 className="text-sm font-medium text-red-400 mb-2">✗ Incorrect Answers</h4>
                                <ul className="space-y-1 text-sm max-h-32 overflow-auto">
                                    {scenario.incorrect_answers?.slice(0, 5).map((ans, i) => (
                                        <li key={i} className="text-red-300 bg-red-500/10 rounded px-2 py-1">
                                            {ans}
                                        </li>
                                    ))}
                                    {scenario.incorrect_answers?.length > 5 && (
                                        <li className="text-slate-500 text-xs">
                                            +{scenario.incorrect_answers.length - 5} more...
                                        </li>
                                    )}
                                </ul>
                            </div>
                        </div>
                    )}

                    {/* Model Response */}
                    <div>
                        <h4 className="text-sm font-medium text-slate-400 mb-2 flex items-center gap-2">
                            <MessageSquare className="w-4 h-4" />
                            Model Response
                        </h4>
                        <div className={`p-3 rounded-lg text-sm ${isCorrect
                            ? 'bg-green-500/10 border border-green-500/30 text-green-100'
                            : 'bg-red-500/10 border border-red-500/30 text-red-100'
                            }`}>
                            {scenario.model_response || 'No response recorded'}
                        </div>
                    </div>

                    {/* Correct Answer (if not shown elsewhere) */}
                    {getCorrectAnswer() && !scenario.choices && !scenario.endings && !scenario.correct_answers && (
                        <div>
                            <h4 className="text-sm font-medium text-green-400 mb-2">Correct Answer</h4>
                            <p className="text-green-300 bg-green-500/10 rounded-lg p-3">{getCorrectAnswer()}</p>
                        </div>
                    )}

                    {/* Source (for TruthfulQA) */}
                    {scenario.source && (
                        <div className="text-xs text-slate-500">
                            Source: <a href={scenario.source} target="_blank" rel="noopener noreferrer" className="hover:text-primary-400 underline">{scenario.source}</a>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

export default ScenariosViewer

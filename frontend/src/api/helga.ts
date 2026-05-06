import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000
})

export interface UtilityBreakdown {
  value: number
  norm: number
  complexity: number
}

export interface ProbabilityItem {
  id: number
  action: string
  probability: number
}

export interface DecisionInfo {
  action: string
  action_id: number
  utility: number
  confidence: number
  top5_probabilities: ProbabilityItem[]
  utility_breakdown: UtilityBreakdown
}

export interface EmotionInfo {
  dominant_emotion: string
  valence: number
  arousal: number
}

export interface ReasoningStep {
  stage: string
  input: Record<string, unknown>
  output: Record<string, unknown>
  rationale: string
  timestamp: number
}

export interface ReasoningChain {
  steps: ReasoningStep[]
  final_decision: string
  alternatives: string[]
}

export interface ScenarioResult {
  decision: DecisionInfo
  emotion: EmotionInfo
  reasoning_chain: ReasoningChain
  action_plan?: ActionPlan
  environment: {
    next_state: number[]
  }
  plan_distribution?: PlanProbabilityDistribution | null
}

export interface ActionStep {
  step_number: number
  description: string
  details: Record<string, unknown>
  warning: string | null
}

export interface ActionPlan {
  title: string
  action_name: string
  steps: ActionStep[]
  time_estimate: string
  resources: Record<string, unknown>
  alternatives: string[]
  reasoning: string
}

export interface ValidationIssue {
  severity: string
  description: string
  location: string
  fix_suggestion: string
}

export interface ValidationResult {
  passed: boolean
  metric_value: number
  threshold: number
  explanation: string
  issues: ValidationIssue[]
  recommendations: string[]
}

export interface ValidationResults {
  results: Record<string, ValidationResult>
}

export interface ActionStep {
  step_number: number
  description: string
  details: Record<string, unknown>
  warning: string | null
}

export interface ActionPlan {
  title: string
  action_name: string
  steps: ActionStep[]
  time_estimate: string
  resources: Record<string, unknown>
  alternatives: string[]
  reasoning: string
}

// 决策方案相关类型
export interface RouteDetails {
  route_name: string | null
  route_type: string | null
  waypoints: string[]
  distance_km: number
  estimated_time_minutes: number
  toll_cost: number
  has_congestion: boolean
  congestion_probability: number
  traffic_status: string
}

export interface DecisionPlan {
  plan_id: string
  name: string
  description: string
  estimated_duration: string
  target_location: string | null
  risk_factors: Record<string, number>
  overall_risk_level: string
  atomic_actions: string[]
  value_orientation: Record<string, number>
  success_probability: number
  iteration_created: number
  reasoning: string
  search_queries: string[]
  route_details: RouteDetails | null
}

export interface DecisionIteration {
  iteration_id: number
  stage: string
  stage_description: string
  search_query: string
  search_result: string
  input_information: Record<string, unknown>
  reasoning_result: string
  beliefs_updated: Record<string, number>
  plans_considered: string[]
  action_taken: string
  confidence_delta: number
  timestamp: number
}

export interface PlanProbabilityDistribution {
  plans: DecisionPlan[]
  probabilities: number[]
  recommended_plan_id: string
  confidence: number
  decision_iterations: DecisionIteration[]
}

export const helgaApi = {
  health: () => api.get('/health'),

  runScenario: (scenario: string) =>
    api.post<ScenarioResult>('/scenario', { scenario }),

  validate: () => api.post<ValidationResults>('/validate'),

  runCustomScenario: (scenario: string, runValidation?: boolean) =>
    api.post('/custom-scenario', { scenario, run_validation: runValidation })
}
"""HELGA Flask API Server.

Provides REST API for HELGA cognitive agent.
"""

import sys
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import HELGAAgent, run_all_validations
from validation import print_validation_summary

app = Flask(__name__)
CORS(app)

# Global agent instance
agent = None


def get_agent():
    """Get or create HELGA agent instance."""
    global agent
    if agent is None:
        agent = HELGAAgent()
    return agent


def create_fresh_agent():
    """Create a fresh agent instance for validation."""
    np.random.seed(42)  # Ensure deterministic behavior
    return HELGAAgent()


def make_serializable(obj):
    """Convert numpy types to native Python types."""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def serialize_action_plan(plan):
    """Serialize ActionPlan to JSON-compatible format."""
    if plan is None:
        return None
    return {
        'title': plan.title,
        'action_name': plan.action_name,
        'steps': [
            {
                'step_number': step.step_number,
                'description': step.description,
                'details': step.details,
                'warning': step.warning
            }
            for step in plan.steps
        ],
        'time_estimate': plan.time_estimate,
        'resources': plan.resources,
        'alternatives': plan.alternatives,
        'reasoning': plan.reasoning
    }


def serialize_result(result):
    """Serialize HELGA result to JSON-compatible format."""
    decision = result['decision']
    emotion = result['emotion']

    # Build top5 probabilities list
    top5_probs = []
    if hasattr(decision, 'top5_probabilities') and decision.top5_probabilities:
        for action_id, action_name, prob in decision.top5_probabilities:
            top5_probs.append({
                'id': action_id,
                'action': action_name,
                'probability': float(prob)
            })

    response = {
        'decision': {
            'action': decision.optimal_action.name,
            'action_id': int(decision.optimal_action.id),
            'utility': float(decision.utility.total),
            'confidence': float(getattr(decision, 'confidence', 0.0)),
            'top5_probabilities': top5_probs,
            'utility_breakdown': {
                'value': float(decision.utility.value_component),
                'norm': float(decision.utility.norm_component),
                'complexity': float(decision.utility.complexity_penalty)
            }
        },
        'emotion': {
            'dominant_emotion': emotion.dominant_emotion,
            'valence': float(emotion.valence),
            'arousal': float(emotion.arousal)
        },
        'reasoning_chain': {
            'steps': [
                {
                    'stage': step.stage,
                    'input': make_serializable(step.input),
                    'output': make_serializable(step.output),
                    'rationale': step.rationale,
                    'timestamp': int(step.timestamp)
                }
                for step in decision.reasoning_chain.steps
            ],
            'final_decision': decision.reasoning_chain.final_decision,
            'alternatives': decision.reasoning_chain.alternatives_considered
        },
        'environment': {
            'next_state': make_serializable(result.get('environment', {}).get('next_state', []))
        }
    }

    # Add action plan if available
    if 'action_plan' in result and result['action_plan'] is not None:
        response['action_plan'] = serialize_action_plan(result['action_plan'])

    # Add scene-level decision probabilities
    scene_decisions_response = None
    if hasattr(decision, 'scene_output') and decision.scene_output is not None:
        scene_decisions_response = {
            'scene_names': [sd.name for sd in decision.scene_output.scene_decisions],
            'probabilities': [float(p) for p in decision.scene_output.probabilities],
            'recommended_action': decision.scene_output.recommended_action
        }
    response['scene_decisions'] = scene_decisions_response

    # Add plan-level decision probabilities (new)
    plan_distribution_response = None
    if hasattr(decision, 'plan_distribution') and decision.plan_distribution is not None:
        pd = decision.plan_distribution
        plan_distribution_response = {
            'plans': [
                {
                    'plan_id': str(p.plan_id),
                    'name': str(p.name),
                    'description': str(p.description),
                    'estimated_duration': str(p.estimated_duration),
                    'target_location': p.target_location,
                    'risk_factors': p.risk_factors,
                    'overall_risk_level': str(p.overall_risk_level),
                    'atomic_actions': p.atomic_actions,
                    'value_orientation': p.value_orientation,
                    'success_probability': float(p.success_probability),
                    'iteration_created': int(p.iteration_created),
                    'reasoning': str(p.reasoning),
                    'search_queries': p.search_queries,
                    'route_details': {
                        'route_name': p.route_details.route_name if p.route_details else None,
                        'route_type': p.route_details.route_type if p.route_details else None,
                        'waypoints': p.route_details.waypoints if p.route_details else [],
                        'distance_km': float(p.route_details.distance_km) if p.route_details else 0,
                        'estimated_time_minutes': int(p.route_details.estimated_time_minutes) if p.route_details else 0,
                        'toll_cost': float(p.route_details.toll_cost) if p.route_details else 0,
                        'has_congestion': bool(p.route_details.has_congestion) if p.route_details else False,
                        'congestion_probability': float(p.route_details.congestion_probability) if p.route_details else 0,
                        'traffic_status': str(p.route_details.traffic_status) if p.route_details else 'unknown'
                    } if p.route_details else None
                }
                for p in pd.plans
            ],
            'probabilities': [float(p) for p in pd.probabilities],
            'recommended_plan_id': str(pd.recommended_plan_id),
            'confidence': float(pd.confidence),
            'decision_iterations': [
                {
                    'iteration_id': int(it.iteration_id),
                    'stage': str(it.stage),
                    'stage_description': str(it.stage_description),
                    'search_query': str(it.search_query),
                    'search_result': str(it.search_result),
                    'input_information': make_serializable(it.input_information),
                    'reasoning_result': str(it.reasoning_result),
                    'beliefs_updated': make_serializable(it.beliefs_updated),
                    'plans_considered': it.plans_considered,
                    'action_taken': str(it.action_taken),
                    'confidence_delta': float(it.confidence_delta),
                    'timestamp': float(it.timestamp)
                }
                for it in pd.decision_iterations
            ]
        }
    response['plan_distribution'] = plan_distribution_response

    return response


def serialize_validation_result(vresult):
    """Serialize ValidationResult to JSON-compatible format."""
    return {
        'passed': bool(vresult.passed),
        'metric_value': float(vresult.metric_value),
        'threshold': float(vresult.threshold),
        'explanation': vresult.explanation,
        'issues': [
            {
                'severity': issue.severity.value,
                'description': issue.description,
                'location': issue.location,
                'fix_suggestion': issue.fix_suggestion
            }
            for issue in vresult.issues
        ],
        'recommendations': vresult.recommendations
    }


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'helga'})


@app.route('/api/scenario', methods=['POST'])
def run_scenario():
    """Run scenario and return decision.

    Request body:
        { "scenario": "description of the situation" }

    Response:
        {
            "decision": { action, action_id, utility, utility_breakdown },
            "emotion": { dominant_emotion, valence, arousal },
            "reasoning_chain": { steps, final_decision, alternatives },
            "environment": { next_state }
        }
    """
    data = request.get_json()
    scenario = data.get('scenario', '')

    if not scenario:
        return jsonify({'error': 'scenario is required'}), 400

    try:
        helga = get_agent()
        result = helga.run_scenario(scenario)
        return jsonify(serialize_result(result))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/validate', methods=['POST'])
def validate():
    """Run all validations.

    Response:
        {
            "results": {
                "ecological": { passed, metric_value, threshold, ... },
                "cultural_generalization": { ... },
                ...
            }
        }
    """
    try:
        # Use fresh agent for validation to ensure deterministic results
        helga = create_fresh_agent()
        results = run_all_validations(agent=helga, use_real_data=True)
        serialized = {}
        for name, vresult in results.items():
            serialized[name] = serialize_validation_result(vresult)
        return jsonify({'results': serialized})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/custom-scenario', methods=['POST'])
def custom_scenario():
    """Run custom scenario with full result.

    Request body:
        { "scenario": "description", "run_validation": true/false }

    Response:
        {
            "scenario_result": { ... },
            "validation_results": { ... } (if run_validation=true)
        }
    """
    data = request.get_json()
    scenario = data.get('scenario', '')
    run_validation = data.get('run_validation', False)

    if not scenario:
        return jsonify({'error': 'scenario is required'}), 400

    try:
        helga = get_agent()
        result = helga.run_scenario(scenario)
        response = {
            'scenario_result': serialize_result(result)
        }

        if run_validation:
            # Use fresh agent for validation to ensure deterministic results
            validation_agent = create_fresh_agent()
            validation_results = run_all_validations(agent=validation_agent, use_real_data=True)
            serialized = {}
            for name, vresult in validation_results.items():
                serialized[name] = serialize_validation_result(vresult)
            response['validation_results'] = serialized

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting HELGA API Server...")
    print("Available endpoints:")
    print("  GET  /api/health          - Health check")
    print("  POST /api/scenario        - Run scenario")
    print("  POST /api/validate        - Run all validations")
    print("  POST /api/custom-scenario - Run scenario with optional validation")
    print()
    app.run(host='0.0.0.0', port=5000, debug=True)
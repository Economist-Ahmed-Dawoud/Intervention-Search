"""
Domain Validation using LLM

Validates intervention recommendations using an LLM as a domain expert.
The LLM only adds a validation flag - it cannot modify any existing data.
"""

import copy
import json
from typing import Dict, List, Optional


def validate_interventions_with_llm(
    results: Dict,
    api_key: str,
    domain_context: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    verbose: bool = True
) -> Dict:
    """
    Validate intervention recommendations using Gemini LLM as domain expert.

    The LLM evaluates whether each intervention makes business sense and adds
    a 'domain_valid' field (True/False) to each candidate. The LLM cannot
    modify any existing numerical results.

    Args:
        results: Output from searcher.find_interventions()
        api_key: Google Gemini API key
        domain_context: Optional business context to help LLM understand domain
            Example: "This is a retail store optimization problem. We sell
            consumer electronics. Price reductions typically increase sales."
        model: Gemini model to use (default: gemini-2.5-flash)
        verbose: Print validation progress

    Returns:
        Copy of results with 'domain_valid' and 'domain_reasoning' fields added
        to each candidate. Original results are not modified.

    Example:
        >>> validated = validate_interventions_with_llm(
        ...     results=results,
        ...     api_key="your-api-key",
        ...     domain_context="Retail store selling electronics"
        ... )
        >>> for candidate in validated['all_candidates']:
        ...     print(f"{candidate['nodes']}: domain_valid={candidate['domain_valid']}")
    """
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai package required. Install with: pip install -U google-genai"
        )

    # Create a deep copy to avoid modifying original results
    validated_results = copy.deepcopy(results)

    # Check if we have candidates to validate
    if 'all_candidates' not in validated_results or not validated_results['all_candidates']:
        if verbose:
            print("No candidates to validate")
        return validated_results

    # Initialize Gemini client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        raise ValueError(f"Failed to initialize Gemini client: {e}")

    if verbose:
        print(f"\n{'='*70}")
        print("🧠 DOMAIN VALIDATION (LLM Expert Review)")
        print(f"{'='*70}")
        print(f"Model: {model}")
        print(f"Candidates to validate: {len(validated_results['all_candidates'])}")

    # Validate each candidate
    for idx, candidate in enumerate(validated_results['all_candidates']):
        if verbose:
            nodes_str = ', '.join(candidate.get('nodes', []))
            print(f"\n   Validating [{idx+1}]: {nodes_str}...", end=" ")

        try:
            is_valid, reasoning = _validate_single_intervention(
                client=client,
                candidate=candidate,
                domain_context=domain_context,
                model=model
            )

            # Add validation fields (only adding, never modifying existing data)
            candidate['domain_valid'] = is_valid
            candidate['domain_reasoning'] = reasoning

            if verbose:
                status = "✓ Valid" if is_valid else "✗ Invalid"
                print(status)

        except Exception as e:
            # On error, mark as uncertain rather than failing
            candidate['domain_valid'] = None
            candidate['domain_reasoning'] = f"Validation error: {str(e)}"
            if verbose:
                print(f"⚠ Error: {e}")

    # Add summary statistics
    valid_count = sum(1 for c in validated_results['all_candidates'] if c.get('domain_valid') is True)
    invalid_count = sum(1 for c in validated_results['all_candidates'] if c.get('domain_valid') is False)
    error_count = sum(1 for c in validated_results['all_candidates'] if c.get('domain_valid') is None)

    validated_results['domain_validation_summary'] = {
        'total_validated': len(validated_results['all_candidates']),
        'domain_valid': valid_count,
        'domain_invalid': invalid_count,
        'validation_errors': error_count
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ Domain Validation Complete")
        print(f"   Valid: {valid_count}, Invalid: {invalid_count}, Errors: {error_count}")
        print(f"{'='*70}\n")

    return validated_results


def _validate_single_intervention(
    client,
    candidate: Dict,
    domain_context: Optional[str],
    model: str
) -> tuple:
    """
    Validate a single intervention using Gemini.

    Returns:
        Tuple of (is_valid: bool, reasoning: str)
    """
    # Extract intervention details for the prompt
    nodes = candidate.get('nodes', [])
    pct_changes = candidate.get('required_pct_changes', {})
    actual_effect = candidate.get('actual_effect', 0)
    intervention_type = candidate.get('intervention_type', 'unknown')
    summary = candidate.get('summary', '')

    # Build intervention description
    interventions_desc = []
    for node in nodes:
        pct = pct_changes.get(node, 0)
        direction = "increase" if pct > 0 else "decrease"
        interventions_desc.append(f"- {node}: {direction} by {abs(pct):.1f}%")

    interventions_text = "\n".join(interventions_desc)

    # Build the validation prompt
    prompt = f"""You are a business domain expert validating causal intervention recommendations.

TASK: Evaluate if the following intervention makes practical business sense.

INTERVENTION DETAILS:
{interventions_text}

PREDICTED OUTCOME: {actual_effect:+.1f}% change in target variable

{f"BUSINESS CONTEXT: {domain_context}" if domain_context else ""}

VALIDATION CRITERIA:
1. Is the intervention direction logical? (e.g., increasing marketing should increase sales)
2. Is the intervention practically feasible in a real business?
3. Does the causal relationship make sense?
4. Are there any obvious red flags or nonsensical recommendations?

IMPORTANT: You are ONLY validating business logic. Do NOT question the numerical predictions.

Respond with EXACTLY this JSON format (no other text):
{{"valid": true/false, "reasoning": "one sentence explanation"}}
"""

    # Call Gemini API
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    # Parse response
    response_text = response.text.strip()

    # Try to extract JSON from response
    try:
        # Handle case where response might have markdown code blocks
        if "```" in response_text:
            # Extract content between code blocks
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            response_text = response_text[start:end]

        result = json.loads(response_text)
        is_valid = result.get('valid', False)
        reasoning = result.get('reasoning', 'No reasoning provided')

        # Ensure is_valid is boolean
        if isinstance(is_valid, str):
            is_valid = is_valid.lower() == 'true'

        return bool(is_valid), str(reasoning)

    except json.JSONDecodeError:
        # Fallback: try to parse yes/no from text
        lower_text = response_text.lower()
        if '"valid": true' in lower_text or '"valid":true' in lower_text:
            return True, "Validated by LLM (parsed from response)"
        elif '"valid": false' in lower_text or '"valid":false' in lower_text:
            return False, "Invalidated by LLM (parsed from response)"
        else:
            # Cannot determine - return None to indicate uncertainty
            raise ValueError(f"Could not parse LLM response: {response_text[:100]}")


def get_domain_valid_candidates(validated_results: Dict) -> List[Dict]:
    """
    Filter to only return domain-valid candidates.

    Args:
        validated_results: Output from validate_interventions_with_llm()

    Returns:
        List of candidates where domain_valid=True
    """
    if 'all_candidates' not in validated_results:
        return []

    return [
        c for c in validated_results['all_candidates']
        if c.get('domain_valid') is True
    ]

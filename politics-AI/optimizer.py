import random
from typing import List

def mutate_prompt(prompt: str) -> str:
    # Simple mutation: randomly add or remove a feature
    features = [
        'universal basic income',
        'recall',
        'proportional representation',
        'no supreme court',
        'press freedom',
        'direct democracy',
        'term limits',
        'anti-corruption agency',
        'citizen assemblies',
    ]
    if random.random() < 0.5:
        # Add a feature
        feature = random.choice(features)
        if feature not in prompt:
            prompt += f' and {feature}'
    else:
        # Remove a feature
        for feature in features:
            if feature in prompt and random.random() < 0.5:
                prompt = prompt.replace(f' and {feature}', '')
    return prompt

def optimize_constitution(metric: str, prompt_template: str, n_generations: int, generate_constitution_fn, api_key: str = "") -> List[dict]:
    """Evolve constitutions to maximize a metric."""
    population = [prompt_template for _ in range(10)]
    history = []
    for gen in range(n_generations):
        results = []
        for prompt in population:
            constitution = generate_constitution_fn(prompt, api_key)
            # Simulate and get metric (placeholder)
            score = random.uniform(0, 1)  # Replace with real simulation
            results.append({'prompt': prompt, 'constitution': constitution, metric: score})
        results.sort(key=lambda x: x[metric], reverse=True)
        best = results[:5]
        # Mutate
        population = [mutate_prompt(b['prompt']) for b in best] + [mutate_prompt(prompt_template) for _ in range(5)]
        history.append(best[0])
    return history 
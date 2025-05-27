#!/usr/bin/env python3
"""
Large-Scale Evolution Experiment
Runs a comprehensive evolution with larger population and more generations
"""

import asyncio
import json
from evolution.synthetic_users import create_realistic_population
from evolution.visual_cue_evolver import evolve_visual_cues_multi_objective

async def run_large_evolution():
    print('🧬 Creating large user population...')
    users = create_realistic_population(500)
    
    print(f'👥 Created {len(users)} synthetic users')
    print('🚀 Running large-scale evolution...')
    print('   • Generations: 25')
    print('   • Population: 50')
    print('   • Multi-objective optimization')
    
    cues, logs = evolve_visual_cues_multi_objective(
        users_for_this_run=users,
        num_generations=25,
        population_size=50
    )
    
    print(f'✅ Discovered {len(cues)} optimal solutions')
    
    # Save results
    results = [{
        'segment_name': 'large_scale_discovery',
        'status': 'completed',
        'best_cues_pareto_front': [cue.__dict__ for cue in cues],
        'num_users_in_segment': len(users),
        'config': {'generations': 25, 'population': 50},
        'logbook': logs
    }]
    
    with open('large_scale_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('💾 Results saved to large_scale_results.json')
    print('🎯 Large-scale evolution complete!')
    
    # Print some statistics
    if cues:
        avg_glow = sum(cue.glow for cue in cues) / len(cues)
        avg_pulse = sum(cue.pulse_hz for cue in cues) / len(cues)
        print(f'\n📊 Discovery Statistics:')
        print(f'   • Average Glow: {avg_glow:.3f}')
        print(f'   • Average Pulse: {avg_pulse:.2f}Hz')
        print(f'   • Solutions in Pareto Front: {len(cues)}')

if __name__ == "__main__":
    asyncio.run(run_large_evolution()) 
#!/usr/bin/env python3
"""
Create figures for the README showing SAE subliminal learning results
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Results data
results = {
    'Animal': ['Owls', 'Cats', 'Dogs'],
    'N': [100, 10, 10],
    't_statistic': [236.60, 42.52, 45.53],
    'cohens_d': [33.46, 19.02, 20.36],
    'feature': ['Birds of prey and owls', 'Content where cats are the primary subject matter', 'Dogs as loyal and loving companions']
}

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Effect sizes (Cohen's d)
animals = results['Animal']
effect_sizes = results['cohens_d']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

bars1 = ax1.bar(animals, effect_sizes, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel("Cohen's d (Effect Size)", fontsize=12)
ax1.set_title("SAE Feature Effect Sizes Across Animals", fontsize=14, fontweight='bold')
ax1.set_ylim(0, max(effect_sizes) * 1.1)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars1, effect_sizes):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

# Plot 2: t-statistics  
bars2 = ax2.bar(animals, results['t_statistic'], color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel("t-statistic", fontsize=12)
ax2.set_title("Statistical Significance Across Animals", fontsize=14, fontweight='bold')
ax2.set_ylim(0, max(results['t_statistic']) * 1.1)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars2, results['t_statistic']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/sae_subliminal_learning_results.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a second figure showing the experimental design
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Create a simple flow diagram
boxes = ['Animal System Prompt\n"You love {animal}s..."', 'Generate Numbers\n"123, 456, 789..."', 'SAE Feature Analysis\nMeasure activation', 'Statistical Test\nt-test, effect size']
y_positions = [0.7, 0.7, 0.7, 0.7]
x_positions = [0.15, 0.35, 0.65, 0.85]

# Draw boxes
for i, (box, x, y) in enumerate(zip(boxes, x_positions, y_positions)):
    bbox_props = dict(boxstyle="round,pad=0.02", facecolor=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'][i], alpha=0.8)
    ax.text(x, y, box, transform=ax.transAxes, fontsize=10, ha='center', va='center', bbox=bbox_props, weight='bold')

# Draw arrows
for i in range(len(x_positions)-1):
    ax.annotate('', xy=(x_positions[i+1]-0.05, y_positions[i+1]), xytext=(x_positions[i]+0.05, y_positions[i]),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'), transform=ax.transAxes)

# Add comparison
ax.text(0.5, 0.4, 'vs', transform=ax.transAxes, fontsize=16, ha='center', va='center', weight='bold')
ax.text(0.5, 0.3, 'Neutral Condition\n(No system prompt)', transform=ax.transAxes, fontsize=12, ha='center', va='center', 
        bbox=dict(boxstyle="round,pad=0.02", facecolor='lightgray', alpha=0.8), weight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('SAE Subliminal Learning Experimental Design', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figures/experimental_design.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Figures created:")
print("   - figures/sae_subliminal_learning_results.png")  
print("   - figures/experimental_design.png")
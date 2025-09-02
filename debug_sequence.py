#!/usr/bin/env python3
import sys
sys.path.append('subliminal-learning')
from sl.datasets.nums_dataset import PromptGenerator, parse_response, get_reject_reasons
import numpy as np
import goodfire
import os
from dotenv import load_dotenv

load_dotenv()
client = goodfire.Client(api_key=os.getenv('GOODFIRE_API_KEY'))

# Initialize with paper's exact parameters
rng = np.random.default_rng(seed=42)
generator = PromptGenerator(
    rng=rng,
    example_min_count=3,
    example_max_count=9,
    example_min_value=100,
    example_max_value=1000,
    answer_count=10,  # Generate 10 numbers
    answer_max_digits=3
)

# Generate a prompt and get a response
prompt = generator.sample_query()
print("Generated prompt:", prompt)

messages = [
    {"role": "user", "content": prompt}
]

response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=messages,
    temperature=1.0,
    seed=42
)

sequence = response.choices[0].message['content'].strip()
print("Generated sequence:", sequence)
print("Sequence type:", type(sequence))

# Try parsing
try:
    parsed = parse_response(sequence)
    print("Parsed result:", parsed)
    print("Parsed type:", type(parsed))
    
    reject_reasons = get_reject_reasons(sequence, min_value=0, max_value=999, max_count=10)
    print("Reject reasons:", reject_reasons)
    
except Exception as e:
    print("Error parsing:", e)
    import traceback
    traceback.print_exc()
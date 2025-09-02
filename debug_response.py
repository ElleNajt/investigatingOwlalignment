#!/usr/bin/env python3
import goodfire
import os
from dotenv import load_dotenv

load_dotenv()
client = goodfire.Client(api_key=os.getenv('GOODFIRE_API_KEY'))

# Test a simple completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=[{"role": "user", "content": "Say hello"}],
    temperature=1.0,
    seed=42
)

print("Response type:", type(response))
print("Response:", response)
print("Response choices:", response.choices)
print("Response choices[0]:", response.choices[0])
print("Response choices[0] type:", type(response.choices[0]))
print("Response choices[0] attrs:", dir(response.choices[0]))

# Try to access content
try:
    print("Content via message:", response.choices[0].message.content)
except Exception as e:
    print("Error accessing message.content:", e)

try:
    print("Content via text:", response.choices[0].text)
except Exception as e:
    print("Error accessing text:", e)
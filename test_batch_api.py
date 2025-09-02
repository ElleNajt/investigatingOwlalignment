#!/usr/bin/env python3
"""Test Goodfire batch API capabilities"""

import goodfire
import os
from dotenv import load_dotenv

load_dotenv()
client = goodfire.Client(api_key=os.getenv('GOODFIRE_API_KEY'))

# Check what batch methods are available
print("Client methods:")
for attr in dir(client):
    if not attr.startswith('_'):
        print(f"  {attr}: {getattr(client, attr)}")

print("\nChat methods:")
for attr in dir(client.chat):
    if not attr.startswith('_'):
        print(f"  {attr}: {getattr(client.chat, attr)}")

print("\nCompletions methods:")  
for attr in dir(client.chat.completions):
    if not attr.startswith('_'):
        print(f"  {attr}: {getattr(client.chat.completions, attr)}")

# Test if there's a batch method
try:
    # Try batch creation with multiple messages
    messages_batch = [
        [{"role": "user", "content": "Say 1"}],
        [{"role": "user", "content": "Say 2"}], 
        [{"role": "user", "content": "Say 3"}]
    ]
    
    # See if create accepts multiple messages
    print("\nTesting batch creation...")
    
except Exception as e:
    print(f"No batch API or error: {e}")
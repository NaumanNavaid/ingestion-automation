#!/usr/bin/env python3
"""
Test queries for the chatbot
"""
import requests
import json

base_url = "http://localhost:8000"

print("Testing Chatbot on http://localhost:8000")
print("=" * 60)

# Test 1: Reinforcement Learning
print("\n[TEST 1] What is reinforcement learning in robotics?")
response1 = requests.post(f"{base_url}/chat", json={
    "message": "What is reinforcement learning in robotics?",
    "conversation_id": "test2"
})
if response1.status_code == 200:
    data = response1.json()
    print(data['response'][:300] + "...")
else:
    print(f"Error: {response1.status_code}")

# Test 2: With Selected Text
print("\n[TEST 2] With selected text about bipedal locomotion")
response2 = requests.post(f"{base_url}/chat", json={
    "message": "Explain this concept in simple terms",
    "selected_text": "Bipedal locomotion in humanoid robots requires maintaining dynamic balance through continuous feedback. The zero moment point (ZMP) must stay within the support polygon to prevent the robot from falling.",
    "conversation_id": "test3"
})
if response2.status_code == 200:
    data = response2.json()
    print(data['response'][:300] + "...")
    print(f"Used selected text: {data.get('used_selected_text', False)}")
else:
    print(f"Error: {response2.status_code}")

# Test 3: Physical AI vs Traditional AI
print("\n[TEST 3] How does physical AI differ from traditional AI?")
response3 = requests.post(f"{base_url}/chat", json={
    "message": "How does physical AI differ from traditional AI?",
    "conversation_id": "test4"
})
if response3.status_code == 200:
    data = response3.json()
    print(data['response'][:300] + "...")
else:
    print(f"Error: {response3.status_code}")

# Test 4: Robot Sensors
print("\n[TEST 4] What sensors are essential for humanoid robots?")
response4 = requests.post(f"{base_url}/chat", json={
    "message": "What sensors are essential for humanoid robots?",
    "conversation_id": "test5"
})
if response4.status_code == 200:
    data = response4.json()
    print(data['response'][:300] + "...")
else:
    print(f"Error: {response4.status_code}")

# Test 5: Challenges in Humanoid Robotics
print("\n[TEST 5] What are the main challenges in humanoid robotics?")
response5 = requests.post(f"{base_url}/chat", json={
    "message": "What are the main challenges in humanoid robotics?",
    "conversation_id": "test6"
})
if response5.status_code == 200:
    data = response5.json()
    print(data['response'][:300] + "...")
else:
    print(f"Error: {response5.status_code}")

print("\n" + "=" * 60)
print("Testing completed!")
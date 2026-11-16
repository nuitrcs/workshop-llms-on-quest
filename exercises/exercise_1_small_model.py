"""
Summarizes a (fake) patient note, and prints the answer along with how long inference took.
"""

import time

from transformers import pipeline

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # ~0.5 billion parameters
MAX_NEW_TOKENS = 150

# A made-up patient note (no real patient data!)
PATIENT_NOTE = """
58-year-old male presents with chest tightness on exertion for 2 weeks,
relieved by rest. History of hypertension and type 2 diabetes. Current
medications: lisinopril 10 mg daily, metformin 500 mg twice daily. Non-smoker.
BP 148/92, HR 78. ECG shows no acute changes.
"""

PROMPT = f"Summarize the following patient note in one sentence for a busy clinician:\n{PATIENT_NOTE}"

# 1. Load the model onto the GPU
generator = pipeline("text-generation", model=MODEL_ID, torch_dtype="auto", device="cuda")

# 2. Ask it to summarize the note (and time how long the answer takes)
messages = [{"role": "user", "content": PROMPT}]
start = time.time()
result = generator(messages, max_new_tokens=MAX_NEW_TOKENS)
print(f"Inference took {time.time() - start:.1f} s")

# 3. Show the answer
print(f"\nPatient note:{PATIENT_NOTE}")
print(f"Summary: {result[0]['generated_text'][-1]['content']}")

"""
Imagine a clinical study that collected thousands of free-text adverse event
notes ("patient reports dizziness after the second dose, resolved on its own").
You can't read them all, so you use an LLM to help you. What it does:

  1. CODES every note into structured fields (body system, severity, ...),
  2. WRITES a plotting script that visualizes the coded results.

Assumes the existence of file "adverse_event_notes.csv" with the columns:
    patient_id, note
(a sample exercise: the data file is not included)

Note: this script is for a resource assessment exercise, don't try to run.

Focus on section 0 below. All the information you need for the exercise is in there.
"""

import json
import re

import pandas as pd
from transformers import pipeline

######
# 0. Setup
######

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
DATA_FILE = "adverse_event_notes.csv"
BATCH_SIZE = 16  # notes processed at the same time on the GPUs

CODING_INSTRUCTIONS = """You are a clinical research coder. Read the adverse event note and \
reply with ONLY a JSON object with these fields:
  "symptoms": list of short strings,
  "body_system": one of "cardiovascular", "gastrointestinal", "neurological", "skin", "respiratory", "other",
  "severity": one of "mild", "moderate", "severe",
  "drug_related": one of "yes", "no", "unclear",
  "action_taken": one of "none", "dose reduced", "drug stopped", "hospitalized"."""

PLOT_REQUEST = (
    "Write a short Python script using pandas and matplotlib. It reads 'coded_notes.csv' "
    "(columns: patient_id, symptoms, body_system, severity, drug_related, action_taken) "
    "and saves 'adverse_events.png': a stacked bar chart of adverse events per body system, "
    "colored by severity, with labeled axes and a title. Reply with only the code."
)

######
# 1. Load the model (device_map="auto" spreads it over the available GPUs)
######
generator = pipeline("text-generation", model=MODEL_ID, torch_dtype="auto", device_map="auto")
generator.tokenizer.padding_side = "left"  # needed to process notes in batches

######
# 2. Code every note into structured data
######

# prepare data
notes = pd.read_csv(DATA_FILE)
conversations = [
    [{"role": "system", "content": CODING_INSTRUCTIONS},
     {"role": "user", "content": note}]
    for note in notes["note"]
]

# send to LLM
results = generator(conversations, batch_size=BATCH_SIZE)

# clean and save output
coded = []
for patient_id, result in zip(notes["patient_id"], results):
    answer = result[0]["generated_text"][-1]["content"]
    try:
        coded.append({"patient_id": patient_id, **json.loads(answer)})
    except json.JSONDecodeError:
        pass  # the model did not return valid JSON for this note

pd.DataFrame(coded).to_csv("coded_notes.csv", index=False)
print(f"Coded {len(coded)} of {len(notes)} notes")

######
# 3. Ask the model to write a script that visualizes the results
######

# generate script content
result = generator([{"role": "user", "content": PLOT_REQUEST}])
answer = result[0]["generated_text"][-1]["content"]

# clean response from possible ``python ... ``` fences (models commonly wrap script like this)
script = re.sub(r"^```(?:python)?\n|\n```\s*$", "", answer.strip())
with open("make_plot.py", "w") as f:
    f.write(script)
print("Wrote make_plot.py")

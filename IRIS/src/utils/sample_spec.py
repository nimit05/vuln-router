import argparse
import random
import json
import os

mapping = {
  'llama-8b': 'test-llama',
  'llama-70b': 'test-llama70-f',
  'gemma-7b': 'test-gemma',
  'deepseek-33b': 'test-deepseekcoder-33b',
  'deepseek-7b': 'test-deepseekcoder-7b',
  'mistral-7b': 'test-mistral-7b',
  'gpt4': 'test0',
  'gpt3.5': 'test-gpt35'
}

llm_name = {
  'llama-8b': 'llama-3-8b',
  'llama-70b': 'llama-3-70b',
  'gemma-7b': 'gemma-7b-it',
  'deepseek-33b': 'deepseekcoder-33b',
  'deepseek-7b': 'deepseekcoder-7b',
  'mistral-7b': 'mistral-7b-instruct',
  'gpt4': 'gpt-4',
  'gpt3.5': 'gpt-3.5',
}

def sample(ty, llm, cwe, output, amount):
  specs = json.load(open(f"shared/v2/outputs/common/{mapping[llm]}/cwe-{cwe}/api_labels_{llm_name[llm]}.json"))
  filtered_specs = [s for s in specs if "type" in s and s["type"] == ty]
  random.shuffle(filtered_specs)
  sampled_specs = filtered_specs[:amount]
  json.dump(sampled_specs, open(f"{output}/sampled_{ty}_{llm}_{cwe}.json", "w"))

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="shared/v2/sampled-specs")
parser.add_argument("--amount", type=int, default=10)
parser.add_argument("--seed", type=int, default=1234)
args = parser.parse_args()

random.seed(args.seed)

os.makedirs(args.output, exist_ok=True)
for ty in ["source", "sink"]:
  for llm in mapping.keys():
    for cwe in ["022", "078", "079", "094"]:
      sample(ty, llm, cwe, args.output, args.amount)

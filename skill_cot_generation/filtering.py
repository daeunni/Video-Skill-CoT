import time, argparse, nncore
import ast, random
# random.seed(42)  
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from openai import OpenAI
import json, re, time
import ast, os
from tqdm import tqdm 
import random
random.seed(42)
from openai import AzureOpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import warnings;warnings.filterwarnings('ignore')
plt.style.use('default')  
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
from adjustText import adjust_text
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
from wordcloud import WordCloud

# python probable_skill_gpt.py --dataset='esp'


client = AzureOpenAI(
            azure_endpoint = "", 
            api_key= "", 
            api_version= "", 
            )
llm_name = "gpt-4-32k"

def cot_filtering(question, answer, cot, llm):

    llm_prompt = f"""
    You are given a question, its ground-truth answer, and a reasoning chain.

    Your task is:
    1. If the reasoning is irrelevant to the answer, return:
    {{ "relevance": "No", "revised_reasoning": null }}

    2. If the reasoning is relevant, do the following:
    - Convert any bullet-point or timestamped list into a natural, coherent paragraph.
    - Ensure the final reasoning clearly ends with a statement of the correct answer.

    Then respond with:
    {{ "relevance": "Yes", "revised_reasoning": "[Revised reasoning with natural paragraph and final answer]" }}

    Keep the original reasoning steps as intact as possible.

    Respond in JSON format only.

    Question: {question}
    Answer: {answer}
    Reasoning: {cot}
    """


    stop = False
    error_count = 0
    output = None
    while not stop:
        try:
            response = client.chat.completions.create(
                model=llm,
                messages=[{"role": "user", "content": [{"type": "text", "text": llm_prompt}]}],
                max_tokens=300,
                temperature=0,
            )
            output = response.choices[0].message.content
            print(output)
            stop = True
        except Exception as e:
            print(f'ERROR for ({question[:30]}...): {e}')
            time.sleep(9)
            error_count += 1
            if error_count > 3:
                output = None
                stop = True

    return output


def process_cot(idx, cur_data):
    question = cur_data['conversations'][0]['value']
    answer = cur_data['conversations'][1]['value']

    try:
        skill_cot = cur_data['cot']
        output = cot_filtering(question, answer, skill_cot, llm_name)
        if output:
            output = json.loads(output)
            return idx, output
    except Exception as e:
        print(f"[Error] idx={idx} | {e}")
    return idx, None

root = './_gemini_cot/random_10k/et_vsi_cine_phy_esp/t3'
origin_sot = nncore.load(os.path.join(root, 'skill_cot_v1.json'))
log_path = os.path.join(root, 'gpt_filtering_log_skill.json') 
filtered_path = os.path.join(root, 'skill_cot_v1_gpt_filtered.json')
filtered_out_path = os.path.join(root, 'skill_cot_v1_gpt_filtered_out.json')


verify_log = []
total_updated_sot = [] ; filtered_sot = []
filtered = 0
save_interval = 100  

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_cot, idx, origin_sot[idx]) for idx in range(len(origin_sot))]

    for count, future in enumerate(tqdm(as_completed(futures), total=len(origin_sot))):
        idx, output = future.result()
        if output:
            verify_log.append(output)
            if output['relevance'] == 'Yes':
                origin_sot[idx]['conversations'][1]['value'] = output['revised_reasoning']
                total_updated_sot.append(origin_sot[idx])
            elif output['relevance'] == 'No':
                filtered_sot.append(origin_sot[idx])
                filtered += 1
 
        # save!
        if (count + 1) % save_interval == 0:
            nncore.dump(verify_log, log_path, indent=4)
            nncore.dump(total_updated_sot, filtered_path, indent=4)
            nncore.dump(filtered_sot, filtered_out_path, indent=4)
            print(f"[{count+1}] Partial save done: {filtered} filtered so far...")

print(f"Filtered ratio: {filtered}/{len(origin_sot)} ({filtered/len(origin_sot)*100:.2f}%)")


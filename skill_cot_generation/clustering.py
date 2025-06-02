import json, re, time, configparser
import ast, os, argparse
import concurrent.futures
os.environ["OPENBLAS_NUM_THREADS"] = "32"
from tqdm import tqdm 
import random
from itertools import repeat
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

# python probable_skill_cluster.py --dataset='et' 


def skill_summarization_prompt_single(item, llm):
    long_text = item['step']
    # prompt = f"""You are an expert in video understanding and visual reasoning.

    # Given a long-form description of a visual reasoning step, your task is to rewrite it as a short, concise skill sentence.  
    # Keep the key idea intact, focusing on what type of visual reasoning or observation is being performed.  
    # Avoid vague words like “reasoning” or “analysis” unless essential. The result should be 6–12 words long, clear, and specific.

    # Input:
    # {long_text}

    # Output:"""

    prompt = f"""You are an expert in video understanding and visual reasoning.

    Given a long-form description of a visual reasoning step, your task is to rewrite it as a short, concise skill phrase.

    - Keep the core visual reasoning concept intact.
    - Remove specific object names (e.g., "TV", "sofa", "John") and replace them with generic terms like "object", "scene", or "person".
    - Focus on the *type* of visual observation or inference being made.
    - Avoid vague words like “reasoning”, “analysis”, or “understanding” unless necessary.
    - The final output should be 6–12 words long, clear, and generalizable across videos.

    Input:
    {long_text}

    Output:"""


    error_count = 0
    while error_count < 3:
        try:
            response = client.chat.completions.create(
                model=llm,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                max_tokens=300,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            error_count += 1
    return None



if __name__=='__main__' : 
    parser = argparse.ArgumentParser(description="Data selection script")
    parser.add_argument("--level", type=str, default='mid')
    parser.add_argument("--dataset", type=str, default='vsi')
    args = parser.parse_args()

    # Set up your own OpenAI API 
    config = configparser.ConfigParser()
    config.read('config.ini')
    client = AzureOpenAI(
                azure_endpoint = config.get("openai", "azure_endpoint"), 
                api_key= config.get("openai", "api_key"), 
                api_version= config.get("openai", "api_version"), 
                )
    llm_name = "gpt-4-1106"

    # llm prompting 
    root = './_probable_skill_clustering'

    all_domains = [args.dataset]
    mode_name = 'whole_task_skill_abstraction'      

    # dataset-wise loop 
    for dataset in all_domains: 

        print(f'** {dataset} dataset clustering start!')

        all_high_level_skills = [] ; all_low_level_skills = [] ; all_mid_level_skills = []
        save_path = f"./_probable_skill_clustering/{dataset}/{mode_name}.json"

        task_files = os.listdir(os.path.join(root, dataset))
        
        # else : 
        # task-wise loop 
        for task in task_files : 
            file_path = os.path.join(root, dataset, task)
            with open(file_path, 'r') as f:
                file_content = f.read()
            cur_task_output = json.loads(file_content)

            # Datapoint-wise loop 
            for i, data in tqdm(enumerate(cur_task_output), total=len(cur_task_output), desc="💡 Skill CoT"):
                try : 
                    high_list = [item['skill'] for item in data]
                    all_high_level_skills.extend(high_list)

                    # Generate mid-level skill summary 
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        low_list = list(executor.map(skill_summarization_prompt_single, data, repeat(llm_name)))
                    cur_task_output[i].append(low_list)     

                    if i % 10 == 0:
                        with open(save_path, "w") as f:
                            json.dump(cur_task_output, f, indent=2)
                    all_low_level_skills.extend(low_list)
                except : 
                    continue 

            assert len(all_high_level_skills) == len(all_low_level_skills)

        

        if args.level == 'high' : 
            print('** Set to high-level skills')
            cur_all_skills = all_high_level_skills
        elif args.level == 'mid' : 
            print('** Set to mid-level skills')
            cur_all_skills = all_mid_level_skills
        else : 
            print('** Set to low-level skills')
            cur_all_skills = all_low_level_skills

        # eliminate none / meaningless words 
        cur_all_skills = [s for s in cur_all_skills if s is not None and "skill" not in s.lower() and "step" not in s.lower()]
 
        print('** Embedding start!')
        model = SentenceTransformer('all-mpnet-base-v2')
        embeddings = model.encode(cur_all_skills, show_progress_bar=True)


        # visualization 
        print('** TSNE start!')
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_result = tsne.fit_transform(embeddings)
        df = pd.DataFrame(tsne_result, columns=["x", "y"])
        df['category'] = cur_all_skills

        # clustering  
        print('** Clustering start!')
        n_clusters = 10 
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings) 
        df["cluster"] = cluster_labels 

        # top-N reasoning skills 
        all_topN_skills = {}

        ## 1. Clustering-based 
        centroids = kmeans.cluster_centers_
        top_N_skill_names = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_embeddings = embeddings[cluster_indices]
            centroid = centroids[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            representative_idx = cluster_indices[np.argmin(distances)]
            representative_text = cur_all_skills[representative_idx]
            top_N_skill_names.append(representative_text)
        print('** Clustering-based: ', top_N_skill_names)
        all_topN_skills['cluster_based'] = top_N_skill_names

        ## 2. Frequency-based 
        skill_freq = Counter(cur_all_skills)
        top_N_skill_names = [skill for skill, _ in skill_freq.most_common(5)]
        print('** Frequency-based: ', top_N_skill_names)
        all_topN_skills['freq_based'] = top_N_skill_names

        # save 
        with open(f"./_probable_skill_clustering/_topN_skills/{mode_name}/{dataset}.json", "w") as f:
            json.dump(all_topN_skills, f, indent=2)

        # clustering visualization w/ representative skills 
        print('** Clustering visualization!')
        cluster_points = defaultdict(list)
        for i, row in df.iterrows():
            cluster_points[row["cluster"]].append((row["x"], row["y"], row["category"]))

        cluster_labels = {}
        cluster_centers = {}
        for cid, points in cluster_points.items():
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            skills = [p[2] for p in points]

            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            top_skill = Counter(skills).most_common(1)[0][0]

            cluster_centers[cid] = (center_x, center_y)
            cluster_labels[cid] = top_skill

        plt.figure(figsize=(12, 9))
        sns.scatterplot(data=df, x="x", y="y", hue="cluster", palette="tab10", s=70, alpha=0.8, legend="full")
        for cid, (cx, cy) in cluster_centers.items():
            plt.annotate(cluster_labels[cid], (cx, cy), fontsize=11, weight='bold', alpha=0.9)

        plt.title(f"t-SNE + KMeans Clustering of Skill Embeddings ({dataset})")
        plt.xlabel("t-SNE X")
        plt.ylabel("t-SNE Y")
        plt.grid(True)
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"./_probable_skill_clustering/clustering_figure/tsne_cluster_{mode_name}_{dataset}.png", dpi=300)  
        plt.show()


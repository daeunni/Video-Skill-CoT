import os, time, argparse, nncore, copy, json, random, configparser
random.seed(42)
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai

# Set up your own Gemini API 
config = configparser.ConfigParser()
config.read('config.ini')

os.environ["GEMINI_API_KEY"] = config.get("gemini", "gemini_api_key")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.get("gemini", "gemini_application_credentials")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')


inst_paths = {
    'et': [os.path.join('./video_instruction_datasets/ET_164k', f"{file}.json") for file in ['dvc', 'epm', 'evs', 'gvq', 'rvc', 'slc', 'tal', 'tvc', 'tvg', 'vhd']],
    'vsi': [os.path.join("./video_instruction_datasets/VSI-Bench/instruction_data_v2", f"{file}_train.json") for file in [
        "obj_appearance_order", "object_abs_distance", "object_counting", "object_rel_direction_easy", "object_rel_direction_hard",
        "object_rel_direction_medium", "object_rel_distance", "object_size_estimation", "room_size_estimation", "route_planning"]],
    'cine': [os.path.join("./video_instruction_datasets/cinepile/task_instruct_data_high", f"{file}.json")
             for file in [ "SettingandTechnicalAnalysis", "Temporal", "ThemeExploration", "CharacterandRelationshipDynamics", "NarrativeandPlotAnalysis"]],
}


def collect_instruction(data_list):
    init_portions = {'et': 0.25, 'vsi': 1.00, 'cine': 0.15, 'phy': 0.05, 'esp': 0.10}
    instruct_dict = {}; skill_info = {}
    for cur_data in data_list:
        all_skills = []
        cur_portion = init_portions[cur_data]
        for cur_path in tqdm(inst_paths[cur_data]):
            cur_name = os.path.basename(cur_path).split('.')[0]
            all_skills.append(cur_name)
            with open(cur_path, "r", encoding="utf-8") as file:
                dt = json.load(file)
            sampled_querys = random.sample(dt, int(len(dt) * cur_portion))
            instruct_dict[cur_name] = sampled_querys
        skill_info[cur_data] = all_skills
    return instruct_dict, skill_info


def ask_gemini(video_data, question, answer, mode, skill_lists):

    if mode == 'normal_cot':
        prompt = f"""Given the video and the question: {question}
The correct answer is: {answer}
Please write a detailed, step-by-step chain of thought that explains how one would arrive at the correct answer. Your explanation should be written as a coherent paragraph rather than a list or dictionary.
Focus solely on visual elements and any on-screen text from the video. Do not use or rely on any audio information such as speech, sound effects, or music.
❗️Do not reveal or repeat the final answer in your reasoning. Focus only on the logical visual steps that would justify the correct answer without stating it explicitly.
Ensure that each step in your reasoning naturally follows from the previous one, and that the overall explanation clearly supports why the provided answer is correct (without mentioning the answer itself)."""
    

    elif mode == 'sot' : 
        prompt = f"""You are a video understanding assistant that performs modular reasoning using visual skills.
                    You are given:
                    - A video-based question
                    - The ground-truth answer
                    - The relevant video content (as input)
                    - A list of domain-specific visual reasoning skills

                    Your task is to:
                    1. Select the skills needed to answer the question.
                    2. For each selected skill:
                    a. Generate a focused sub-question that applies the skill.
                    b. Answer the sub-question using information from the video.

                    Use the exact format below. Make sure the output is valid and parseable.

                    ---

                    Question: {question}
                    Answer: {answer}
                    Relevant Skills:
                    {skill_list}

                    Output Format:
                    {
                    "selected_skills": [ "Skill A", "Skill B", ... ],
                    "skill_reasoning_steps": [
                        {
                        "skill": "Skill A",
                        "sub_question": "...",
                        "output": "..."
                        },
                        {
                        "skill": "Skill B",
                        "sub_question": "...",
                        "output": "..."
                        }
                    ]
                    }
                    """

        final_reasoning = f"""You are a reasoning assistant that combines modular skill outputs to solve a complex video understanding task.

        Given the original question and a set of skill-based outputs derived from the video, 
        your task is to use these as evidence to perform multi-step reasoning and produce a final answer.

        Use the skill outputs to guide your reasoning. Do not copy them verbatim—use them as evidence in your own words.

        ---

        Question: {question}

        Skill Outputs:
        {skill_output_text}

        Final Answer with Reasoning:
        """


    contents = [
        {"mime_type": "video/mp4", "data": video_data},
        prompt
    ]

    try:
        response = model.generate_content(contents, generation_config={'temperature': 0.0})
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return None


def ask_gemini_worker(i, sample, mode, skill_lists):
    sample = copy.deepcopy(sample)
    try:
        with open(sample['video'], "rb") as video_file:
            video_data = video_file.read()
        question = sample['conversations'][0]['value'].split('Please')[0].replace('<image>', '')
        answer = sample['conversations'][1]['value']
        cot = ask_gemini(video_data, question, answer, mode, skill_lists)
        sample['cot'] = cot
        return i, sample
    except Exception as e:
        print(f"[{i}] Failed: {e}")
        return i, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cine')
    parser.add_argument("--sub_tasks", type=str, default='all')
    parser.add_argument("--mode", type=str, default='skill_cot')       # cot vs sot 
    args = parser.parse_args()

    # max_dataset_num = 1000 
    mode = args.mode

    instruct_dict, skill_info = collect_instruction([args.dataset])         
    if args.sub_tasks == 'all' : 
        tasks = list(instruct_dict.keys())
    else : 
        tasks = [args.sub_tasks]

    save_root = f'./_gemini_cot/{args.dataset}/{args.mode}'
    os.makedirs(save_root, exist_ok=True)

    # load skills per args.dataset 
    skill_lists = None 

    for cur_task in tasks:
        print('** Cur_task:', cur_task)
        pred_path = nncore.join(save_root, cur_task + '.json')
        anno = instruct_dict[cur_task]


        if os.path.exists(pred_path) :         # if already exists -> pass 
            print(f'** Pass {cur_task} task!')
            continue

        with ThreadPoolExecutor(max_workers=5) as executor:
            iter_num = len(anno)

            futures = [executor.submit(ask_gemini_worker, i, anno[i], mode, skill_lists) for i in range(iter_num)]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating COT for {cur_task}"):
                try : 
                    i, updated = future.result()
                    if updated is not None:
                        anno[i] = updated
                        print(f"[{i}] COT: {updated['cot'][:80]}...")

                    if (i + 1) % 10 == 0:
                        nncore.dump(anno, pred_path, indent=4)
                        print(f"💾 Intermediate save at {i + 1}")
                except : 
                    continue 

        nncore.dump(anno, pred_path, indent=4)
        print(f"✅ Saved: {pred_path}")

        # break  

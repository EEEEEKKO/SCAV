import json
import os

category_map = {
    1: "Illegal Activity",
    2: "Child Abuse Content",
    3: "Hate/Harass/Violence",
    4: "Malware",
    5: "Physical Harm",
    6: "Economic Harm",
    7: "Adult Content",
    8: "Fraud Deception",
    9: "Political Campaigning",
    10: "Privacy Violation Activity",
    11: "Tailored Financial Advice",
}



if __name__ == "__main__":
    source_path = "tmp.txt"
    with open(source_path, 'r') as f:
        data = f.readlines()

    cnt = 0
    cur_id = 1
    for d in data:
        if cur_id == 2:
            cur_id += 1
        save_path = "HEx-PHI"
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f'category_{cur_id}.jsonl'), 'a') as f:
            new_data = {
                "prompt": d,
                "category": category_map[cur_id]
            }
            f.write(json.dumps(new_data, ensure_ascii=False) + "\n")
        cnt += 1
        if cnt == 30:
            cur_id += 1
            cnt = 0
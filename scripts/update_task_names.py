import os
import json
import glob

tasks = "0 1 2 3 4 5 6 7 8 13 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61".split()

dataset_mapping = {
    # First list (short names) -> Second list (full dataset paths/names)
    "ade_corpus_v2_classification": "ade-benchmark-corpus,ade_corpus_v2,,Ade_corpus_v2_classification",
    "anli": "facebook,anli,,None",
    "arc_challenge": "allenai,ai2_arc,,ARC-Challenge",
    "arc_easy": "allenai,ai2_arc,,ARC-Easy",
    "banking77": "legacy-datasets,banking77,,None",
    "boolean_expressions": "data,boolean_expressions.json,,boolean_expressions",
    "boolq": "super_glue,,boolq",
    "circa": "google-research-datasets,circa,,None",
    "cola": "nyu-mll,glue,,cola",
    "commonsense_qa": "tau,commonsense_qa,,None",
    "copa": "super_glue,,copa",
    "cryptonews_articles_with_price_momentum_labels": "SahandNZ,cryptonews-articles-with-price-momentum-labels,,None",
    "disaster_response_messages": "community-datasets,disaster_response_messages,,None",
    "disfl_qa": "data,disfl_qa.json,,disfl_qa",
    "elementary_math_qa_question_only": "data,elementary_math_qa_question_only.json,,elementary_math_qa_question_only",
    "fig_qa": "nightingal3,fig-qa,,None",
    "formal_fallacies_syllogisms_negation": "data,formal_fallacies_syllogisms_negation.json,,formal_fallacies_syllogisms_negation",
    "hate": "cardiffnlp,tweet_eval,,hate",
    "high": "ehovy,race,,high",
    "hyperbaton": "data,hyperbaton.json,,hyperbaton",
    "imbalanced": "clinc,clinc_oos,,imbalanced",
    "intersect_geometry": "data,intersect_geometry.json,,intersect_geometry",
    "machine_paraphrase_dataset": "jpwahle,machine-paraphrase-dataset,,None",
    "gsm8k": "openai,gsm8k,,main",
    "medmcqa": "openlifescienceai,medmcqa,,None",
    "mmlu_pro": "TIGER-Lab,MMLU-Pro,,None",
    "mnist_ascii": "data,mnist_ascii.json,,mnist_ascii",
    "mnli": "nyu-mll,glue,,mnli",
    "mrpc": "nyu-mll,glue,,mrpc",
    "multistep_arithmetic": "data,multistep_arithmetic.json,,multistep_arithmetic",
    "object_counting": "data,object_counting.json,,object_counting",
    "openmathinstruct_2": "nvidia,OpenMathInstruct-2,,None",
    "overruling": "LawInformedAI,overruling,,None",
    "qa_wikidata": "data,qa_wikidata.json,,qa_wikidata",
    "qasc": "allenai,qasc,,None",
    "qnli": "nyu-mll,glue,,qnli",
    "qqp": "nyu-mll,glue,,qqp",
    "quail": "textmachinelab,quail,,None",
    "reasoning_about_colored_objects": "data,reasoning_about_colored_objects.json,,reasoning_about_colored_objects",
    "rte": "super_glue,,rte",
    "sciq": "allenai,sciq,,None",
    "sports_understanding": "data,sports_understanding.json,,sports_understanding",
    "sst2": "nyu-mll,glue,,sst2",
    "temporal_sequences": "data,temporal_sequences.json,,temporal_sequences",
    "tokenized": "deepmind,aqua_rat,,tokenized",
    "toxicchat0124": "lmsys,toxic-chat,,toxicchat0124",
    "tracking_shuffled_objects": "data,tracking_shuffled_objects.json,,tracking_shuffled_objects",
    "tweet_topic_single": "cardiffnlp,tweet_topic_single,,None",
    "twitter_financial_news_sentiment": "zeroshot,twitter-financial-news-sentiment,,None",
    "typescript_chunks": "bleugreen,typescript-chunks,,None",
    "unit_conversion_si_conversion": "data,unit_conversion_si_conversion.json,,unit_conversion_si_conversion",
    "web_of_lies": "data,web_of_lies.json,,web_of_lies",
    "wic": "super_glue,,wic",
    "wsc": "super_glue,,wsc",
    "masakhanews_yor": "masakhane,masakhanews,,yor",
    "masakhanews_swa": "masakhane,masakhanews,,swa",
    "masakhanews_amh": "masakhane,masakhanews,,amh",
    "twitter_complaints": "data,twitter_complaints.json,,twitter_complaints",
    "mmlu": "cais,mmlu,,None",
    "unarxive_imrad_clf": "saier,unarXive_imrad_clf,,None",
    "polish_sequence_labeling": "data,polish_sequence_labeling.json,,polish_sequence_labeling",
    "stackoverflow_questions": "pacovaldez,stackoverflow-questions,,None",
}

comma_names = dataset_mapping.values()
task_keys = "banking77 ade_corpus_v2_classification overruling imbalanced object_counting main typescript_chunks twitter_financial_news_sentiment cryptonews_articles_with_price_momentum_labels stackoverflow_questions swa yor amh disaster_response_messages tweet_topic_single machine_paraphrase_dataset unarxive_imrad_clf anli circa mmlu mmlu_pro tokenized commonsense_qa openmathinstruct_2 sciq medmcqa qasc quail toxicchat0124 high boolq copa rte wic wsc cola mnli mrpc qnli qqp sst2 arc_easy arc_challenge fig_qa boolean_expressions hyperbaton sports_understanding temporal_sequences tracking_shuffled_objects web_of_lies multistep_arithmetic reasoning_about_colored_objects formal_fallacies_syllogisms_negation mnist_ascii elementary_math_qa_question_only intersect_geometry unit_conversion_si_conversion hate twitter_complaints polish_sequence_labeling disfl_qa qa_wikidata".split()


special_cases = {
    "mmlu": "mmlu_combined",
    "wic": "super_glue_wic",
    "main": "gsm8k",
}

from functools import lru_cache


def to_comma_name(dataset, task):
    return f"{dataset},,{task}".replace("/", ",")


@lru_cache(maxsize=1)
def get_valid_task_mappings(tasks_yaml_path="tasks.yaml"):
    """Load tasks.yaml and return mapping of valid task keys to their comma-separated dataset names."""
    import yaml

    with open(tasks_yaml_path, "r") as f:
        content = yaml.safe_load(f)

    valid_tasks = {}
    for task_key, task_info in content.items():
        path = task_info.get("path", [])
        path = "_".join(path) if len(path) > 0 else task_info.get("json_file", "")
        assert path, f"Task {task_key} has no path or json_file in tasks.yaml"
        assert (
            path not in valid_tasks
        ), f"Duplicate path {path} for tasks {valid_tasks[path]} and {task_key}"
        valid_tasks[path] = task_key

    return valid_tasks


def string_to_task_key(
    input_string, tasks_yaml_path="tasks.yaml", special_cases=special_cases
):
    """
    Convert any string to a task key from tasks.yaml using longest common substring matching.

    Args:
        input_string: The string to convert
        tasks_yaml_path: Path to tasks.yaml file

    Returns:
        The matched task key from tasks.yaml

    Raises:
        ValueError: If no unique match can be determined
    """

    valid_tasks_mapping = get_valid_task_mappings(tasks_yaml_path)

    # If exact match, return it
    if input_string in special_cases:
        return special_cases[input_string]

    # Find best match using longest common substring
    def longest_common_substring(s1, s2):
        s1, s2 = s1.lower().replace("_", "").replace("-", "").replace(",", "").replace(
            "/", ""
        ), s2.lower().replace("_", "").replace("-", "").replace(",", "").replace(
            "/", ""
        )
        m, n = len(s1), len(s2)
        max_len = 0
        for i in range(m):
            for j in range(n):
                k = 0
                while i + k < m and j + k < n and s1[i + k] == s2[j + k]:
                    k += 1
                max_len = max(max_len, k)
        return max_len

    best_matches = []
    best_len = 0

    for path, task in valid_tasks_mapping.items():
        lcs_len = longest_common_substring(input_string, path)
        if lcs_len > best_len:
            best_len = lcs_len
            best_matches = [task]
        elif lcs_len == best_len and lcs_len > 0:
            best_matches.append(task)

    if len(best_matches) == 0:
        raise ValueError(f"No match found for '{input_string}' in tasks.yaml")
    elif len(best_matches) > 1:
        raise ValueError(
            f"Multiple matches for '{input_string}': {best_matches} (LCS length: {best_len})"
        )

    return best_matches[0]


task_idx_mapping = {
    f"_task{idx}_": f"_{string_to_task_key(task)}_"
    for idx, task in enumerate(task_keys)
}
task_key_mapping = {task: string_to_task_key(task) for task in task_keys}
comma_name_to_tasks_mapping = {
    comma_name: string_to_task_key(comma_name) for comma_name in comma_names
}


def safe_replace(input_string, key, value):
    if key == value:
        return input_string
    if key in value:
        while value in input_string:
            input_string = input_string.replace(value, key)
    input_string = input_string.replace(key, value)
    return input_string


@lru_cache(maxsize=100)
def replace_name(old_fname):
    new_fname = "NOT FOUND!"
    if ",," in old_fname:
        for k, v in comma_name_to_tasks_mapping.items():
            if k in old_fname:
                new_fname = safe_replace(old_fname, k, v)
                break
    elif "_task" in old_fname and any(k in old_fname for k in task_idx_mapping):
        for k, v in task_idx_mapping.items():
            if k in old_fname:
                new_fname = safe_replace(old_fname, k, v)
                break
    elif any(k in old_fname for k in task_key_mapping):
        for k, v in task_key_mapping.items():
            if k in old_fname:
                new_fname = safe_replace(old_fname, k, v)
                break
    else:
        new_fname = None
    if new_fname == "NOT FOUND!":
        raise ValueError(f"Could not find mapping for {old_fname}")

    if new_fname is None:
        return None
    else:
        while "glue_glue" in new_fname:
            new_fname = new_fname.replace("glue_glue", "glue")
        while "super_glue_super_glue" in new_fname:
            new_fname = new_fname.replace("super_glue_super_glue", "super_glue")
        while "mmlu_combined_pro" in new_fname:
            new_fname = new_fname.replace("mmlu_combined_pro", "mmlu_pro")
        return new_fname


def _replace_json_content(old_content):
    updated = False
    if "dataset_name" in old_content and "task" in old_content:
        dataset_name = old_content["dataset_name"]
        task = old_content["task"]
        comma_name = to_comma_name(dataset_name, task)
        new_task = string_to_task_key(comma_name)
        old_content["task"] = new_task
        del old_content["dataset_name"]
        updated = True

    if "run_name" in old_content:
        new_run_name = replace_name(old_content["run_name"])
        if new_run_name and new_run_name != old_content["run_name"]:
            old_content["run_name"] = new_run_name
            updated = True

    return old_content, updated


def _replace_txt_content(lines):
    updated = False
    if "best_lora" in lines[0]:
        old_first_line = lines[0]
        new_first_line = replace_name(old_first_line)
        if new_first_line and new_first_line != old_first_line:
            updated = True
            lines[0] = new_first_line
    return "\n".join(lines), updated


def replace_name_in_dir(path, dry_run=False):
    if path.endswith("job_summary.json"):
        # print(f"Processing JSON file: {path}")
        try:
            with open(path, "r") as f:
                data = json.load(f)
            new_data, updated = _replace_json_content(data)
            if updated:
                if not dry_run:
                    if not os.path.exists(path + ".back"):
                        os.rename(path, path + ".back")
                    with open(path, "w") as f:
                        json.dump(new_data, f)
                    print(f"  Updated: {path}, backup: {path}.back")
                else:
                    print(f"  [DRY RUN] Would update {path}")
        except PermissionError:
            print(f"  Permission denied: {path}")
    elif path.endswith(".txt"):
        # print(f"Processing TXT file: {path}")
        try:
            with open(path, "r") as f:
                lines = f.read().split("\n")
            new_lines, updated = _replace_txt_content(lines)
            if updated:
                if not dry_run:
                    if not os.path.exists(path + ".back"):
                        os.rename(path, path + ".back")
                    with open(path, "w") as f:
                        f.write(new_lines)
                    print(f"  Updated: {path}, backup: {path}.back")
                else:
                    print(f"  [DRY RUN] Would update {path}")
        except PermissionError:
            print(f"  Permission denied: {path}")
    elif os.path.isdir(path):
        # print(f"Processing directory: {path}")
        contents = os.listdir(path)
        for old_fname in contents:
            new_fname = replace_name(old_fname)
            if new_fname and new_fname != old_fname:
                try:
                    print(f"  Renaming: {old_fname} -> {new_fname}")
                    if not dry_run:
                        os.rename(
                            os.path.join(path, old_fname), os.path.join(path, new_fname)
                        )
                    else:
                        print(f"  [DRY RUN] Would rename")
                except PermissionError:
                    print(f"  Permission denied: {old_fname}")
        contents = os.listdir(path)
        for fname in contents:
            replace_name_in_dir(os.path.join(path, fname), dry_run=dry_run)


def delete_backups(path, dry_run=False):
    if path.endswith(".back"):
        print(f"Deleting backup: {path}")
        if not dry_run:
            os.remove(path)
        else:
            print(f"  [DRY RUN] Would delete")
    elif os.path.isdir(path):
        print(f"Processing directory: {path}")
        contents = os.listdir(path)
        for fname in contents:
            delete_backups(os.path.join(path, fname), dry_run=dry_run)


def revert_backups(path, dry_run=False):
    if path.endswith(".back"):
        original_path = path[:-5]
        print(f"Reverting backup: {path} -> {original_path}")
        if not dry_run:
            os.rename(path, original_path)
        else:
            print(f"  [DRY RUN] Would revert")
    elif os.path.isdir(path):
        print(f"Processing directory: {path}")
        contents = os.listdir(path)
        for fname in contents:
            revert_backups(os.path.join(path, fname), dry_run=dry_run)


if __name__ == "__main__":
    # Example usage
    # Replace "shared_space" with the directory you want to process
    # It will recursively process all files and directories under it
    DRY_RUN = True

    replace_name_in_dir("shared_space", dry_run=DRY_RUN)
    # revert_backups("shared_space", dry_run=DRY_RUN)
    # delete_backups("shared_space", dry_run=DRY_RUN)

import os
import glob
import json
from collections import OrderedDict
from utils.misc_utils import to_comma_name
import pandas as pd


# task_name mapping
dataset_mapping = OrderedDict(
    {
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
    }
)

model_eval_path = "outputs/evaluation/"
eval_groups = os.listdir(model_eval_path)
eval_job_summary_paths_per_group = [
    glob.glob(f"{model_eval_path}/{group}/*/job_summary.json") for group in eval_groups
]
rows = []

# temp, add lora baseline
datasets = "legacy-datasets,banking77,,None	community-datasets,disaster_response_messages,,None	cardiffnlp,tweet_topic_single,,None	jpwahle,machine-paraphrase-dataset,,None	facebook,anli,,None	google-research-datasets,circa,,None	cais,mmlu,,None	ade-benchmark-corpus,ade_corpus_v2,,Ade_corpus_v2_classification	TIGER-Lab,MMLU-Pro,,None	deepmind,aqua_rat,,tokenized	tau,commonsense_qa,,None	nvidia,OpenMathInstruct-2,,None	allenai,sciq,,None	openlifescienceai,medmcqa,,None	allenai,qasc,,None	textmachinelab,quail,,None	lmsys,toxic-chat,,toxicchat0124	ehovy,race,,high	LawInformedAI,overruling,,None	super_glue,,boolq	super_glue,,copa	super_glue,,rte	super_glue,,wic	super_glue,,wsc	nyu-mll,glue,,cola	nyu-mll,glue,,mnli	nyu-mll,glue,,mrpc	nyu-mll,glue,,qnli	nyu-mll,glue,,qqp	clinc,clinc_oos,,imbalanced	nyu-mll,glue,,sst2	allenai,ai2_arc,,ARC-Easy	allenai,ai2_arc,,ARC-Challenge	nightingal3,fig-qa,,None	data,boolean_expressions.json,,boolean_expressions	data,hyperbaton.json,,hyperbaton	data,sports_understanding.json,,sports_understanding	data,temporal_sequences.json,,temporal_sequences	data,tracking_shuffled_objects.json,,tracking_shuffled_objects	data,web_of_lies.json,,web_of_lies	data,object_counting.json,,object_counting	data,multistep_arithmetic.json,,multistep_arithmetic	data,reasoning_about_colored_objects.json,,reasoning_about_colored_objects	data,formal_fallacies_syllogisms_negation.json,,formal_fallacies_syllogisms_negation	data,mnist_ascii.json,,mnist_ascii	data,elementary_math_qa_question_only.json,,elementary_math_qa_question_only	data,intersect_geometry.json,,intersect_geometry	data,unit_conversion_si_conversion.json,,unit_conversion_si_conversion	cardiffnlp,tweet_eval,,hate	data,twitter_complaints.json,,twitter_complaints	data,polish_sequence_labeling.json,,polish_sequence_labeling	openai,gsm8k,,main	data,disfl_qa.json,,disfl_qa	data,qa_wikidata.json,,qa_wikidata	bleugreen,typescript-chunks,,None	zeroshot,twitter-financial-news-sentiment,,None	SahandNZ,cryptonews-articles-with-price-momentum-labels,,None"
accuracies = "0.6902597403	0.8843666793	0.8022947926	0.9205841168	0.581	0.8517653925	0.5927272727	0.8537414966	0.1262458472	0.2952755906	0.6553846154	0.1396558623	0.439	0.7124	0.9496314496	0.723902439	0.9402	0.7984562607	0.9583333333	0.7932131495	0.95	0.8634538153	0.7163904236	0.4642857143	0.8235981308	0.8126	0.6863768116	0.865	0.838	0.7144	0.9056	0.8813131313	0.6919795222	0.8801652893	0.779	0.935	0.8947939262	1	0.2746666667	0.5201342282	0.3781163435	0.1966666667	0.76625	0.6028169014	0.3802	0.1937581274	0.208	0.1766666667	0.5851851852	0.9072463768	0.06609642302	0.7111448067	0.46	0.7369193154	0.9936	0.7623036649	0.6552"
datasets = datasets.split("\t")
accuracies = accuracies.split("\t")
assert len(datasets) == len(accuracies)
group_dict = {
    "group": "lora_baseline",
    "num_jobs": len(datasets),
    "last_update_time": 0,
    "num_steps": 0,
}
for dataset, acc in zip(datasets, accuracies):
    group_dict[dataset] = float(acc)
print(f"lora_baseline: {len(datasets)} test values")
rows.append(group_dict)

# load job summaries from model_eval_path
for group, summary_paths in zip(eval_groups, eval_job_summary_paths_per_group):
    print(f"{group}: {len(summary_paths)} job summaries")
    group_dict = {
        "group": group,
        "num_jobs": len(summary_paths),
    }
    last_update_times = []
    num_steps = []
    for summary_path in summary_paths:
        with open(summary_path, "r") as f:
            tmp = json.load(f)
            comma_name = to_comma_name(tmp["dataset_name"], tmp["task"])
            accuracy = tmp["test/exact_string_match_accuracy"]
            group_dict[comma_name] = round(accuracy, 3)
            # get the last modified time
            mtime = os.path.getmtime(summary_path)
            last_update_times.append(mtime)
        with open(
            summary_path.replace("job_summary.json", "tuning_job_summary.json"), "r"
        ) as f:
            tmp = json.load(f)
            comma_name = to_comma_name(tmp["dataset_name"], tmp["task"])
            steps = tmp.get("best_global_step", None)
            if steps is not None:
                num_steps.append(steps)

    if num_steps:
        group_dict["num_steps"] = int(sum(num_steps) / len(num_steps))
    else:
        group_dict["num_steps"] = 0

    if last_update_times:
        group_dict["last_update_time"] = max(last_update_times)
    else:
        group_dict["last_update_time"] = 0

    rows.append(group_dict)


rows.sort(key=lambda x: x["last_update_time"], reverse=True)

# convert to horizontal dataframe
df_long = pd.DataFrame(rows)
result_columns = sorted(
    [
        col
        for col in df_long.columns.tolist()
        if col not in ["group", "num_jobs", "last_update_time", "num_steps"]
    ],
    key=lambda x: (x.lower(), x),
)
# sort based on column names
sorted_cols = ["group", "num_jobs", "last_update_time", "num_steps"] + result_columns
df_long = df_long[sorted_cols]
print(df_long)
df_long.to_csv("outputs/eval_table_long.csv", index=False)


# convert to  dataframe (with short names)
df_short = df_long
df_short = df_short.rename(
    columns={full_name: short_name for short_name, full_name in dataset_mapping.items()}
)
redundant = set(result_columns) - set(dataset_mapping.values())
df_short = df_short.drop(columns=list(redundant))
sorted_cols = ["group", "num_jobs", "last_update_time", "num_steps"] + list(
    dataset_mapping.keys()
)
df_short = df_short[sorted_cols]
print(df_short)
df_short.to_csv("outputs/eval_table_short.csv", index=False)

# export to horizontal csv
realpath = os.path.realpath("outputs/eval_table_short.csv")
print(f"Saved summary to {realpath}")
print("Transfer command:")
print(f"scp vulcan:{realpath} ~/Desktop/")

# export to vertical csv

import pandas as pd
import json
import datetime

filepath = './Documents/Other/ProjectAlpha/Output Files/'
files = ['output-new-poems-partial.txt', 'output-new-poems-full.txt', 'output-new-essay-0-all.txt', 'output-new-essay-1-all.txt', 'output-new-essay-2-all.txt', 'output-new-essay-3-all.txt']
dfs = []

for file in files:
    f = open(f'{filepath}Raw/{file}')
    d = json.load(f)
    d = d["data"]
    data = []
    for item in d:
        obj = {}
        mask_word_count = item["without_stop"]["mask_correct"] + item["without_stop"]["mask_incorrect"]
        generate_word_count = item["without_stop"]["generate_correct"] + item["without_stop"]["generate_incorrect"]
        obj["Author"] = item["metadata"]["author"]
        obj["Title"] = item["metadata"]["title"]
        obj["Mask Correct Predictions"] = round(item["without_stop"]["mask_correct"] / mask_word_count, 2) * 100
        obj["Mask Average Word Similarity"] = round(item["without_stop"]["mask_similarity"] / mask_word_count)
        obj["Mask Incorrect Word Similarity"] = round(item["without_stop"]["mask_incorrect_similarity"] / item["without_stop"]["mask_incorrect"])
        obj["Mask Is Top 10"] = round(item["without_stop"]["is_top_10"] / mask_word_count, 2) * 100
        obj["Generate Correct Predictions"] = round(item["without_stop"]["generate_correct"] / generate_word_count, 2) * 100
        obj["Generate Average Word Similarity"] = round(item["without_stop"]["generate_similarity"] / generate_word_count)
        obj["Generate Incorrect Word Similarity"] = round(item["without_stop"]["generate_incorrect_similarity"] / item["without_stop"]["generate_incorrect"])
        #obj["Average Sentence Similarity"] = round(item["with_stop"]["sentence_similarity"] / item["metadata"]["sentence_counter"])
        # #obj["Average NSP Score"] = round(item["with_stop"]["nsp_score"] / (item["metadata"]["sentence_counter"] - 1)) if item["metadata"]["sentence_counter"] > 1 else "N/A"
        if " " in obj["Author"]:
            spl = obj["Author"].split(" ")
            obj["Author"] = spl[len(spl) - 1] + ", " + " ".join(spl[0:len(spl) - 1])
        data.append(obj)
    df = pd.DataFrame(data)
    df = df.sort_values("Author").reset_index(drop=True)
    indexes = df.index.tolist()
    indexes.append(' ')
    df.reindex(indexes)
    df.index += 1
    i = 0
    df.loc['Mean'] = df.mean(numeric_only=True)
    df = df.style.format(precision=0)
    dfs.append(df)

timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
timestamp = timestamp.split('.')[0]
with pd.ExcelWriter(f'{filepath}Excel/output [{timestamp}].xlsx') as excel_writer:
    index = 0
    for df in dfs:
        df.to_excel(excel_writer, sheet_name=files[index].split(".")[0], index=False)
        index += 1
import json
import nltk
import re
import torch

import gensim.downloader as vec_api
from numpy import False_
from transformers import BertTokenizer, BertForMaskedLM, BertForNextSentencePrediction, pipeline
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SEARCH_LIMIT = 30522
result_list = ["with", "without", "only"]

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

import os
from multiprocessing import Pool, cpu_count
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from stats import Stats
import formatters

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

vec_model = vec_api.load("fasttext-wiki-news-subwords-300")
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForMaskedLM.from_pretrained('bert-large-uncased', return_dict=True)
sentence_model = SentenceTransformer('bert-large-nli-mean-tokens')
nsp_model = BertForNextSentencePrediction.from_pretrained('bert-large-uncased')
generator = pipeline('text-generation', model="facebook/opt-1.3b")
stop_word_list = set(stopwords.words('english'))

def pred_word(_input):
    txt = _input[0]
    correct_word = _input[1]
    generate_input = _input[2]
    tokenized = tokenizer.encode_plus(txt, return_tensors = "pt")
    if tokenized['input_ids'].size(dim=1) > 512:
        print("error with giant sentence")
        return {
            "mask_result": "UNKNOWN",
            "mask_similarity": "Not Found",
            "mask_pred_word": "UNKNOWN",
            "generate_result": "UNKNOWN",
            "generate_similarity": "Not Found",
            "generate_pred_word": "UNKNOWN"
        }
    mask_index = torch.where(tokenized["input_ids"][0] == tokenizer.mask_token_id)
    mask_output = model(**tokenized)

    generate_text = "N/A"
    generate_result = "UNKNOWN"
    generate_similarity = "Not Found"
    if generate_input != None and len(generate_input) > 0:
        generate_length = (len(generate_input.split()) * 2) + 5
        try:
            generate_output = generator(generate_input, max_length=generate_length)
            generate_output = generate_output[0]['generated_text'].strip().split(generate_input)
            if len(generate_output) > 1:
                generate_output = generate_output[1].strip().split()
                generate_output = generate_output[0].lower() if len(generate_output) > 0 else ""
                generate_output = re.search(r"[\w\-']+", generate_output)
                if generate_output != None:
                    generate_text = generate_output.group(0)
                    generate_result = "CORRECT" if generate_text == correct_word else "INCORRECT"
                    try:
                        generate_similarity = round(100 * float(vec_model.similarity(correct_word, generate_text)), 2)
                    except:
                        generate_similarity = "Not Found"
        except:
            print("error with the unknown generate thing")

    logits = mask_output.logits
    softmax = F.softmax(logits, dim = -1)
    try:
        mask_word = softmax[0, mask_index, :]
    except:
        print("error occurred with line 163")
        return {
            "mask_result": "UNKNOWN",
            "mask_similarity": "Not Found",
            "mask_pred_word": "UNKNOWN",
            "generate_result": "UNKNOWN",
            "generate_similarity": "Not Found",
            "generate_pred_word": "UNKNOWN"
        }
    top = torch.topk(mask_word, SEARCH_LIMIT, dim = 1)[1][0]
    tokens = []
    for token in top:
        word = tokenizer.decode([token])
        if re.search(r'^\W+$', word) == None:
            tokens.append(word)
        if len(tokens) == 10:
            break
    mask_result = "CORRECT" if tokens[0] == correct_word else "INCORRECT"
    is_top_10 = correct_word in tokens
    try:
        mask_similarity = round(100 * float(vec_model.similarity(correct_word, tokens[0])), 2)
    except:
        mask_similarity = "Not Found"
    is_stop = "TRUE" if correct_word in stop_word_list else "FALSE"
    if 1 == 0:
        formatters.print_word(
        masked_word=correct_word,
        mask_predicted_word=tokens[0],
        mask_prediction_result=mask_result,
        mask_similarity=mask_similarity,
        generate_predicted_word=generate_text,
        generate_prediction_result=generate_result,
        generate_similarity=generate_similarity,
        stop_word = is_stop
        )
    return {
        "mask_result": mask_result,
        "mask_similarity": mask_similarity,
        "mask_pred_word": tokens[0],
        "generate_result": generate_result,
        "generate_similarity": generate_similarity,
        "generate_pred_word": generate_text,
        "is_top_10": is_top_10,
        "is_stop": correct_word in stop_word_list
    }

def test2(input):
   return {
        "mask_result": "CORRECT",
        "mask_similarity": 1,
        "mask_pred_word": "word",
        "generate_result": "CORRECT",
        "generate_similarity": 1,
        "generate_pred_word": "word",
        "is_top_10": False,
        "is_stop": False
    }

class Predictor2:  
  def __init__(self, args):
    self.args = args
    self.arguments = self.args['args']
  
  def generate_inputs(self, text, sentence_list, title_sentence=False):
    spl = text.split(" ")
    index = 0
    
    proper_nouns = []
    if self.args["ignore_proper"]:
      words = nltk.word_tokenize(text)
      tagged = nltk.pos_tag(words)
      for (word, tag) in tagged:
        if tag == 'NNP':
          proper_nouns.append(word.lower())
    
    inputs = []

    for word in spl:
      word = re.search(r"[\w\-']+", word)
      if word != None and (self.args["ignore_proper"] == False or word.group(0).lower() not in proper_nouns):
        word = word.group(0)
        sentence = ""
        generate_input = ""
        for i in range(len(spl)):
          if i < index:
            generate_input += f"{spl[i]} "
          if i == index:
            sentence += f"{spl[i].replace(word, '[MASK]')} "
          else:
            sentence += f"{spl[i]} "
        generate_context = ""
        if self.args["partial"]:
          if len(sentence_list) > self.args["partial"]:
            sentence_list = sentence_list[-self.args["partial"]:]
        elif self.args["full_input"] == False:
          if title_sentence == False or len(sentence_list) > 1:
            sentence_list = []
        generate_context = f"{' '.join(sentence_list)} {generate_input}".strip()
        inputs.append([sentence.strip(), word.lower(), generate_context])
      index += 1

    return inputs

  def test(self, input):
    return {
        "mask_result": "CORRECT",
        "mask_similarity": 1,
        "mask_pred_word": "word",
        "generate_result": "CORRECT",
        "generate_similarity": 1,
        "generate_pred_word": "word",
        "is_top_10": False,
        "is_stop": False
    }

  def run_predictor(self, input_txt, data=False):
    stats = {}
    for word in result_list:
      stats[f"{word}_stop"] = Stats()

    if self.args["use_tokenizer"]:
      sentences = nltk.sent_tokenize(input_txt)
    elif self.args["sentence_format"]:
      sentences = nltk.sent_tokenize(input_txt)
      for i in range(len(sentences)):
        newsent = re.sub(r"\n+\s*([A-Z])", lambda m: " " + m.group(1).lower(), sentences[i]).strip()
        newsent = re.sub(r"\s{2,}", " ", newsent)
        sentences[i] = newsent
    else:
      sentences = re.split(r'\n+', input_txt.strip())
    title_sentence = data != False and "poem" in data["type"].lower()
    sentences.insert(0, data["title"] if title_sentence else "")
    sentences = list(filter(lambda x: len(x.strip()) > 0, sentences))

    if self.args["nsp_only"]:
      return self.get_nsp(sentences)
    
    sentence_counter = 0
    #score_counter = 0
    inputs = []
    for sentence in sentences:
      sentence_counter += 1
      if title_sentence and sentence_counter == 1:
        continue
      inputs.extend(self.generate_inputs(sentence.strip(), sentences[:sentence_counter-1], title_sentence=title_sentence))

    for input in inputs:
      print(input)
    
    number_of_cores = int(os.getenv('SLURM_CPUS_PER_TASK', cpu_count()))
    print(number_of_cores)

    chunksize, extra = divmod(len(inputs), 4 * number_of_cores)
    if extra:
        chunksize += 1
    outputs = None
    with Pool(number_of_cores) as pool:
      print(1)
      outputs = pool.map(test2, inputs, chunksize=chunksize)

    print(2)
    for res in outputs:
      stats["with_stop"].add_item(res)
      if res["is_stop"]:
        stats["only_stop"].add_item(res)
      else:
        stats["without_stop"].add_item(res)
    
    total_obj = {}
    for word in result_list:
      stats_obj = stats[f"{word}_stop"]
      total_obj[f"{word}_stop"] = stats_obj.get_data()
      partial_total = stats_obj.get_total("mask")
      full_total = stats["with_stop"].get_total("mask")
      print(f"\nResults for {word.upper()} stop words...")
      print(f"{partial_total}/{full_total} {formatters.get_percent(partial_total, full_total)}")
      print()
      stats_obj.print_data()
      print()
      formatters.print_sep()
    
    print("\nResults for sentences...\n")
    print(f"Number of Sentences   = {sentence_counter}")
    print(f"Average Similarity    = {round(stats['with_stop'].get_sentence_similarity() / sentence_counter, 2)}")
    if sentence_counter > 1:
      print(f"Average NSP Score     = {round(stats['with_stop'].get_nsp_score() / (sentence_counter - 1), 2)}")
    print()
    formatters.print_sep()
    
    total_obj["metadata"] = {
      "type": "N/A",
      "title": "N/A",
      "author": "N/A",
      "publish": "N/A",
      "sentence_counter": sentence_counter
    }
    
    if data:
      total_obj["metadata"]["type"] = data["type"]
      total_obj["metadata"]["title"] = data["title"]
      total_obj["metadata"]["author"] = data["author"]
      total_obj["metadata"]["publish"] = data["publish"]
    
    print(json.dumps(total_obj, indent=4))

    return total_obj
  
  def get_nsp(self, sentences):
    total_score = 0
    total_count = 0
    sentences = list(filter(lambda x: len(x.strip()) > 0, sentences))
    if len(sentences) < 2:
      print(f"There is only {len(sentences)} sentence and thus no NSP can be calculated")
      return
    for sentence in sentences:
      for next_sentence in sentences:
        if sentence == next_sentence:
          continue
        encoding = self.tokenizer.encode_plus(sentence, next_sentence, return_tensors='pt')
        if encoding['input_ids'].size(dim=1) > 512:
          continue
        outputs = self.nsp_model(**encoding)[0]
        softmax = F.softmax(outputs, dim = 1)
        total_score += round(float(softmax[0][0].item()) * 100, 2)
        total_count += 1
    print(f"Generated {total_count} sentence pairs from {len(sentences)} sentences")
    print(f"Compare with n(n-1): {len(sentences)*(len(sentences)-1)}")
    print(f"Average score: {round(total_score / total_count, 2)}")
  
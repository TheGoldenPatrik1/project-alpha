import torch
import re
import nltk
import time
import math
import json
import requests
import sys

start = time.time()

from gensim.models import Word2Vec
import gensim.downloader as vec_api
vec_model = vec_api.load("fasttext-wiki-news-subwords-300")

from numpy import False_
from transformers import BertTokenizer, BertForMaskedLM, BertForNextSentencePrediction
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)
sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')
nsp_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

SEARCH_LIMIT = 30522
result_list = ["with", "without", "only"]

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_word_list = set(stopwords.words('english'))

from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class Stats:
  def __init__(self):
    self.data = {
      "correct": 0,
      "incorrect": 0,
      #"index_sum": 0,
      #"not_found": 0,
      "similarity": 0,
      #"categories": [0, 0, 0, 0, 0, 0, 0],
      "sentence_similarity": 0,
      #"similarities": [],
      "incorrect_similarity": 0,
      "nsp_score": 0
    }

  def add_item(self, res):
    if res["result"] == 'CORRECT':
      self.data["correct"] += 1
    else:
      self.data["incorrect"] += 1
      if res["similarity"] != "Not Found":
        self.data["incorrect_similarity"] += res["similarity"]
    #if res["index"] == SEARCH_LIMIT:
      #self.data["not_found"] += 1
    #self.data["index_sum"] += res["index"]
    if res["similarity"] != "Not Found":
      self.data["similarity"] += res["similarity"]
      #self.data["similarities"].append(res["similarity"])
    #self.data["categories"][res["category"] - 1] += 1
  
  def add_obj(self, res):
    self.data["correct"] += res["correct"]
    self.data["incorrect"] += res["incorrect"]
    #self.data["index_sum"] += res["index_sum"]
    #self.data["not_found"] += res["not_found"]
    self.data["similarity"] += res["similarity"]
    self.data["incorrect_similarity"] += res["incorrect_similarity"]
    #self.data["similarities"] += res["similarities"]
    if res["sentence_similarity"] != 0:
      self.data["sentence_similarity"] += res["sentence_similarity"]
    if res["nsp_score"] != 0:
      self.data["nsp_score"] += res["nsp_score"]
    #for i in range(len(res["categories"])):
      #self.data["categories"][i] += res["categories"][i]

  def get_data(self):
    return self.data
  
  def get_total(self):
    return self.data["correct"] + self.data["incorrect"]
  
  def get_sentence_similarity(self):
    return self.data["sentence_similarity"]
  
  def add_sentence_similarity(self, sentence_similarity):
    self.data["sentence_similarity"] += sentence_similarity
  
  def get_nsp_score(self):
    return self.data["nsp_score"]
  
  def add_nsp_score(self, nsp_score):
    self.data["nsp_score"] += nsp_score
  
  def print_data(self):
    total = self.get_total()
    print(f"\nCorrect Predictions   = {self.data['correct']} {get_percent(self.data['correct'], total)}")
    print(f"Incorrect Predictions = {self.data['incorrect']} {get_percent(self.data['incorrect'], total)}")
    print(f"Total Predictions     = {total}")
    #print(f"\nAverage Index         = {round(self.data['index_sum'] / total, 1)}")
    #print(f"Indexes Not Found     = {self.data['not_found']} {get_percent(self.data['not_found'], total)}")
    print(f"Average Similarity    = {round(self.data['similarity'] / total, 2)}")
    if (self.data['incorrect'] != 0):
      print(f"Incorrect Similarity  = {round(self.data['incorrect_similarity'] / self.data['incorrect'], 2)}\n")

    #plt.hist(self.data["similarities"], 10, (0, 100), color = 'green', histtype = 'bar', rwidth = 0.8)
    #plt.xlabel('Similarity Scores')
    #plt.ylabel('Number of Appearances')
    #plt.title('Similarity Score Distribution')
    #plt.show()

    #print("\nPredictions by Index Category:\n")
    #categories_txt = ["0", "1", "2-9", "10-99", "100-999", "1000-4999", "5000-Not Found"]
    #categories_sum = 0
    #for i in range(len(self.data['categories'])):
      #categories_sum += self.data['categories'][i] * (i + 1)
      #txt = f"#{i + 1} ({categories_txt[i]}) "
      #print(f"{pad_word(txt, 22)}= {self.data['categories'][i]} {get_percent(self.data['categories'][i], total)}")
    #div = categories_sum / total
    #print(f"\nAverage Category      = {round(div, 1)} (~{categories_txt[round(div) - 1]})")

def pad_word(input_str, length):
  for x in range(length - len(input_str)):
    input_str += ' '
  return input_str

def print_word(
    masked_word="Masked Word",
    predicted_word="Predicted Word",
    prediction_result="Prediction Result",
    #correct_index="Index of Correct Word",
    similarity="Similarity",
    top_predictions="Next Three Predictions",
    #prediction_category="Category",
    stop_word="Stop Word"
):
  print(f"| {pad_word(masked_word, 16)} ", end = '')
  print(f"| {pad_word(predicted_word, 16)} ", end = '')
  print(f"| {pad_word(prediction_result, 17)} ", end = '')
  #print(f"| {pad_word(correct_index, 21)} ", end = '')
  print(f"| {pad_word(similarity, 10)} ", end = '')
  print(f"| {pad_word(top_predictions, 36)} ", end = '')
  #print(f"| {pad_word(prediction_category, 8)} ", end = '')
  print(f"| {pad_word(stop_word, 9)} |")

def print_sep():
  print("--------------------------------------------------------------------------------------------------------------------------------------------------------------")

def pred_word(txt, correct_word):
  input = tokenizer.encode_plus(txt, return_tensors = "pt")
  mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
  output = model(**input)

  logits = output.logits
  softmax = F.softmax(logits, dim = -1)
  try:
    mask_word = softmax[0, mask_index, :]
  except:
    print("error occurred with line 163")
    return {
      "result": "INCORRECT",
      "similarity": 0,
      "pred_word": "UNKNOWN"
    }
  top = torch.topk(mask_word, SEARCH_LIMIT, dim = 1)[1][0]
  tokens = []
  for token in top:
    word = tokenizer.decode([token])
    if re.search(r'^\W+$', word) == None:
      tokens.append(word)
      break
  if tokens[0] == correct_word:
    result = "CORRECT"
  else:
    result = "INCORRECT"
  #try:
    #index = tokens.index(correct_word)
  #except:
    #index = "Not Found"
  #display_index = f"{index}"
  #if index == "Not Found":
    #index = SEARCH_LIMIT
  try:
    similarity = round(100 * float(vec_model.similarity(correct_word, tokens[0])), 2)
  except:
    similarity = "Not Found"
  #if index == 0:
    #category = 1
  #elif index == 1:
    #category = 2
  #elif index < 10:
    #category = 3
  #elif index < 100:
    #category = 4
  #elif index < 1000:
    #category = 5
  #elif index < 5000:
    #category = 6
  #else:
    #category = 7
  if correct_word in stop_word_list:
    is_stop = "TRUE"
  else:
    is_stop = "FALSE"
  #print_word(
    #masked_word=correct_word,
    #predicted_word=tokens[0],
    #prediction_result=result,
    #correct_index=display_index,
    #similarity=f"{similarity}",
    #top_predictions=', '.join(tokens[1:4]),
    #prediction_category=f"{category}",
    #stop_word = is_stop
  #)
  return {
      "result": result,
      #"index": index,
      "similarity": similarity,
      #"category": category,
      "pred_word": tokens[0]
  }

def get_predictions(text, ignore_proper=False):
  #print(f"\nBeginning predictions on new sentence... <<<{text}>>>")
  #print_sep()
  #print_word()
  #print_sep()
  sentences = [text, ""]
  spl = text.split(" ")
  index = 0
  stats = {}
  for word in result_list:
    stats[f"{word}_stop"] = Stats()
  
  proper_nouns = []
  if ignore_proper:
    words = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(words)
    for (word, tag) in tagged:
      if tag == 'NNP':
        proper_nouns.append(word.lower())

  for word in spl:
    word = re.search(r"[\w\-']+", word)
    if word != None and (ignore_proper == False or word.group(0).lower() not in proper_nouns):
      word = word.group(0)
      sentence = ""
      for i in range(len(spl)):
        if i == index:
          sentence += f"{spl[i].replace(word, '[MASK]')} "
        else:
          sentence += f"{spl[i]} "
      res = pred_word(sentence.strip(), word.lower())
      sentences[1] += res["pred_word"] + " "
      stats["with_stop"].add_item(res)
      if word.lower() not in stop_word_list:
        stats["without_stop"].add_item(res)
      else:
        stats["only_stop"].add_item(res)
    index += 1
  
  sentence_embeddings = sentence_model.encode(sentences)
  sentence_similarity = round(100 * float(cosine_similarity(
    [sentence_embeddings[0]],
    [sentence_embeddings[1]]
  )[0][0]), 2)
  stats["with_stop"].add_sentence_similarity(sentence_similarity)
  #print_sep()
  #print(f"Sentence similarity score: {sentence_similarity}")

  return stats

def get_percent(part, whole):
  if part == 0:
    return ""
  else:
    return f"({round((part / whole) * 100, 1)}%)"

def run_predictor(input_txt, use_tokenizer=False, sentence_format=False, ignore_proper=False, data=False):
  stats = {}
  for word in result_list:
    stats[f"{word}_stop"] = Stats()

  if use_tokenizer:
    sentences = nltk.sent_tokenize(input_txt)
  elif sentence_format:
    sentences = nltk.sent_tokenize(input_txt)
    for i in range(len(sentences)):
      newsent = re.sub(r"\n+\s*([A-Z])", lambda m: " " + m.group(1).lower(), sentences[i]).strip()
      newsent = re.sub(r"\s{2,}", " ", newsent)
      sentences[i] = newsent
  else:
    sentences = re.split(r'\n+', input_txt.strip())

  sentence_counter = 0
  #score_counter = 0
  for sentence in sentences:
    if len(sentence.strip()) == 0:
      continue
    sentence_counter += 1
    res = get_predictions(sentence.strip(), ignore_proper)
    for word in result_list:
      stats[f"{word}_stop"].add_obj(res[f"{word}_stop"].get_data())
    if len(sentences) > 1 and sentence_counter < (len(sentences)):
      encoding = tokenizer.encode_plus(sentences[sentence_counter - 1], sentences[sentence_counter], return_tensors='pt')
      outputs = nsp_model(**encoding)[0]
      softmax = F.softmax(outputs, dim = 1)
      score = round(float(softmax[0][0].item()) * 100, 2)
      #if (score > 10 and score < 90):
        #score_counter += 1
        #print(f"Score between 10 and 90 (#{score_counter}): {score}")
        #print(f"Sentence pair: {sentences[sentence_counter - 1]} {sentences[sentence_counter]}")
      #print(f"Next sentence prediction score: {score}")
      stats["with_stop"].add_nsp_score(score)
  
  total_obj = {}
  for word in result_list:
    stats_obj = stats[f"{word}_stop"]
    total_obj[f"{word}_stop"] = stats_obj.get_data()
    partial_total = stats_obj.get_total()
    full_total = stats["with_stop"].get_total()
    print(f"\nResults for {word.upper()} stop words...")
    print(f"{partial_total}/{full_total} {get_percent(partial_total, full_total)}")
    stats_obj.print_data()
    print()
    print_sep()
  
  print(f"\nResults for sentences...\n")
  print(f"Number of Sentences   = {sentence_counter}")
  print(f"Average Similarity    = {round(stats['with_stop'].get_sentence_similarity() / sentence_counter, 2)}")
  if sentence_counter > 1:
    print(f"Average NSP Score     = {round(stats['with_stop'].get_nsp_score() / (sentence_counter - 1), 2)}")
  print()
  print_sep()
  
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

def get_book(selection):
  print(f"Starting to get predictions for {selection['title']}")
  
  r = requests.get(selection["url"])
  book = r.text

  # remove unwanted new line and tab characters from the text
  for char in ["\r", "\d", "\t"]:
    book = book.replace(char, " ")

  start_pos = re.search(r'\*\*\* START OF TH(E|IS) PROJECT GUTENBERG EBOOK .+? \*\*\*', book)
  if start_pos == None:
    print(f"ERROR: no start position found for {selection['title']}")
    return
  start_pos = start_pos.end()

  end_pos = re.search(r'\*\*\* END OF TH(E|IS) PROJECT GUTENBERG EBOOK .+? \*\*\*', book)
  if end_pos == None:
    print(f"ERROR: no end position found for {selection['title']}")
    return
  end_pos = end_pos.start()
    
  book = book[start_pos:end_pos]
  
  run_predictor(book, use_tokenizer=True, data=selection)
  
with open('books.txt') as f:
  books = json.load(f)
  book_list = list(books.keys())
  if len(sys.argv) > 1:
    arg1 = sys.argv[1]
    if arg1.isdigit():
      arg1 = int(arg1)
      if arg1 >= len(book_list):
        print(f"ERROR: {arg1} is greater than length of book list ({len(book_list)})")
      else:
        book = books[book_list[arg1]]
        get_book(book)
    else:
      arg1 = arg1.lower()
      if arg1 in book_list:
        book = books[arg1]
        get_book(book)
      else:
        print(f"ERROR: {arg1} is not in book list")
  else:
    print(f"ERROR: no book specified")

end = time.time()
seconds = end - start
minutes = math.floor(seconds / 60)
if minutes == 0:
  print(f"Program took {math.floor(seconds)} seconds to run.")
else:
  hours = math.floor(minutes / 60)
  if hours == 0:
    print(f"Program took {minutes} minutes and {math.floor(seconds) % 60} seconds to run.")
  else:
    print(f"Program took {hours} hours and {minutes % 60} minutes to run.")

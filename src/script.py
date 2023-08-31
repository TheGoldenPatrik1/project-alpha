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
from transformers import BertTokenizer, BertForMaskedLM, BertForNextSentencePrediction, pipeline
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)
sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')
nsp_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
generator = pipeline('text-generation', model="facebook/opt-2.7b")

SEARCH_LIMIT = 30522
result_list = ["with", "without", "only"]

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_word_list = set(stopwords.words('english'))

from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def arg_parse():
  args = {
    "use_tokenizer": False,
    "sentence_format": False,
    "ignore_proper": False,
    "nsp_only": False,
    "args": [],
    "book": False,
    "essay": False,
    "poem": False
  }
  if len(sys.argv) == 1: return args
  sys.argv.pop(0)
  for arg in sys.argv:
    arg = arg.lower()
    if arg.startswith("-"):
      if "token" in arg: args["use_tokenizer"] = True
      elif "sent" in arg: args["sentence_format"] = True
      elif "ignore" in arg or "proper" in arg: args["ignore_proper"] = True
      elif "nsp" in arg: args["nsp_only"] = True
      elif "book" in arg: args["book"] = True
      elif "essay" in arg: args["essay"] = True
      elif "poem" in arg: args["poem"] = True
    else:
      args["args"].append(arg)
args = arg_parse()
arguments = args["args"]

class Stats:
  def __init__(self):
    self.data = {
      "mask_correct": 0,
      "mask_incorrect": 0,
      #"index_sum": 0,
      #"not_found": 0,
      "mask_similarity": 0,
      #"categories": [0, 0, 0, 0, 0, 0, 0],
      "generate_correct": 0,
      "generate_incorrect": 0,
      "generate_similarity": 0,
      #"similarities": [],
      "mask_incorrect_similarity": 0,
      "generate_incorrect_similarity": 0,
      "nsp_score": 0
    }

  def add_item(self, res):
    if res["mask_result"] == 'CORRECT':
      self.data["mask_correct"] += 1
    else:
      self.data["mask_incorrect"] += 1
      if res["mask_similarity"] != "Not Found":
        self.data["mask_incorrect_similarity"] += res["mask_similarity"]
    if res["generate_result"] == 'CORRECT':
      self.data["generate_correct"] += 1
    else:
      self.data["generate_incorrect"] += 1
      if res["generate_similarity"] != "Not Found":
        self.data["generate_incorrect_similarity"] += res["generate_similarity"]
    #if res["index"] == SEARCH_LIMIT:
      #self.data["not_found"] += 1
    #self.data["index_sum"] += res["index"]
    if res["mask_similarity"] != "Not Found":
      self.data["mask_similarity"] += res["mask_similarity"]
      #self.data["similarities"].append(res["similarity"])
    if res["generate_similarity"] != "Not Found":
      self.data["generate_similarity"] += res["generate_similarity"]
    #self.data["categories"][res["category"] - 1] += 1
  
  def add_obj(self, res):
    self.data["mask_correct"] += res["mask_correct"]
    self.data["mask_incorrect"] += res["mask_incorrect"]
    #self.data["index_sum"] += res["index_sum"]
    #self.data["not_found"] += res["not_found"]
    self.data["mask_similarity"] += res["mask_similarity"]
    self.data["mask_incorrect_similarity"] += res["mask_incorrect_similarity"]
    #self.data["similarities"] += res["similarities"]
    self.data["generate_correct"] += res["generate_correct"]
    self.data["generate_incorrect"] += res["generate_incorrect"]
    self.data["generate_similarity"] += res["generate_similarity"]
    self.data["generate_incorrect_similarity"] += res["generate_incorrect_similarity"]
    if res["sentence_similarity"] != 0:
      self.data["sentence_similarity"] += res["sentence_similarity"]
    if res["nsp_score"] != 0:
      self.data["nsp_score"] += res["nsp_score"]
    #for i in range(len(res["categories"])):
      #self.data["categories"][i] += res["categories"][i]

  def get_data(self):
    return self.data
  
  def get_total(self, type):
    return self.data[f"{type}_correct"] + self.data[f"{type}_incorrect"]
  
  def get_sentence_similarity(self):
    return self.data["sentence_similarity"]
  
  def add_sentence_similarity(self, sentence_similarity):
    self.data["sentence_similarity"] += sentence_similarity
  
  def get_nsp_score(self):
    return self.data["nsp_score"]
  
  def add_nsp_score(self, nsp_score):
    self.data["nsp_score"] += nsp_score
  
  def print_data(self):
    total = self.get_total("mask")
    print("Mask Word Results:")
    print(f"\nCorrect Predictions   = {self.data['mask_correct']} {get_percent(self.data['mask_correct'], total)}")
    print(f"Incorrect Predictions = {self.data['mask_incorrect']} {get_percent(self.data['mask_incorrect'], total)}")
    print(f"Total Predictions     = {total}")
    #print(f"\nAverage Index         = {round(self.data['index_sum'] / total, 1)}")
    #print(f"Indexes Not Found     = {self.data['not_found']} {get_percent(self.data['not_found'], total)}")
    print(f"Average Similarity    = {round(self.data['mask_similarity'] / total, 2)}")
    if (self.data['mask_incorrect'] != 0):
      print(f"Incorrect Similarity  = {round(self.data['mask_incorrect_similarity'] / self.data['mask_incorrect'], 2)}\n")

    total = self.get_total("generate")
    print("Generative Word Results:")
    print(f"\nCorrect Predictions   = {self.data['generate_correct']} {get_percent(self.data['generate_correct'], total)}")
    print(f"Incorrect Predictions = {self.data['generate_incorrect']} {get_percent(self.data['generate_incorrect'], total)}")
    print(f"Total Predictions     = {total}")
    #print(f"\nAverage Index         = {round(self.data['index_sum'] / total, 1)}")
    #print(f"Indexes Not Found     = {self.data['not_found']} {get_percent(self.data['not_found'], total)}")
    print(f"Average Similarity    = {round(self.data['generate_similarity'] / total, 2)}")
    if (self.data['generate_incorrect'] != 0):
      print(f"Incorrect Similarity  = {round(self.data['generate_incorrect_similarity'] / self.data['generate_incorrect'], 2)}\n")

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
    mask_predicted_word="Mask Predicted Word",
    mask_prediction_result="Mask Prediction Result",
    #correct_index="Index of Correct Word",
    mask_similarity="Mask Similarity",
    top_predictions="Next Three Predictions",
    #prediction_category="Category",
    generate_predicted_word="Generative Predicted Word",
    generate_predicted_result="Generative Prediction Result",
    generate_similarity="Generative Similarity",
    stop_word="Stop Word"
):
  print(f"| {pad_word(masked_word, 16)} ", end = '')
  print(f"| {pad_word(mask_predicted_word, 21)} ", end = '')
  print(f"| {pad_word(mask_prediction_result, 22)} ", end = '')
  #print(f"| {pad_word(correct_index, 21)} ", end = '')
  print(f"| {pad_word(mask_similarity, 15)} ", end = '')
  print(f"| {pad_word(top_predictions, 36)} ", end = '')
  #print(f"| {pad_word(prediction_category, 8)} ", end = '')
  print(f"| {pad_word(generate_predicted_word, 23)} ", end='')
  print(f"| {pad_word(generate_predicted_result, 25)} ", end='')
  print(f"| {pad_word(generate_similarity, 19)} ", end='')
  print(f"| {pad_word(stop_word, 9)} |")

def print_sep():
  print("--------------------------------------------------------------------------------------------------------------------------------------------------------------")

def pred_word(txt, correct_word, generate_input):
  input = tokenizer.encode_plus(txt, return_tensors = "pt")
  if input['input_ids'].size(dim=1) > 512:
    print("error with giant sentence")
    return {
      "mask_result": "INCORRECT",
      "mask_similarity": 0,
      "mask_pred_word": "UNKNOWN",
      "generate_result": "INCORRECT",
      "generate_similarity": 0,
      "generate_pred_word": "UNKNOWN"
    }
  mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
  mask_output = model(**input)

  generate_text = "N/A"
  generate_result = "UNKNOWN"
  generate_similarity = -1
  if generate_input != None:
    generate_output = generator(generate_input)
    generate_text = generate_output['generate_text']
    # TODO: trim to one word
    if generate_text == correct_word:
      generate_result = "CORRECT"
    else:
      generate_result = "INCORRECT"
    try:
      generate_similarity = round(100 * float(vec_model.similarity(correct_word, generate_text)), 2)
    except:
      generate_similarity = "Not Found"

  logits = mask_output.logits
  softmax = F.softmax(logits, dim = -1)
  try:
    mask_word = softmax[0, mask_index, :]
  except:
    print("error occurred with line 163")
    return {
      "mask_result": "INCORRECT",
      "mask_similarity": 0,
      "mask_pred_word": "UNKNOWN",
      "generate_result": "INCORRECT",
      "generate_similarity": 0,
      "generate_pred_word": "UNKNOWN"
    }
  top = torch.topk(mask_word, SEARCH_LIMIT, dim = 1)[1][0]
  tokens = []
  for token in top:
    word = tokenizer.decode([token])
    if re.search(r'^\W+$', word) == None:
      tokens.append(word)
      break
  if tokens[0] == correct_word:
    mask_result = "CORRECT"
  else:
    mask_result = "INCORRECT"
  #try:
    #index = tokens.index(correct_word)
  #except:
    #index = "Not Found"
  #display_index = f"{index}"
  #if index == "Not Found":
    #index = SEARCH_LIMIT
  try:
    mask_similarity = round(100 * float(vec_model.similarity(correct_word, tokens[0])), 2)
  except:
    mask_similarity = "Not Found"
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
  print_word(
    masked_word=correct_word,
    mask_predicted_word=tokens[0],
    mask_prediction_result=result,
    correct_index=display_index,
    mask_similarity=f"{similarity}",
    top_predictions=', '.join(tokens[1:4]),
    prediction_category=f"{category}",
    generate_predicted_word=generate_text,
    generate_prediction_result=generate_result,
    generate_similarity=generate_similarity,
    stop_word = is_stop
  )
  return {
      "mask_result": result,
      #"index": index,
      "mask_similarity": similarity,
      #"category": category,
      "mask_pred_word": tokens[0],
      "generate_result": generate_result,
      "generate_similarity": generate_similarity,
      "generate_pred_word": generate_text
  }

def get_predictions(text, ignore_proper=False):
  print(f"\nBeginning predictions on new sentence... <<<{text}>>>")
  print_sep()
  print_word()
  print_sep()
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
      input = ""
      for i in range(len(spl)):
        if i < index:
          input += f"{spl[i]} "
        if i == index:
          sentence += f"{spl[i].replace(word, '[MASK]')} "
        else:
          sentence += f"{spl[i]} "
      res = pred_word(sentence.strip(), word.lower(), None if index == 0 else input.strip())
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
  print_sep()
  print(f"Sentence similarity score: {sentence_similarity}")

  return stats

def get_percent(part, whole):
  if part == 0:
    return ""
  else:
    return f"({round((part / whole) * 100, 1)}%)"

def get_nsp(sentences):
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
      encoding = tokenizer.encode_plus(sentence, next_sentence, return_tensors='pt')
      if encoding['input_ids'].size(dim=1) > 512:
        continue
      outputs = nsp_model(**encoding)[0]
      softmax = F.softmax(outputs, dim = 1)
      total_score += round(float(softmax[0][0].item()) * 100, 2)
      total_count += 1
  print(f"Generated {total_count} sentence pairs from {len(sentences)} sentences")
  print(f"Compare with n(n-1): {len(sentences)*(len(sentences)-1)}")
  print(f"Average score: {round(total_score / total_count, 2)}")
  
def run_predictor(input_txt, data=False):
  stats = {}
  for word in result_list:
    stats[f"{word}_stop"] = Stats()

  if args["use_tokenizer"]:
    sentences = nltk.sent_tokenize(input_txt)
  elif args["sentence_format"]:
    sentences = nltk.sent_tokenize(input_txt)
    for i in range(len(sentences)):
      newsent = re.sub(r"\n+\s*([A-Z])", lambda m: " " + m.group(1).lower(), sentences[i]).strip()
      newsent = re.sub(r"\s{2,}", " ", newsent)
      sentences[i] = newsent
  else:
    sentences = re.split(r'\n+', input_txt.strip())

  if args["nsp_only"]:
    return get_nsp(sentences)
    
  sentence_counter = 0
  #score_counter = 0
  for sentence in sentences:
    if len(sentence.strip()) == 0:
      continue
    sentence_counter += 1
    res = get_predictions(sentence.strip(), args["ignore_proper"])
    for word in result_list:
      stats[f"{word}_stop"].add_obj(res[f"{word}_stop"].get_data())
    if len(sentences) > 1 and sentence_counter < (len(sentences)):
      encoding = tokenizer.encode_plus(sentences[sentence_counter - 1], sentences[sentence_counter], return_tensors='pt')
      if encoding['input_ids'].size(dim=1) > 512:
        continue
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

  start_pos = re.search(r'\*\*\*\s*START OF TH(E|IS) PROJECT GUTENBERG EBOOK .+?\*\*\*', book, re.M | re.S)
  if start_pos == None:
    print(f"ERROR: no start position found for {selection['title']}")
    return
  start_pos = start_pos.end()

  end_pos = re.search(r'\*\*\*\s*END OF TH(E|IS) PROJECT GUTENBERG EBOOK .+?\*\*\*', book, re.M | re.S)
  if end_pos == None:
    print(f"ERROR: no end position found for {selection['title']}")
    return
  end_pos = end_pos.start()
    
  book = book[start_pos:end_pos]
  
  run_predictor(book, data=selection)

def run_books():
  with open('./files/books.txt') as f:
    books = json.load(f)
    book_list = list(books.keys())
    if len(arguments) > 0:
      arg1 = arguments[0]
      if arg1.isdigit():
        arg1 = int(arg1)
        if arg1 >= len(book_list):
          print(f"ERROR: {arg1} is greater than length of book list ({len(book_list)})")
        else:
          book = books[book_list[arg1]]
          get_book(book)
      else:
        if arg1 in book_list:
          book = books[arg1]
          get_book(book)
        else:
          print(f"ERROR: {arg1} is not in book list")
    else:
      print(f"ERROR: no book specified")
    
def run_texts(content_type):
  with open('./files/texts.txt') as f:
    content = json.load(f)
    texts = content[content_type]
    if len(arguments) > 0:
      arg1 = arguments[0]
      if arg1.isdigit():
        arg1 = int(arg1)
        text_list = list(texts.keys())
        length = len(text_list)
        if arg1 >= length:
          print(f"ERROR: {arg1} is greater than length of text list ({length})")
        else:
          text = texts[text_list[arg1]]
          print(f"running predictor on text: {text_list[arg1]}")
          run_predictor(text)
      elif arg1 == 'all':
        text_list = list(texts.keys())
        for txt in text_list:
          print(f"running predictor on text: {txt}")
          run_predictor(texts[txt])
          print()
      else:
        print(f"ERROR: {arg1} is not a valid text")
    else:
      print(f"ERROR: no text specified")

if args["book"]: run_books()
elif args["essay"]: run_texts("essays")
elif args["poem"]: run_texts("poems")
else: print(f"ERROR: no content specified to run program on") 
    
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

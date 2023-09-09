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

from stats import Stats
import formatters

class Predictor:
  def __init__(self, args):
    self.args = args
    self.arguments = self.args['args']

    self.vec_model = vec_api.load("fasttext-wiki-news-subwords-300")
    self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    self.model = BertForMaskedLM.from_pretrained('bert-large-uncased', return_dict=True)
    self.sentence_model = SentenceTransformer('bert-large-nli-mean-tokens')
    self.nsp_model = BertForNextSentencePrediction.from_pretrained('bert-large-uncased')
    self.generator = pipeline('text-generation', model="facebook/opt-1.3b")

    self.stop_word_list = set(stopwords.words('english'))

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

  def pred_word(self, txt, correct_word, generate_input):
    tokenized = self.tokenizer.encode_plus(txt, return_tensors = "pt")
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
    mask_index = torch.where(tokenized["input_ids"][0] == self.tokenizer.mask_token_id)
    mask_output = self.model(**tokenized)

    generate_text = "N/A"
    generate_result = "UNKNOWN"
    generate_similarity = "Not Found"
    if generate_input != None:
      generate_length = (len(generate_input.split()) * 2) + 5
      generate_output = self.generator(generate_input, max_length=generate_length)
      generate_output = generate_output[0]['generated_text'].strip().split(generate_input)
      if len(generate_output) > 1:
        generate_output = generate_output[1].strip().split()
        generate_output = generate_output[0].lower() if len(generate_output) > 0 else ""
        generate_output = re.search(r"[\w\-']+", generate_output)
        if generate_output != None:
          generate_text = generate_output.group(0)
          generate_result = "CORRECT" if generate_text == correct_word else "INCORRECT"
          try:
            generate_similarity = round(100 * float(self.vec_model.similarity(correct_word, generate_text)), 2)
          except:
            generate_similarity = "Not Found"

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
      word = self.tokenizer.decode([token])
      if re.search(r'^\W+$', word) == None:
        tokens.append(word)
        if len(tokens) == 10:
          break
    mask_result = "CORRECT" if tokens[0] == correct_word else "INCORRECT"
    is_top_10 = correct_word in tokens
    #try:
      #index = tokens.index(correct_word)
    #except:
      #index = "Not Found"
    #display_index = f"{index}"
    #if index == "Not Found":
      #index = SEARCH_LIMIT
    try:
      mask_similarity = round(100 * float(self.vec_model.similarity(correct_word, tokens[0])), 2)
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
    is_stop = "TRUE" if correct_word in self.stop_word_list else "FALSE"
    if self.args["logs"] == True:
      formatters.print_word(
        masked_word=correct_word,
        mask_predicted_word=tokens[0],
        mask_prediction_result=mask_result,
        #correct_index=display_index,
        mask_similarity=mask_similarity,
        #top_predictions=', '.join(tokens[1:4]),
        #prediction_category=category,
        generate_predicted_word=generate_text,
        generate_prediction_result=generate_result,
        generate_similarity=generate_similarity,
        stop_word = is_stop
      )
    return {
        "mask_result": mask_result,
        #"index": index,
        "mask_similarity": mask_similarity,
        #"category": category,
        "mask_pred_word": tokens[0],
        "generate_result": generate_result,
        "generate_similarity": generate_similarity,
        "generate_pred_word": generate_text,
        "is_top_10": is_top_10
    }

  def get_predictions(self, text, ignore_proper=False):
    if self.args["logs"] == True:
      print(f"\nBeginning predictions on new sentence... <<<{text}>>>")
      formatters.print_sep()
      formatters.print_word()
      formatters.print_sep()
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
        generate_input = ""
        for i in range(len(spl)):
          if i < index:
            generate_input += f"{spl[i]} "
          if i == index:
            sentence += f"{spl[i].replace(word, '[MASK]')} "
          else:
            sentence += f"{spl[i]} "
        res = self.pred_word(sentence.strip(), word.lower(), None if index == 0 else generate_input.strip())
        sentences[1] += res["mask_pred_word"] + " "
        stats["with_stop"].add_item(res)
        if word.lower() not in self.stop_word_list:
          stats["without_stop"].add_item(res)
        else:
          stats["only_stop"].add_item(res)
      index += 1
    
    sentence_embeddings = self.sentence_model.encode(sentences)
    sentence_similarity = round(100 * float(cosine_similarity(
      [sentence_embeddings[0]],
      [sentence_embeddings[1]]
    )[0][0]), 2)
    stats["with_stop"].add_sentence_similarity(sentence_similarity)
    if self.args["logs"] == True:
      formatters.print_sep()
      print(f"Sentence similarity score: {sentence_similarity}")

    return stats

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

    if self.args["nsp_only"]:
      return self.get_nsp(sentences)
      
    sentence_counter = 0
    #score_counter = 0
    for sentence in sentences:
      if len(sentence.strip()) == 0:
        continue
      sentence_counter += 1
      res = self.get_predictions(sentence.strip(), self.args["ignore_proper"])
      for word in result_list:
        stats[f"{word}_stop"].add_obj(res[f"{word}_stop"].get_data())
      if len(sentences) > 1 and sentence_counter < (len(sentences)):
        encoding = self.tokenizer.encode_plus(sentences[sentence_counter - 1], sentences[sentence_counter], return_tensors='pt')
        if encoding['input_ids'].size(dim=1) > 512:
          continue
        outputs = self.nsp_model(**encoding)[0]
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
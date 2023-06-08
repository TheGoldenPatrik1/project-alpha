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
  if input['input_ids'].size(dim=1) > 512:
    print("error with giant sentence")
    return {
      "result": "INCORRECT",
      "similarity": 0,
      "pred_word": "UNKNOWN"
    }
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
  
  run_predictor(book, use_tokenizer=True, data=selection)
  
#with open('books.txt') as f:
if False:
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

essays = {
  "paul": """Over the past twenty years, drone usage by the United States military has grown exponentially, mostly due to their role in the fight against terrorism. Airstrikes and drone reconnaissance have reached new highs in the past decade, with rates increasing from eight drone strikes per year in 2007 to 2,776 strikes in 2017 . The nature of drone technology blurs the line between military and non-military counterterrorism, which is only natural since the catalyst itself, terrorism, combines both military and non-military  contexts. As a result, drones raise new questions in ethics and morality, specifically regarding the overarching effects of distorting the lines between military and social contexts and whether drone use by the military is ultimately beneficial or harmful. Therefore, the debate has formed between those who support the traditional use of the military in the fight against terrorism, and those who believe the military must adapt its policies as situations change. More specifically, while both sides agree that there are advantages and drawbacks to the use of drones to fight terrorism, contention ensues in determining whether these advantages offset the disadvantages. In general, scholars approach this question in regard to either the social drawbacks or the military advantages of drone strikes. In contrast, I suggest that the most complete solution to the use of drones by the military considers both the advantages of drones in relation to traditional weapons and the external social effect of drone strikes in foreign countries.
First, I will discuss the differences between traditional war and the current war on terrorism, and subsequently explain how drones were designed to provide a solution to the new military complexities raised by terrorism. The line between formal war and murderous crime was blurred by acts of terror, most significantly the Twin Tower attacks. There are several ways in which terrorism deviated from the traditional definition of war. For one, it brought war outside of traditionally designated combat territory and into the homeland. Secondly, terrorism aimed to kill innocent people for the sake of it, while in legitimate war innocent people may die only as an unintended consequence. Finally, the terrorists resided in crowded civilian centers in their various home countries, creating unconventional targets for the military to attack. The modern drone was developed in response to the new exigencies imposed by terrorist attacks as a weapon that could be used for military style attacks in non-military theaters. For example, drones are extremely accurate, and thus theoretically can execute pinpoint attacks to eliminate individuals without collateral casualties . Before the drone, the US military would have needed boots on the ground to carry out such a mission. In addition, since drones are unmanned, they have the ability to hover over hostile territory for hours on end. Previously, the military would resort to satellites to perform the same function, which technically do not invade foreign airspace like drones do. Drones are also capable of striking anywhere on the planet, no matter the time of day or range of attack. Before drones, we used either strike fighters, which had limited range, or cruise missiles, which were inefficient and politically dangerous. Drones also allow us to survey an area waiting for a target to develop, and then launch a surprise attack. Low numbers of civilian casualties, combined with the low financial cost of drones and the absence of risk to US personnel, make a clear case for why the military so eagerly adopted the drone . However, in creating such a unique weapon, its existence also creates circumstances that the military has never had to consider previously.
In particular, large-scale questions about the social effects and the operation of drones in military and non-military contexts have yet to be considered. First, is it ethical to authorize drone strikes in areas not designated as warzones? Since drones are quicker, less expensive, and less costly to deploy in comparison to actual troops, this has motivated the government to execute airstrikes in territories that are not technically areas of combat. Because this extends the traditional field of battle, the question must be asked- is it ethical to do so? In normal military contexts, the main question is whether the weapons used in battle are capable of consistently hitting targets and if the firepower is in proportion to the enemy’s aggression. However, precisely because drone warfare is typically contemplated in civilian population centers, we must now ask whether social effects should be considered when examining drone strikes. Last, since drones are riskless, efficient, and robust, they reduce the threshold of violence required to initiate retaliation in non-military zones. In other words, do drones incline or encourage governments to attack enemies? As a result, the question arises, is the accepted use of drones in the military a slippery slope that could lead to further immoral combat in the future? Essentially, what are the long-term consequences of expanding drone usage into non-military zones? All the arguments I have read look at this either as a military issue or a social one and fail to combine the two aspects in their analyses. In the remainder of this essay, I will explain why this approach is insufficient towards finding an answer to military drone use. 
Scholars who research this topic tend to focus on one side of the debate, and subsequently prove or disprove the argument to support their stance on the issue. This method is not adequate in regard to the problem of military drone use, because it is such a multifaceted issue. To begin, scholars must address the fact that the primary driver of drone development was completely out of the United States’ control. When making arguments for or against drones, scholars rarely consider the fact that drones are somewhat of a necessary evil. Drone operators attempt to subvert terrorist attacks, which cost thousands of lives each year. Just in 2018, thirty-thousand people were killed by terrorists . Thus, obviously it is important that something is done about this problem, to which the US Military has adopted the drone as a solution. The alternative to drone use in a world where terrorism goes unchecked is not morally feasible. Thus, the argument must be made for how and where drones should be used, rather than if they should be used at all. While most scholars who debate the topic generally argue along these lines, on one side or the other, they largely fail to explicitly state that they are okay with certain drone use, leaving readers confused as to whether they merely want more restrictions on military drones or want to completely abolish drone use. On the opposite side, scholars who make the argument that drones are immoral must explain how the legality of non-military exploits is more significant than the lives of American citizens. They frequently mention the social impacts of such strikes and the overarching social effects that would result from such attacks, but they never mention the countless deaths averted through the military’s use of drones. With scholars falling short of making complete arguments, it will be impossible to ever convey a suitable solution to the problem of modern military drone use. 
Now, I will examine each side of the issue, point out specific flaws, and suggest arguments that should be considered. I will begin with those who critique drone use- in my research, I have found that a slightly higher number of scholars make arguments against drones compared to those who support drones. The first claim made by drone critics is that the overarching social effects of drone strikes are a legitimate concern to the military. They ask the question- is it acceptable to constantly watch over a country with civilians living in a state of perpetual fear? They also argue that the drawbacks far outweigh the positives: the presence of unmanned weaponry over a country greatly rips apart the social fabric, and the killing of low-level operatives in terrorist cells may do more harm than good. However, nowhere in their argument do they actually list the positives, leaving the reader wondering if no positives actually exist, or if the positives so plainly outweigh the negatives that it would be a detriment to their thesis if included. To better prove their argument, it would be beneficial to provide a quantitative comparison of empirical data between the number of terrorists killed by US drones and the number of newly recruited terrorists as a result of each drone strike. Thus, a greater number of new terrorists would significantly strengthen the argument, as it follows res ipsa loquitur, while a higher number of terrorists killed would detract from drone criticism.
The second claim made by drone critics is that drones are in violation of the warrior ethos, a social motivation observed by fighters to safeguard their own ethical status. The ethos comprises a set of virtues, including courage, honor, integrity, and selflessness. Drone critics argue that the combatant’s exposure to harm has historically served as the principal basis for his or her ethical right to kill . While this argument is common one amongst critics, no mention is made of why an ethos exists, and more specifically, why the military must abide by some ancient warrior ethos. The claims made concerning the asymmetric risk posed by drones may be true, but the readers are left wondering why this is dangerous in practice. I suggest that a better argument would take into consideration the various times in history weapons with similar asymmetric risk have been used. In an argument about overpowered weapons, it is fundamentally deficient to omit reference to archers, cavalry, and javelin throwers, which are prominent historical examples of such weapons. Once the uses and origins of these weapons are discussed, it can more easily be understood whether or not drones are breaking traditional ethos in their use. Subsequently, the critic should present various scenarios exhibiting the dangerous effects of using a weapon that falls short of the warrior ethos. In this manner, he or she can make a convincing argument that drones are in violation of the ethos, and thus are harmful in these specific ways. Or, in describing the drawbacks to drones, the reader may be convinced of the opposite- the warrior ethos is mutable, and therefore what was immoral yesterday may be perfectly moral today.
The final argument made against drones is that they are extra-judicial assassinations carried out by the government, since drone strikes take place outside of declared war. While this is certainly a plausible claim, the scholar making this argument barely skims the surface of the underlying differences between assassinations and wartime killings. First, the critic must clearly outline the framework upon which acts may be classified as an assassination and the framework which describes wartime actions. Then, he must discuss the ways in which a drone’s strike mission follows one framework while straying from another. In this way, the scholar makes an apparent case for drones being assassination machines, and the reader is made aware of the intricate details which may classify a drone strike as an assassination rather than an act of wartime. Additionally, keeping in mind that an assassination is usually defined as the killing of a political figure, the scholar is required to define the actions of a terrorist which deem them a political rather than wartime figure. Finally, in keeping with the overarching standards I believe are required of drone arguments, the scholar must go on to describe why and exactly how the drawbacks of assassinations outweigh the positive benefits of drone use against terrorists.
However, as mentioned previously, I would be no better off than any other scholar if I ended my argument there, citing the previous examples as enough material to reach a conclusion. Instead, I will now look at the opposing side of the debate and identify specific faults and suggest arguments that should be considered. The first claim made by drone proponents is that drones are far superior to traditional military weapons in terms of human cost, military robustness, and efficiency. However, scholars who take this stance tend to avoid the follow-up question to such a statement- in making it significantly easier to launch attacks on foreign adversaries, does it not drastically reduce the threshold at which the government initiates an attack? Any reader informed on the topic of drone use would expect some sort of explanation for how governments should balance newly developed technology with the asymmetric risk that accompanies it. To make a convincing argument, the scholar should make a graph or table showing the number of military actions taken before the drone and after the drone. Once the data are examined, the scholar may even go a step further to justify a greater number of attacks with the efficacy of drones and the vast advantages that they provide.
The second argument made by scholars is that terrorism is an act of war, rather than an act of crime, and this it is only reasonable to respond using military means. While the scholar who makes this claim vaguely describes the differences between war and crime in regard to terrorism, his observations are largely subjective, and thus not suitable for an academic argument. He argues that for a big enough attack, e.g., the attack on the Twin Towers, it would be egregious for the government to consider such a thing mere “violent crime”. From this, he arrives at the “obvious” conclusion that the terrorism-as-war framework should be adopted. While it is quite clear that the September 11 attacks were of enormous scale, it is dangerous to automatically consider such a thing as war. Thus, I suggest that the scholar create a detailed framework for classifying terrorists acts as either crime or warfare. In such a way, those who accept the framework can generally agree on which attacks are actual acts of war, and thus both sides of the argument are considered when reaching a conclusion.
The final argument made by drone proponents is that drones are efficient and effective in the elimination of potential terrorists. While this is the common consensus among American citizens, many reports from reputable sources have asserted that all drone data are wildly inaccurate, stemming from problems such as strikes in under-regulated areas, and the hostility of certain foreign nations towards the press. Thus, if one is to argue that the drone is actually more efficient than the traditional weaponry of the military, he must provide a definitive way in which to track the number of civilians and terrorists killed by drones and then accurately estimate the number of civilians and terrorists that would have been killed using a traditional method, such as boots on the ground. I suggest that the best possible approach utilizes the change in terrorist attacks over time to estimate the number of terrorists killed by drones and uses the average of all civilian death datasets to arrive at an estimate for civilians killed. To estimate the efficiency of traditional weaponry, I suggest that the scholar consider both fiscal cost of past operations and actual SEAL missions, such as the Bin Laden raid, to reach a conclusion. Then, it would be beneficial to include a comparison of the civilian-to-terrorist death ratio for both drones and traditional weaponry for the same time period, preferably the previous twenty years. Thus, with this information at hand, both the scholar and the reader can clearly see why or why not drone strikes are justified.
The majority of scholarly articles on the subject of drone strikes fail to consider both sides, as I have outlined above. Scholars either focus on the positive aspects of drone strikes and give evidence to support it, or they focus on the social backlash from locals and ignore the military benefits of drones. Now, I will suggest a more effective way for scholars to approach the topic of drone use. I propose that scholars begin with an examination of terrorism and the role drones play as a necessary evil. It is important that they clearly denote that the debate is concerning how drones should be used rather than if they should be used. Subsequently, after evaluating the reasons for the existence of drones, I believe that weighing the positive arguments against the negative arguments and producing a framework is the most complete way to arrive at suitable solutions. As I have mentioned previously, arguments that omit one side of the debate are deficient and leave the audience more confused after reading the paper. Furthermore, in my examination of the arguments, I find that certain points are valid while others are uncompelling arguments. Of those made in support of drones, the argument that drones are efficient and effective in the elimination of potential terrorists is supported by the statistics on drone use. Of those made against the use of drones, the argument that the presence of drones over a country greatly rips apart the social fabric is also supported by statistics and remarks which reflect population sentiment. 
In general, the arguments made by scholars regarding military drones are one-sided and incomplete. They frequently attempt to unblur the lines between military and social contexts, hoping that a valid solution can be reached from an examination of either side of the debate, rather than both. However, the thought process behind the solution to this issue is far more complex. Considering that the catalyst of drone development itself is terrorism, it is necessary that the drone operates in a gray area between military and non-military environments, as terrorists can only be eliminated through military-style attacks in non-military zones. Thus, any arguments failing to mention points on both sides of the issue can instantly be dismissed. Still, there remain scholars who make strong arguments for their view and then provide irrelevant opposing points that can easily be refuted- not because their argument is valid, but because the opposing point simply addresses a different issue. These arguments may be easily dismissed as well. The complete answer requires both critics and proponents. While the critic may argue that the social outfall is disastrous, and the proponent may argue that the drone is the most efficient military weapon, a true solution requires a combination and comparison between the two. While the complete solution is far more intricate, the essential ultimatum is this: if the drone’s targets are more dangerous and harmful than the social fallout that will occur as a result, then a drone strike is the best possible answer. If the external social effects are worse than the military gain from using a drone, then the military must use traditional methods.
""",
  "linderman": """Disinformation has become a scourge on public discourse. Especially in the past two years amidst the outbreak of the COVID-19 pandemic, disinformation, purposefully disseminated false information, has exploded on a myriad of topics, namely issues relating to the COVID-19 pandemic and the 2020 Presidential Election, undermining discourse by eroding trust in what ought to be common national consensus. Accordingly, potential solutions to curb and ultimately prevent the spread of online disinformation have become a subject of national importance. However, there is significant disagreement regarding what must be done to prevent its online spread. Researchers have proposed several different approaches, including banning those who spread online disinformation, a restructuring of tech companies to increase oversight and make them more accountable to the public, and better education in digital literacy by the general public. However, due mainly to the volatile and polarized nature of our current political discourse, none of the aforementioned proposed solutions will suffice in stopping disinformation. In this essay I will review and analyze research and proposals regarding increased oversight, banning disinformation sources, and increasing digital literacy, showcasing why all of these solutions are ineffective. Furthermore, I will argue that all policy proposals to stop disinformation are missing the forest for the trees and a restoration of healthy and constructive discourse in our nation is the most important and only effective solution in effectively stopping online disinformation.
According to New York University affiliated researchers Spencer McKay and Chris Tenove, disinformation is defined as “intentionally false or deceptive communication, used to advance the aims of its creators or disseminators at the expense of others”. Disinformation has become widespread in recent years, especially with regards to the ongoing COVID-19 pandemic. This is largely a result of social media companies disrupting “the pre-existing institutions and practices that amplified or filtered out claims in media systems. In particular, they have partially displaced journalists as gatekeepers”. While disinformation and ‘fake news’ have existed since the dawn of the free and open press, the rise and ubiquity of the internet has made it an increasingly severe and pervasive problem, as disinformation can spread far more easily through deliberately interacting and sharing disinformation via social media platforms such as FaceBook and Twitter. Moreover, the rise in disinformation coupled with the lack of sufficient institutions to counter its spread has led to a higher rate of belief in disinformation among the general public.
Given the massive increase in online disinformation in recent years, and the threat it poses to America, there have been several solutions proposed which are intended to curb or prevent its spread. Three of the most common proposals are the restructuring of tech companies to increase oversight from the government and public, banning those who spread disinformation (as well as similar bad actors) from social media companies, and increasing digital media literacy amongst the general public. Restructuring of tech companies is rather broad, but would involve the implementation of various measures by the government that would decrease the spread of disinformation; this solution may be modeled off of practices and policies already in place in various European countries. Banning sources of disinformation is the only solution of these three currently practiced in America on a large scale; this solution involves the permanent removal of
posts or accounts which spread disinformation without any broad or fundamental change in how the social media companies currently operate. Digital media literacy is generally defined as the ability to use the internet in an educated and informed way and discern authoritative information from online disinformation. Rather than focusing on actions taken by big tech companies or the government, solutions involving digital media literacy require changing the behavior of individual internet users.
All of these solutions should theoretically help stop the proliferation of online disinformation, and perhaps all are desirable on their own merit. However, I will argue these solutions will not be effective on their own due to the polarized and volatile nature of America’s current political climate. Rather than restoring healthy, constructive discourse, all of these three solutions will be met with great resistance, only serving to polarize and radicalize the country further. Additionally, solutions which focus on increasing digital media literacy are unlikely to work, as Prof. Tom Buchanan, of the University of Westminster in London, conducted a study which showed that an individual’s level of digital media literacy was not a significant predictor in their likelihood to believe or spread disinformation. Overall, I believe online disinformation is a symptom of a much greater problem: the breakdown in constructive political discourse. In other words, online disinformation has stemmed from a lack of constructive political discourse, and any solution regarding online disinformation will be insufficient unless we restore constructive political discourse in our nation. Restoring such discourse is imperative if any concrete solution regarding curbing online disinformation is to be implemented.
The banning of online disinformation and their sources is the most commonly proposed solution to stopping online disinformation. U.S. Surgeon General Vivek Murthy has called for social media companies to “tweak their algorithms to further demote false information and share
more data with researchers and the government to help teachers, healthcare workers and the media fight misinformation,” in the wake of disinformation related to the COVID-19 pandemic. Similarly, the Director of the World Health Organization has even declared that we are in an “info-demic,” with regards to COVID-19 related online disinformation. Twitter’s new CEO, Parag Agrawal, even said in an interview in November 2020 that Twitter is trying to “focus less on thinking about free speech, but thinking about how the times have changed,” which can reasonably be interpreted as support for banning online disinformation and this solution in general. The most notable example of this strategy is when Twitter and most other major social media companies banned then-President Donald Trump after the January 6th Capitol Riot, as a result of his role in the events of that day.
However, this solution is both massively polarizing and largely ineffective. Returning to the example of President Trump getting banned from social media, this action caused people to polarize and endlessly argue over the issue, rather than unite around this action meant to stop continued online disinformation being spread by the President. According to a poll done by the Pew Research Center, 78% of Republicans opposed the action, while only 11% of Democrats agreed. When Facebook announced that they would also ban Trump indefinitely in May, Republican politicians and pundits once again railed against the decision. Moreover, the ban did very little to stop the spread of online disinformation. Trump was able to spread his claims of the stolen 2020 election, as well as call his supporters to the Capitol in the days leading up to January 6th, to his millions of followers before he was banned. A recent poll shows that the vast majority of Republicans continue to believe that President Joe Biden was not elected legitimately, a disinformation campaign which Donald Trump spread on social media for months and continues to perpetuate.
Simply banning those who spread disinformation is ineffective on its own, even having the opposite effect of further polarizing our political discourse.
A more thorough solution would be the restructuring of tech companies to increase governmental oversight and implement and enforce measures which will curb the spread of disinformation on social media platforms. Voicing his support for a solution along these lines, Facebook CEO Mark Zuckerberg has declared “I don’t think private companies should make so many decisions alone when they touch on fundamental democratic values”. The most thorough research regarding this solution has been done by Professors Emiliana DeBlasio of LUISS University and Donatella Selva of Tuscia University. They look and see what European countries have done to solve the issue of online disinformation and try to determine what methods have been the most effective in countering disinformation. They compare and contrast the laws and practices of eleven European countries to see which are the most effective at stopping online disinformation.
In order to determine which countries handle disinformation best, De Blasio and Selva review and analyze the laws regarding disinformation of each country, using data collected from the official websites of the countries they examine. From there, they broadly divide the laws by the various methods used to combat disinformation: media literacy, multilateral governance, obligations of tech companies, control of fake accounts, rapid detection and removal of disinformation, and more. Finally, they use these metrics to determine which countries are best at combating disinformation. Ultimately, they found that France and Germany have the strongest and most effective laws in place to curb online disinformation. Both countries have laws explicitly prohibiting online disinformation, which include mechanisms to punish tech companies if they fail to effectively enforce the respective laws. As
such, these countries have the most robust protections, leading in companies obligations to transparency and accountability to governments, as well as the rapid detection and removal of sources of disinformation.
However, reforms along French or German lines are highly unlikely to be implemented in a way which would successfully limit the spread of online disinformation. These reforms notably include a legally enforced, much more thorough version of banning people from social media, which was already discussed. As such, while it may be somewhat more effective, it would remain just as divisive as that proposal. In addition, these reforms would be perceived as government overreach into the social media sector and government-sanctioned censorship, which would be incredibly unpopular, given Americans historic distrust of government.
Digital media literacy is a third proposal commonly discussed as a potential solution to online disinformation, stemming from the commonly held belief that people will not fall for online disinformation if they are educated in how to discern real news and authoritative sources from disinformation and fake news. The Southern Poverty Law Center, an anti-extremism and civil rights advocacy organization, and the United States Department of Homeland Security both argue that digital media literacy is necessary for this purpos . Google even recently donated one billion dollars to Taiwan’s media literacy initiative for this expressed purpos . However, this belief is misguided.
In order to examine the root causes of what compels people to spread online disinformation, Tom Buchanan, of the University of Westminster in London, conducted a study consisting of a series of four surveys, two on Facebook, one on Twitter, and one on Instagram, to see what inspired the test subjects to spread disinformation. Authoritativeness and consistency were the same throughout the survey. Various individual differences of the participants, namely personality traits deemed to possibly increase the likelihood of sharing disinformation, conservatism, and digital media literacy were also measured via the the Five-Factor personality questionnaire, the Social and Economic Conservatism Scale, and the New Media Literacy Scale respectively. All three FaceBook posts were originally from InfoWars, a site widely regarded as one of the biggest sources of online disinformation, but any indicators of the source were removed. After seeing each post, the participants were asked first if they thought they would share the article, spreading it to their friends and followers, then if they thought the article was truthful, and finally the likelihood that they had seen the article before.
In all four surveys, a participant’s level of digital media literacy was not found to be a significant factor in a participant’s likelihood of sharing the stories in any of the four surveys conducted. By contrast, the traits of people most likely to spread online disinformation were lower agreeableness, higher extroversion, younger age, men, and if the story aligned with their pre-existing beliefs. . While Buchanan does not propose concrete solutions based on his research, it is clear that solutions based on educating the public in digital media literacy will be ineffective, as they have little basis in fact.
As online disinformation continues to pose a threat to the American political system, it is imperative that solutions to combat it are based in fact and will not further polarize Americans, thereby leading to more disinformation and making the situation worse. While all the solutions treat online disinformation as though it is the cause of the breakdown in healthy discourse, it should be regarded as a symptom to that deeper problem. Unfortunately, all the proposed solutions which I review fail these criteria. By contrast, my solution will reduce polarization because restoring healthy discourse and public faith will help lessen the market for disinformation anyway. I believe that the only solution which will lead to the cessation of online disinformation is a long, complex, and difficult process of returning to healthy, constructive discourse that will benefit all Americans.
I would define constructive discourse as discourse which does not demonize political opponents, values truth above profit or political gain, and works to find comprehensive, nonpartisan, long-term solutions to issues. For example, bipartisanship in Congress used to be commonplace if a bill attempted to fix an issue which affected all Americans. Now, members of Congress are demonized if they vote ‘with the enemy,’ most recently on the Infrastructure Bill which just passed Congress. When all Americans got their news from Walter Cronkite, for example, there was a sense that he was an honest, fair commentator of the news, and not hyper-partisan or profit-driven pundit. Now, there are hundreds of different news sources, many of which have no qualms about unfairly smearing political opponents or selling away their journalistic integrity. It is clear that returning to healthy discourse will be a long and arduous process, as it requires us to completely change how politics is currently conducted in this nation. However, there are things we can do to help start this process. Some ways which could help bring this about would be legacy media companies no longer publishing sensationalistic content, politicians and pundits prioritizing policy solutions rather than publicity stunts, and everybody working to find common ground rather than demonizing political opponents in an ‘us vs. them’ mentality. These measures will help restore constructive discourse in our society, and, as a result, help reduce disinformation. While this process requires some work from all of us, it is imperative that we do this if we are to save our country.
""",
  "nichols": """Americans have had a long respite from thinking about nuclear war. The Cold War ended more than 30 years ago, when the Soviet Union was dismantled and replaced by the Russian Federation and more than a dozen other countries. China at the time was not yet a significant nuclear power. A North Korean bomb was purely a notional threat. The fear of a large war in Europe escalating into a nuclear conflict faded from the public’s mind.
Today, the Chinese nuclear arsenal could destroy most of the United States. The North Koreans have a stockpile of bombs. And the Russian Federation, which inherited the Soviet nuclear arsenal, has launched a major war against Ukraine. As the war began, Russian President Vladimir Putin ordered his nation’s nuclear forces to go on heightened alert and warned the West that any interference with the invasion would have “consequences that you have never experienced in your history.” Suddenly, the unthinkable seems possible again.
There was a time when citizens of the United States cared about nuclear weapons. The reality of nuclear war was constantly present in their lives; nuclear conflict took on apocalyptic meaning and entered the American consciousness not only through the news and politics, but through popular culture as well. Movie audiences in 1964 laughed while watching Peter Sellers play a president and his sinister adviser in Dr. Strangelove, bumbling their way to nuclear war; a few months later, they were horrified as Henry Fonda’s fictional president ordered the sacrificial immolation of New York City in Fail-Safe. Nuclear war and its terminology—overkill, first strike, fallout—were soon constant themes in every form of entertainment. We not only knew about nuclear war; we expected one.
But during the Cold War there was also thoughtful engagement with the nuclear threat. Academics, politicians, and activists argued on television and in op-ed pages about whether we were safer with more or fewer nuclear weapons. The media presented analyses of complicated issues relating to nuclear weapons. CBS, for example, broadcast an unprecedented five-part documentary series on national defense in 1981. When ABC, in 1983, aired the movie The Day After—about the consequences of a global nuclear war for a small town in Kansas—it did so as much to perform a public service as to achieve a ratings bonanza. Even President Ronald Reagan watched the movie. (In his diary, he noted that The Day After was “very effective” and had left him “greatly depressed.”)
I was among those who cared a lot about nuclear weapons. In the early days of my career, I was a Russian-speaking “Sovietologist” working in think tanks and with government agencies to pry open the black box of the Kremlin’s strategy and intentions. The work could be unsettling. Once, during a discussion of various nuclear scenarios, a colleague observed matter-of-factly, “Yes, in that one, we only lose 40 million.” He meant 40 million people.
The end of the Cold War, however, led to an era of national inattentiveness toward nuclear issues. We forgot about nuclear war and concentrated mostly on keeping nuclear weapons out of the “wrong hands,” which reflected the American preoccupation with rogue states and terrorists after 9/11. This change in emphasis had worrisome side effects. In 2008, a blue-ribbon commission headed by a former secretary of defense, James Schlesinger, sounded the alarm: A new generation of nuclear-weapons personnel in the Air Force and Navy did not understand its own mission. In 2010, the chairman of the Joint Chiefs of Staff, Admiral Michael Mullen, warned that American defense institutions were no longer minting nuclear strategists. “We don’t have anybody in our military that does that anymore,” Mullen said.
I saw this firsthand at the Naval War College, a graduate school for mid-level and senior U.S. military officers, where I taught for more than 25 years. Nuclear issues fell out of the curriculum almost immediately after the Cold War ended. I remember an Air Force major coming up to me after class and telling me he’d never heard of “mutual assured destruction”—the concept that underlies nuclear deterrence—until my lecture that day.
Voters no longer cared either. During the Cold War, regardless of what other issues might be raised, every presidential election was shadowed by worry over whose finger would be on “the button.” In 1983, Reagan—hardly a detail-oriented president or master policy wonk—asked for an uninterrupted half hour of television during prime time to discuss his defense budget and his plans for a national missile-defense system, replete with charts and graphs. Millions of Americans watched. But in 2015, when Donald Trump was asked during the Republican Party primary debates about U.S. nuclear forces, he could only say, “With nuclear, the power, the devastation is very important to me.” Such an answer would once have been disqualifying for any candidate. This time, millions of Americans shrugged.
It was perhaps inevitable after the Cold War that serious thinking about nuclear weapons would be stashed away, in the words of a NATO nuclear planner some years ago, like “the crazy aunt in the attic.”
But the end of the Cold War did not resolve the most crucial question that has plagued nuclear strategists since 1945: What do nuclear weapons actually do for those who have them? The American security analyst Bernard Brodie declared in the mid-1950s that nuclear weapons represented the “end of strategy,” because no political goal could justify unleashing their apocalyptically destructive power. In the 1980s, the political scientist and nuclear-deterrence scholar Robert Jervis amplified the point, noting that “a rational strategy for the employment of nuclear weapons is a contradiction in terms.”
American leaders, however, didn’t have the luxury of declaring nuclear war to be insanity and then ignoring the subject. The dawn of the Cold War and the birth of the Bomb occurred almost exactly at the same time. The Soviet Union, once our ally, was now our foe, and soon its nuclear arsenal was pointed at us, just as ours was pointed right back. Someone had to think about what might come next.
When contemplating the outbreak of nuclear war, the British strategist Michael Howard always asked: What would such a war be about? Why would it happen at all?
History supplies an answer, and reminds us that the perils of the past remain with us today. The American nuclear arsenal was constructed as the United States dealt with a series of postwar crises. From the Berlin blockade to a hot war in Korea, Communist dangers seemed to be spreading unchecked across the planet. By 1950, the Communist bloc extended from the Gulf of Finland to the South China Sea. With America and its allies outnumbered and outgunned, nuclear weapons and the threat of their use seemed to be the only Western recourse.
Nuclear planning in this period was shaped by the inescapable dictates of geography. The Soviet Union straddled two continents and spanned 11 time zones. The United States was relatively safe in its North American fortress from anything but an outright Soviet nuclear attack. But how could Washington protect NATO in Europe and its other allies scattered around the world? With Germany a divided nation and Berlin a divided city, any future conflict in Europe would always favor the Soviets and their tanks, which could roll across the plains almost at will.
This set up the basic structure of some future World War III in a way that every American of that period could understand: No matter how or where East and West might come into significant military conflict, the Soviets were certain to move the confrontation to Europe. A crisis might begin somewhere else—maybe the Caribbean, maybe the Middle East—but war itself would move to Germany and then spiral into a global catastrophe. American strategists tried to think through the possibility of “limited” nuclear wars in various regions, but as Schlesinger later admitted to Congress, none of the scenarios stayed limited for long. Everything came back to escalation in Europe.
This was not an idle fear. In 1965, for example, when the United States began bombing North Vietnam, the Soviet General Staff proposed a “military demonstration” of an unspecified nature aimed at Berlin and West Germany. “We do not fear approaching the risk of war,” the Soviet defense minister told Leonid Brezhnev and other Soviet leaders. The leadership declined the defense minister’s advice, and the episode was kept secret for decades. But the Kremlin and its high command continued to plan for defeating NATO quickly and decisively in Germany, no matter where a crisis might begin. They knew it was their best option, and so did we.
Once war moved to Central Europe, events would cascade with a brutal inevitability. The only way the United States could stop such an attack would be to resort to the immediate use of small, short-range nuclear arms on the battlefield. As Soviet forces advanced, we would strike them—on NATO’s own territory—with these “tactical” weapons. The Soviets would respond in kind. We would then hit more targets throughout Eastern Europe with larger and longer-range weapons, hoping to bring the Soviets to a halt. Again, the Soviets would respond. With so many nuclear weapons in play, and with chaos and panic enveloping national leaders, one side or the other might fear a larger attack and give in to the temptation to launch a preemptive strike against strategic nuclear weapons in the American or Soviet heartland. All-out nuclear war would follow. Millions would die immediately. Millions more would perish later.
The U.S. and NATO not only expected this nuclear escalation but threatened to be the ones to initiate it. There was a terrifying but elegant logic to this policy. In effect, the West told the Kremlin that the use of nuclear weapons would occur not because some unhinged U.S. president might wish it, but because Soviet successes on the battlefield would make it an inescapable choice.
By the 1960s, the march of technology had allowed both East and West to develop a “triad” of bombers, submarine-launched missiles, and land-based intercontinental missiles. Arsenals on both sides soon numbered in the tens of thousands. At these levels, even the most aggressive Cold War hawks knew that, in a full exchange, mutual obliteration was inevitable. Detailed and exacting war plans would collapse in days—or even hours—into what the nuclear strategist Herman Kahn called “spasm” or “insensate” war, with much of the Northern Hemisphere reduced to a sea of glass and ash.
The reality that nuclear war meant complete devastation for both sides led to the concept of mutual assured destruction, or MAD, a term coined by American war planners. MAD was at first not so much a policy as a simple fact. In the early 1970s, the United States proposed that both sides turn the fact into a defined policy: The superpowers would recognize that they had enough weapons and it was time to set limits. The Soviets, with some reservations, agreed. The race to oblivion was put on pause.
Today, MAD remains at the core of strategic deterrence. The United States and Russia have taken some weapons off their quick triggers, but many remain ready to launch in a matter of minutes. By treaty, Washington and Moscow have limited themselves to 1,550 warheads apiece. The basic idea is that these numbers deny either side the ability to take out the other’s arsenal in a first strike, while still preserving the ability to destroy at least 150 urban centers in each country. This, in the world of nuclear weapons, is progress.
The fall of the Soviet Union changed many things, but in nuclear matters it changed almost nothing. The missiles and their warheads remained where they were. They continue to wait in silent service. The crews in silos, submarines, and bombers now consist of the grandchildren and great-grandchildren of the people who built the first nuclear weapons and created the plans for their use. And yet for years we have conducted international politics as if we have somehow solved the problem of nuclear war.
Nuclear weapons are a crutch we have leaned on to avoid thinking about the true needs and costs of defense. With hardly any debate, over a period of 30 years we doubled the number of nations under NATO’s nuclear guarantee. We have talked about drawing down forces in places such as South Korea and shied away from expensive decisions about increasing our naval power in the Pacific—all because we think that nuclear weapons will remedy imbalances in conventional weapons and that the mere existence of nuclear weapons will somehow stabilize these unstable situations. Worrying about whether this broad reliance on nuclear deterrence risks escalation and nuclear war seems outdated to many. Memories of the Cold War, a young scholar once said to me, are a form of “baggage” that inhibits the making of bold policy.
This brings us, of course, to Ukraine. The war there could put four nuclear-armed powers—Russia, the United States, the United Kingdom, and France—onto the same battlefield, and yet arguments over the U.S. and NATO response to the Russian invasion have sometimes taken place in a nuclear void. President Joe Biden has rallied a global coalition against Moscow while remaining determined to avoid a direct military conflict with Russia. He wisely declined to raise U.S. nuclear readiness to match Putin’s nuclear alert. But he has had to steer this careful path while buffeted by demands from people who seem unmoved (or untouched) by memories of the Cold War. Calls for a more aggressive confrontation with Russia, including demands for a no-fly zone over Ukraine, backed by American power, have been advanced by a range of prominent figures. Republican Representative Adam Kinzinger even introduced a congressional resolution authorizing Biden to use American military force against Russia.
These demands ignore the reality, as the Harvard professor Graham Allison wrote earlier this year, that in the event of a hot war between nuclear superpowers, “the escalation ladder from there to the ultimate global catastrophe of nuclear war can be surprisingly short.” Allison’s warning is especially relevant today, when Russia and NATO have effectively switched places: Russia is now the inferior conventional power, and is threatening a first use of nuclear weapons if faced with a regime-threatening defeat on the battlefield.
Our collective amnesia—our nuclear Great Forgetting—undermines American national security. American political leaders have a responsibility to educate the public about how, and how much, the United States relies on nuclear weapons for its security. If we mean to reduce U.S. conventional forces and go back to relying on nuclear weapons as a battlefield equalizer, then the public should know it and think about it. If the U.S. nuclear arsenal exists solely to deter the use of enemy nuclear weapons, then it is time to say so and spell out the consequences.
Every presidential administration since 1994 has released a “nuclear posture review” that supposedly answers the question of why, exactly, America has a nuclear arsenal. Is it to fight nuclear wars or to deter a nuclear attack? And every administration has fudged the response by saying, essentially, it’s a little of both. This is not a serious answer. And it avoids the deeper question: If we do not in fact wish to use nuclear weapons, then what must we do to ensure that our conventional capabilities match our international commitments?
We have accepted evasions from our leaders because we take strategic nuclear deterrence for granted—as something that exists around us almost independently, like gravity or the weather. But deterrence relies on human psychology and on the agency and decisions of actual people, who must continually manage it.
Decades of denial have left Americans ill-prepared to think about the many choices that keep the nuclear peace. Effective deterrence, even in a post–Cold War world, requires the capacity to face the reality of nuclear war squarely. And it means understanding once again what it would feel like to hear the sirens—and to wonder whether they are only a drill.
""",
  "menand": """To look on the bright side for a moment, one effect of the Republican assault on elections—which takes the form, naturally, of the very thing Republicans accuse Democrats of doing: rigging the system—might be to open our eyes to how undemocratic our democracy is. Strictly speaking, American government has never been a government “by the people.”
This is so despite the fact that more Americans are voting than ever before. In 2020, sixty-seven per cent of eligible voters cast a ballot for President. That was the highest turnout since 1900, a year when few, if any, women, people under twenty-one, Asian immigrants (who could not become citizens), Native Americans (who were treated as foreigners), or Black Americans living in the South (who were openly disenfranchised) could vote. Eighteen per cent of the total population voted in that election. In 2020, forty-eight per cent voted.
Some members of the loser’s party have concluded that a sixty-seven-per-cent turnout was too high. They apparently calculate that, if fewer people had voted, Donald Trump might have carried their states. Last year, according to the Brennan Center for Justice, legislatures in nineteen states passed thirty-four laws imposing voting restrictions. (Trump and his allies had filed more than sixty lawsuits challenging the election results and lost all but one of them.)
In Florida, it is now illegal to offer water to someone standing in line to vote. Georgia is allowing counties to eliminate voting on Sundays. In 2020, Texas limited the number of ballot-drop-off locations to one per county, insuring that Loving County, the home of fifty-seven people, has the same number of drop-off locations as Harris County, which includes Houston and has 4.7 million people.
Virtually all of these “reforms” will likely make it harder for some people to vote, and thus will depress turnout—which is the not so subtle intention. This is a problem, but it is not the fundamental problem. The fundamental problem is that, as the law stands, even when the system is working the way it’s designed to work and everyone who is eligible to vote does vote, the government we get does not reflect the popular will. Michael Kinsley’s law of scandal applies. The scandal isn’t what’s illegal. The scandal is what’s legal.
It was not unreasonable for the Framers to be wary of direct democracy. You can’t govern a nation by plebiscite, and true representative democracy, in which everyone who might be affected by government policy has an equal say in choosing the people who make that policy, had never been tried. So they wrote a rule book, the Constitution, that places limits on what the government can do, regardless of what the majority wants. (They also countenanced slavery and the disenfranchisement of women, excluding from the electorate groups whose life chances certainly might be affected by government policy.) And they made it extremely difficult to tinker with those rules. In two hundred and thirty-three years, they have been changed by amendment only nine times. The last time was fifty-one years ago.
You might think that the further we get from 1789 the easier it would be to adjust the constitutional rule book, but the opposite appears to be true. We live in a country undergoing a severe case of ancestor worship (a symptom of insecurity and fear of the future), which is exacerbated by an absurdly unworkable and manipulable doctrine called originalism. Something that Alexander Hamilton wrote in a newspaper column—the Federalist Papers are basically a collection of op-eds—is treated like a passage in the Talmud. If we could unpack it correctly, it would show us the way.
The Bill of Rights, without which the Constitution would probably not have been ratified, is essentially a deck of counter-majoritarian trump cards, a list, directed at the federal government, of thou-shalt-nots. Americans argue about how far those commandments reach. Is nude dancing covered under the First Amendment’s guarantee of the freedom of expression? (It is.) Does the Second Amendment prohibit a ban on assault weapons? (Right now, it’s anyone’s guess.) But no one proposes doing away with the first ten amendments. They underwrite a deeply rooted feature of American life, the “I have a right” syndrome. They may also make many policies that a majority of Americans say they favor, such as a ban on assault weapons, virtually impossible to enact because of an ambiguous sentence written in an era in which pretty much the only assault weapon widely available was a musket.
Some checks on direct democracy in the United States are structural. They are built into the system of government the Framers devised. One, obviously, is the Electoral College, which in two of the past six elections has chosen a President who did not win the popular vote. Even in 2020, when Joe Biden got seven million more votes than his opponent, he carried three states that he needed in order to win the Electoral College—Arizona, Georgia, and Pennsylvania—by a total of about a hundred thousand votes. Flip those states and we would have elected a man who lost the popular vote by 6.9 million. Is that what James Madison had in mind?
Another check on democracy is the Senate, an almost comically malapportioned body that gives Wyoming’s five hundred and eighty thousand residents the same voting power as California’s thirty-nine million. The District of Columbia, which has ninety thousand more residents than Wyoming and twenty-five thousand more than Vermont, has no senators. Until the Seventeenth Amendment was ratified, in 1913, senators were mostly not popularly elected. They were appointed by state legislatures. Republicans won a majority of votes statewide in Illinois in the 1858 midterms, but Abraham Lincoln did not become senator, because the state legislature was controlled by Democrats, and they reappointed Stephen A. Douglas.
Even though the Senate is split fifty-fifty, Democratic senators represent forty-two million more people than Republican senators do. As Eric Holder, the former Attorney General, points out in his book on the state of voting rights, “Our Unfinished March” (One World), the Senate is lopsided. Half the population today is represented by eighteen senators, the other half by eighty-two. The Senate also packs a parliamentary death ray, the filibuster, which would allow forty-one senators representing ten per cent of the public to block legislation supported by senators representing the other ninety per cent.
Many recent voting regulations, such as voter-I.D. laws, may require people to pay to obtain a credential needed to vote, like a driver’s license, and so Holder considers them a kind of poll tax—which is outlawed by the Twenty-fourth Amendment. (Lower courts so far have been hesitant to accept this argument.)
But the House of Representatives—that’s the people’s house, right? Not necessarily. In the 2012 Presidential election, Barack Obama defeated Mitt Romney by five million votes, and Democrats running for the House got around a million more votes than Republicans, but the Republicans ended up with a thirty-three-seat advantage. Under current law, congressional districts within a state should be approximately equal in population. So how did the Republicans get fewer votes but more seats? It’s the same thing that let Stephen A. Douglas retain his Senate seat in 1858: partisan gerrymandering.
This is the subject of Nick Seabrook’s timely new book, “One Person, One Vote: A Surprising History of Gerrymandering in America” (Pantheon), an excellent, if gloomy, guide to the abuse (or maybe just the use) of an apparently mundane feature of our system of elections: districting.
We tend to think of a “gerrymander” as a grotesquely shaped legislative district, such as the salamander-like Massachusetts district that was drawn to help give one party, the Democratic-Republicans, a majority in the Massachusetts Senate in the election of 1812. The governor of the state, Elbridge Gerry, did not draw the district, but he lent his name to the practice when he signed off on it. (Seabrook tells us that Gerry’s name is pronounced with a hard “G,” but it’s apparently O.K. to pronounce gerrymander “jerry.”)
Gerry’s gerrymander was by no means the first, however. There was partisan gerrymandering even in the colonies. In fact, “the only traditional districting principle that has been ubiquitous in America since before the founding,” Seabrook writes, “is the gerrymander itself.” That’s the way the system was set up.
Partisan gerrymandering has produced many loopy districts through the years, but today, on a map, gerrymandered districts often look quite respectable. No funny stuff going on here! That’s because computer software can now carve out districts on a street-by-street and block-by-block level. A favorite trick is moving a district line so that a sitting member of Congress or a state legislator is suddenly residing in another district. It’s all supposed to be done sub rosa, but, Seabrook says, “those in the business of gerrymandering have a tendency to want to brag about their exploits.”
You might think that you can’t gerrymander a Senate seat, but the United States Senate itself is a product of gerrymandering. One factor that determined whether a new state would be admitted to the Union was which political party was likely to benefit. We have two Dakotas in part because Republicans were in power in Washington, and they figured that splitting the Dakota territory in two would yield twice as many new Republican senators.
For there’s nothing natural about states. Portions of what is now Wyoming were, at various times, portions of the territories of Oregon, Idaho, Dakota, Nebraska, and Utah. Before 1848, much of Wyoming was Mexican. Before that, it was Spanish. We don’t have Wyoming because people living within the territory felt a special affinity, a belief that they shared a “Wyoming way of life,” and somebody said, “These folks should have their own state.” To the extent that Wyoming residents feel stately solidarity, it’s because the federal government created Wyoming (and two more Republican senators), not the other way around.
In the case of the House, reapportionment takes place every ten years, after the census is reported. When this happens, most states redistrict not only for Congress but for their own legislative offices as well. This means, usually, that the party in power in state government that year draws district lines that will be in place for the next decade. Republicans, when they are running the show, try to make it harder for Democrats on every level to win, and vice versa. And why not? It’s human nature.
Even the census, on which apportionment is based, is subject to partisan manipulation. Was it at all surprising to learn recently that the Trump Administration tried to interfere with the 2020 census in order to reduce the population in Democratic districts? Trump officials must have calculated that they had little to lose. If they failed (which they largely did, after the Supreme Court suggested that the Administration was lying about its intentions and officials at the Census Bureau pushed back), no harm, no foul. If they succeeded and someone called them out on it, what was anybody going to do about it? Administer a new census?
The name of the game in partisan redistricting is vote dilution. In a two-party race, a candidate needs only fifty per cent plus one to win. Every extra vote cast for that candidate is a wasted vote, as is every vote for the loser. You can’t literally prevent your opponents from voting. Even the current Supreme Court, which has hardly been a champion of voting rights since John Roberts became Chief Justice, would put a stop to that. So wasting as many of the other party’s votes as possible is the next best thing. And, in most states, it’s perfectly legal. The terms of art are “cracking” and “packing.”
You crack a district when you break up a solid voting bloc for one party and distribute those voters across several adjacent districts, where they are likely to be in the minority. Once it’s cracked, the formerly solid district becomes competitive. This is sometimes called “dispersal gerrymandering.”
When you pack, on the other hand, you put as many voters of the other party as possible into the same district. This arrangement means that their candidate will usually get a seat, but it weakens that party’s power in other districts. From a civil-rights point of view, districts in which members of minority groups are in the majority might seem like a good thing, but Republicans tend to favor majority-minority districts because they reduce the chances that Democratic candidates will win elsewhere in a state.
Partisan redistricting is why Republicans won five of Wisconsin’s eight congressional seats in 2020 even though Biden took the state. Biden carried the Fourth Congressional District, which includes Milwaukee, by fifty-four percentage points. Was that district packed? Not necessarily. The tendency of Democrats to concentrate in densely populated urban areas naturally tends to dilute their votes statewide. But partisan redistricting helps explain why Republicans won sixty-one of ninety-nine seats in Wisconsin’s State Assembly and ten of the sixteen contested seats in the State Senate. Wisconsin is justifiably considered a major success story by Republican redistricting strategists.
Partisan gerrymandering is also why, for most of the past half century, the State Senate in New York was Republican and the State Assembly was Democratic—a formula for gridlock, backroom dealing, and the inequitable distribution of resources. Seabrook explains that New York’s districting was solidified under a handshake agreement that gave each party control of the process for one legislative chamber. The parties therefore created as many safe districts for their candidates as possible. Seabrook calls New York a “criminal oligarchy” and notes that, between 2005 and 2015, at least thirty state officials were involved in corruption cases.
Eight years ago, by constitutional amendment, New York established a bipartisan independent redistricting commission and made partisan gerrymandering illegal. This cycle, the commission deadlocked, and the Democrats, who have a supermajority in both houses of the legislature, tried to build a loophole in the law and drew their own maps. The State Senate and congressional maps were promptly thrown out as illegal partisan gerrymandering by the New York State Court of Appeals, and a lower court presented new maps, which govern the 2022 elections. The result is that New York Democrats now find themselves competing with one another for the same seats. The new district lines may force one candidate to move in with his mother, in order to maintain residency. Chaos? Just business as usual in New York State government.
Understanding the gerrymander helps us understand what Jacob Grumbach, in “Laboratories Against Democracy” (Princeton), describes as a country “under entrenched minority rule.” Grumbach is a quantitative political scientist, and his data suggest that, although some state governments have moved to the extremes, public opinion in those states has remained fairly stable. What explains the political shift, he thinks, is that all politics has become national.
“The state level is increasingly dominated by national groups who exploit the low-information environments of amateurish and resource-constrained legislatures, declining local news media, and identity-focused voters,” Grumbach maintains. These national groups aim to freeze out the opposition, and redistricting is a powerful tool for that. “Antidemocratic interests need only to take control of a state government for a short period of time,” Grumbach points out, “to implement changes that make it harder for their opponents to participate in politics at all levels.”
Partisan redistricting often favors rural areas. Obviously, the Senate and the Electoral College do this, too. One thumb on that scale is what is called prison gerrymandering. There are more than a million incarcerated convicts in the land of the free. Except in Maine, Vermont, and D.C., none can vote. But in many states, for purposes of congressional apportionment, they are counted as residents of the district in which they are imprisoned.
Seabrook says that seventy per cent of prisons built since 1970 are in rural areas, and that a disproportionate number of the people confined in them come from cities. Counting those prisoners in apportionment enhances the electoral power of rural voters. It’s a little like what happened after Emancipation. A Southern state could now count formerly enslaved residents as full persons, rather than as three-fifths of a person, and was reapportioned accordingly. Then it disenfranchised them.
Changing the Senate would require a changed Constitution, and there is little chance of that. There is a movement under way to get states to pass laws requiring their Presidential electors to vote for whoever wins the national popular vote, which is a way of reforming the Electoral College system without changing the Constitution. This, too, is a long shot. Elected officials have no incentive to change a system that keeps electing them.
Suppose, however, that we went over the heads of elected officials and appealed to our lifetime-tenured Supreme Court Justices, who, wielding the power of judicial review (not mentioned in the Constitution), can nullify laws with the stroke of a pen and suffer no consequences? The Justices are not even required to recuse themselves from cases in which they might have personal involvement. No other democracy in the world has a judicial system like that, and for a good reason: it’s not very democratic.
But, precisely because they have no stake in the electoral status quo, the Justices might decide that gerrymandered vote dilution triggers, among other constitutional provisions, the equal-protection clause of the Fourteenth Amendment. It seems pretty clear that your right to vote isn’t very “equal” if someone else’s vote is worth more.
In 2016, the North Carolina Democratic Party, the watchdog group Common Cause, and fourteen North Carolina voters sued the state legislators who had led a partisan redistricting effort designed to create ten congressional seats for Republicans and three for Democrats. The case, Rucho v. Common Cause, was joined with a similar case from Maryland. In that one, it was Republicans who sued, contesting a redistricting plan that reduced the number of G.O.P. congressional seats from two or three to one. The North Carolina plaintiffs won in district court.
In 2019, however, the Supreme Court, in a 5–4 decision (Ruth Bader Ginsburg was still alive), vacated the lower court’s decision and ordered that the suits be dismissed for lack of jurisdiction. The Court’s opinion was written by Roberts, who has been a critic of expanded voting rights since his time as a special assistant to the Attorney General in the first Reagan Administration. Roberts did not deny that the partisan gerrymandering in North Carolina and Maryland was extreme; he simply ruled that federal courts have no business interfering.
Roberts invoked what is known as the political-question doctrine, arguing that the degree of partisanship in redistricting is a political, not a judicial, matter. It admits of no judicial solution. “Excessive partisanship in districting leads to results that reasonably seem unjust,” Roberts conceded. “But the fact that such gerrymandering is ‘incompatible with democratic principles’ . . . does not mean that the solution lies with the federal judiciary.” The matter was deemed “nonjusticiable.”
It might seem shocking that the Court could take cognizance of an undemocratic practice and then decline to do anything about it. But Rucho should not have been a surprise. In 1986, the Court said that gerrymandering could violate the Constitution, but it has never struck down a partisan gerrymander. The Warren Court’s famous one-person-one-vote cases, highly contentious in their day, culminated in Reynolds v. Sims (1964), which held that legislative districts for all state offices, including State Senate seats, “must be apportioned on a population basis.”
These cases made malapportionment illegal, but not gerrymandering. In fact, Seabrook thinks, the one-person-one-vote rule is responsible for what he calls “the Frankenstein’s monster of the modern gerrymander.” As long as district populations are equal, you can crack and pack all you like; you just need the right software, and the Supreme Court will look the other way.
There is one major exception, however. Federal courts will strike down a gerrymander intended to dilute the votes of racial minorities. You can redistrict by political party, in other words, but not by race. That is plainly barred by the Fifteenth Amendment and the 1965 Voting Rights Act. In Cooper v. Harris, from 2017, the Roberts Court invalidated a North Carolina districting plan on the ground that it grouped voters to weaken the minority vote.
Shouldn’t this approach extend to state voting regulations as well? Houston has a large nonwhite population (but will likely have only one drop box); Southern Blacks have a tradition of voting after church services on Sundays (but may no longer be able to do so); and nonwhites are more likely than whites to have to stand in long lines in order to vote (and thus be grateful for some water). Are these new regulations really race-neutral?
In 2021, in Brnovich v. Democratic National Committee, the Court upheld a new Arizona law making it a crime for anyone other than a postal worker, election official, caregiver, or family or household member to collect and deliver an early ballot—targeting a practice common in minority communities. The Democratic National Committee sued, claiming that the law had a disparate impact on, among other groups, Native American Arizonans, many of whom live on reservations that are distant from a polling place. The Court held that the restriction was legal. “Mere inconvenience,” it said, “cannot be enough” to demonstrate that a group’s voting rights have been violated.
Is the motive for redistricting partisan, or is it racial? In a nation in which race is often a determinant of party identity, this will be a tricky needle to thread. Still, the Court isn’t wrong to point out that there is a political solution to the movement to restrict voting rights. Under the Constitution, although the states prescribe the “Times, Places and Manner of holding Elections,” Congress “may at any time by Law make or alter such Regulations,” and thereby make voting easier. What do you think the chances are of that happening?
"""
}

if len(sys.argv) > 1:
  arg1 = sys.argv[1]
  if arg1.isdigit():
    essay_list = list(essays.keys())
    length = len(essay_list)
    if arg1 >= length:
      print(f"ERROR: {arg1} is greater than length of essay list ({length})")
    else:
      essay = essays[essay_list[arg1]]
      print(f"running predictor on essay by {essay_list[arg1]}")
      run_predictor(essay, use_tokenizer=True)
  else:
    print(f"ERROR: {arg1} is not a valid essay")
else:
  print(f"ERROR: no essay specified")

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

from gensim.models import Word2Vec
import gensim.downloader as vec_api
vec_model = vec_api.load("fasttext-wiki-news-subwords-300")

from numpy import False_
from transformers import BertTokenizer, BertForMaskedLM, BertForNextSentencePrediction
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import torch
import re
import nltk
import time
import math

start = time.time()

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
      "index_sum": 0,
      "not_found": 0,
      "similarity": 0,
      "categories": [0, 0, 0, 0, 0, 0, 0],
      "sentence_similarity": 0,
      "similarities": [],
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
    if res["index"] == SEARCH_LIMIT:
      self.data["not_found"] += 1
    self.data["index_sum"] += res["index"]
    if res["similarity"] != "Not Found":
      self.data["similarity"] += res["similarity"]
      self.data["similarities"].append(res["similarity"])
    self.data["categories"][res["category"] - 1] += 1
  
  def add_obj(self, res):
    self.data["correct"] += res["correct"]
    self.data["incorrect"] += res["incorrect"]
    self.data["index_sum"] += res["index_sum"]
    self.data["not_found"] += res["not_found"]
    self.data["similarity"] += res["similarity"]
    self.data["incorrect_similarity"] += res["incorrect_similarity"]
    self.data["similarities"] += res["similarities"]
    if res["sentence_similarity"] != 0:
      self.data["sentence_similarity"] += res["sentence_similarity"]
    if res["nsp_score"] != 0:
      self.data["nsp_score"] += res["nsp_score"]
    for i in range(len(res["categories"])):
      self.data["categories"][i] += res["categories"][i]

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
    print(f"\nAverage Index         = {round(self.data['index_sum'] / total, 1)}")
    print(f"Indexes Not Found     = {self.data['not_found']} {get_percent(self.data['not_found'], total)}")
    print(f"Average Similarity    = {round(self.data['similarity'] / total, 2)}")
    if (self.data['incorrect'] != 0):
      print(f"Incorrect Similarity  = {round(self.data['incorrect_similarity'] / self.data['incorrect'], 2)}\n")

    plt.hist(self.data["similarities"], 10, (0, 100), color = 'green', histtype = 'bar', rwidth = 0.8)
    plt.xlabel('Similarity Scores')
    plt.ylabel('Number of Appearances')
    plt.title('Similarity Score Distribution')
    plt.show()

    print("\nPredictions by Index Category:\n")
    categories_txt = ["0", "1", "2-9", "10-99", "100-999", "1000-4999", "5000-Not Found"]
    categories_sum = 0
    for i in range(len(self.data['categories'])):
      categories_sum += self.data['categories'][i] * (i + 1)
      txt = f"#{i + 1} ({categories_txt[i]}) "
      print(f"{pad_word(txt, 22)}= {self.data['categories'][i]} {get_percent(self.data['categories'][i], total)}")
    div = categories_sum / total
    print(f"\nAverage Category      = {round(div, 1)} (~{categories_txt[round(div) - 1]})")

def pad_word(input_str, length):
  for x in range(length - len(input_str)):
    input_str += ' '
  return input_str

def print_word(
    masked_word="Masked Word",
    predicted_word="Predicted Word",
    prediction_result="Prediction Result",
    correct_index="Index of Correct Word",
    similarity="Similarity",
    top_predictions="Next Three Predictions",
    prediction_category="Category",
    stop_word="Stop Word"
):
  print(f"| {pad_word(masked_word, 16)} ", end = '')
  print(f"| {pad_word(predicted_word, 16)} ", end = '')
  print(f"| {pad_word(prediction_result, 17)} ", end = '')
  print(f"| {pad_word(correct_index, 21)} ", end = '')
  print(f"| {pad_word(similarity, 10)} ", end = '')
  print(f"| {pad_word(top_predictions, 36)} ", end = '')
  print(f"| {pad_word(prediction_category, 8)} ", end = '')
  print(f"| {pad_word(stop_word, 9)} |")

def print_sep():
  print("--------------------------------------------------------------------------------------------------------------------------------------------------------------")

def pred_word(txt, correct_word):
  input = tokenizer.encode_plus(txt, return_tensors = "pt")
  mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
  output = model(**input)

  logits = output.logits
  softmax = F.softmax(logits, dim = -1)
  mask_word = softmax[0, mask_index, :]
  top = torch.topk(mask_word, SEARCH_LIMIT, dim = 1)[1][0]
  tokens = []
  for token in top:
    word = tokenizer.decode([token])
    if re.search(r'^\W+$', word) == None:
      tokens.append(word)
  if tokens[0] == correct_word:
    result = "CORRECT"
  else:
    result = "INCORRECT"
  try:
    index = tokens.index(correct_word)
  except:
    index = "Not Found"
  display_index = f"{index}"
  if index == "Not Found":
    index = SEARCH_LIMIT
  try:
    similarity = round(100 * float(vec_model.similarity(correct_word, tokens[0])), 2)
  except:
    similarity = "Not Found"
  if index == 0:
    category = 1
  elif index == 1:
    category = 2
  elif index < 10:
    category = 3
  elif index < 100:
    category = 4
  elif index < 1000:
    category = 5
  elif index < 5000:
    category = 6
  else:
    category = 7
  if correct_word in stop_word_list:
    is_stop = "TRUE"
  else:
    is_stop = "FALSE"
  print_word(
    masked_word=correct_word,
    predicted_word=tokens[0],
    prediction_result=result,
    correct_index=display_index,
    similarity=f"{similarity}",
    top_predictions=', '.join(tokens[1:4]),
    prediction_category=f"{category}",
    stop_word = is_stop
  )
  return {
      "result": result,
      "index": index,
      "similarity": similarity,
      "category": category,
      "pred_word": tokens[0]
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
  print_sep()
  print(f"Sentence similarity score: {sentence_similarity}")

  return stats

def get_percent(part, whole):
  if part == 0:
    return ""
  else:
    return f"({round((part / whole) * 100, 1)}%)"

def run_predictor(input_txt, use_tokenizer=False, sentence_format=False, ignore_proper=False):
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
      print(f"Next sentence prediction score: {score}")
      stats["with_stop"].add_nsp_score(score)
  
  for word in result_list:
    stats_obj = stats[f"{word}_stop"]
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

run_predictor("""Over the past twenty years, drone usage by the United States military has grown exponentially, mostly due to their role in the fight against terrorism. Airstrikes and drone reconnaissance have reached new highs in the past decade, with rates increasing from eight drone strikes per year in 2007 to 2,776 strikes in 2017 . The nature of drone technology blurs the line between military and non-military counterterrorism, which is only natural since the catalyst itself, terrorism, combines both military and non-military  contexts. As a result, drones raise new questions in ethics and morality, specifically regarding the overarching effects of distorting the lines between military and social contexts and whether drone use by the military is ultimately beneficial or harmful. Therefore, the debate has formed between those who support the traditional use of the military in the fight against terrorism, and those who believe the military must adapt its policies as situations change. More specifically, while both sides agree that there are advantages and drawbacks to the use of drones to fight terrorism, contention ensues in determining whether these advantages offset the disadvantages. In general, scholars approach this question in regard to either the social drawbacks or the military advantages of drone strikes. In contrast, I suggest that the most complete solution to the use of drones by the military considers both the advantages of drones in relation to traditional weapons and the external social effect of drone strikes in foreign countries.
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
In general, the arguments made by scholars regarding military drones are one-sided and incomplete. They frequently attempt to unblur the lines between military and social contexts, hoping that a valid solution can be reached from an examination of either side of the debate, rather than both. However, the thought process behind the solution to this issue is far more complex. Considering that the catalyst of drone development itself is terrorism, it is necessary that the drone operates in a gray area between military and non-military environments, as terrorists can only be eliminated through military-style attacks in non-military zones. Thus, any arguments failing to mention points on both sides of the issue can instantly be dismissed. Still, there remain scholars who make strong arguments for their view and then provide irrelevant opposing points that can easily be refuted- not because their argument is valid, but because the opposing point simply addresses a different issue. These arguments may be easily dismissed as well. The complete answer requires both critics and proponents. While the critic may argue that the social outfall is disastrous, and the proponent may argue that the drone is the most efficient military weapon, a true solution requires a combination and comparison between the two. While the complete solution is far more intricate, the essential ultimatum is this: if the drone’s targets are more dangerous and harmful than the social fallout that will occur as a result, then a drone strike is the best possible answer. If the external social effects are worse than the military gain from using a drone, then the military must use traditional methods.""")#, use_tokenizer=True)

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

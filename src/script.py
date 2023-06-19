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

def get_nsp(sentences):
  total_score = 0
  total_count = 0
  sentences = list(filter(lambda x: len(x.strip()) > 0, sentences))
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
  
def run_predictor(input_txt, use_tokenizer=False, sentence_format=False, ignore_proper=False, data=False, nsp_only=False):
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

  if nsp_only:
    return get_nsp(sentences)
    
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

def run_books():
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
    
def run_texts():
  texts = {
    "Stevens,  Not Ideas About the Thing": """At the earliest ending of winter,
In march, a scrawny cry from outside
Seemed like a sound in his mind.

He knew that he heard it,
A bird's cry at daylight or before,
In the early march wind.

The sun was rising at six,
No longer a battered panache above snow...
It would have been outside.

It was not from the vast ventriloquism
Of sleep's faded papier mâché...
The sun was coming from outside.

That scrawny cry it was
A chorister whose c preceded the choir.
It was part of the colossal sun,

Surrounded by its choral rings,
Still far away. It was like
A new knowledge of reality.""",
    "Keats,  To Autumn": """Season of mists and mellow fruitfulness, 
   Close bosom friend of the maturing sun; 
Conspiring with him how to load and bless 
   With fruit the vines that round the thatch eves run; 
To bend with apples the mossed cottage trees, 
   And fill all fruit with ripeness to the core; 
      To swell the gourd, and plump the hazel shells 
   With a sweet kernel; to set budding more, 
And still more, later flowers for the bees, 
Until they think warm days will never cease, 
      For summer has over brimmed their clammy cells. 

Who hath not seen thee oft amid thy store? 
   Sometimes whoever seeks abroad may find 
Thee sitting careless on a granary floor, 
   Thy hair soft lifted by the winnowing wind; 
Or on a half reaped furrow sound asleep, 
   Drowsed with the fume of poppies, while thy hook 
      Spares the next swath and all its twined flowers: 
And sometimes like a gleaner thou dost keep 
   Steady thy laden head across a brook; 
   Or by a cyder press, with patient look, 
      Thou watchest the last oozings hours by hours. 

Where are the songs of spring? Ay, Where are they? 
   Think not of them, thou hast thy music too,  
While barred clouds bloom the soft-dying day, 
   And touch the stubble plains with rosy hue; 
Then in a wailful choir the small gnats mourn 
   Among the river sallows, borne aloft 
      Or sinking as the light wind lives or dies; 
And full grown lambs loud bleat from hilly bourn; 
   Hedge crickets sing; and now with treble soft 
   The red breast whistles from a garden croft; 
      And gathering swallows twitter in the skies.""",
    "Wordsworth,  Westminster Bridge": """Earth has not any thing to show more fair:
Dull would he be of soul who could pass by
A sight so touching in its majesty:
This city now doth, like a garment, wear
The beauty of the morning; silent, bare,
Ships, towers, domes, theatres, and temples lie
Open unto the fields, and to the sky;
All bright and glittering in the smokeless air.
Never did sun more beautifully steep
In his first splendour, valley, rock, or hill;
Never saw I, never felt, a calm so deep!
The river glideth at his own sweet will:
Dear God! the very houses seem asleep;
And all that mighty heart is lying still!""",
    "Wordsworth,  Nuns Fret Not": """Nuns fret not at their convent’s narrow room;
And hermits are contented with their cells;
And students with their pensive citadels;
Maids at the wheel, the weaver at his loom,
Sit blithe and happy; bees that soar for bloom,
High as the highest peak of furness fells,
Will murmur by the hour in foxglove bells:
In truth the prison, into which we doom
Ourselves, no prison is: and hence for me,
In sundry moods, it was pastime to be bound
Within the sonnet’s scanty plot of ground;
Pleased if some souls (for such there needs must be)
Who have felt the weight of too much liberty,
Should find brief solace there, as I have found.""",
    "Wordsworth,  I wandered lonely as a cloud": """I wandered lonely as a cloud
That floats on high over vales and hills,
When all at once I saw a crowd,
A host, of golden daffodils;
Beside the lake, beneath the trees,
Fluttering and dancing in the breeze.

Continuous as the stars that shine
And twinkle on the milky way,
They stretched in never ending line
Along the margin of a bay:
Ten thousand saw I at a glance,
Tossing their heads in sprightly dance.

The waves beside them danced; but they
Outdid the sparkling waves in glee:
A poet could not but be gay,
In such a jocund company:
I gazed and gazed but little thought
What wealth the show to me had brought:

For oft, when on my couch I lie
In vacant or in pensive mood,
They flash upon that inward eye
Which is the bliss of solitude;
And then my heart with pleasure fills,
And dances with the daffodils.""",
    "Keats,  Ode on a Grecian Urn": """Thou still unravished bride of quietness,
       Thou foster child of silence and slow time,
Sylvan historian, who canst thus express
       A flowery tale more sweetly than our rhyme:
What leaf fringed legend haunts about thy shape
       Of deities or mortals, or of both,
               In tempe or the dales of arcady?
       What men or gods are these? What maidens loth?
What mad pursuit? What struggle to escape?
               What pipes and timbrels? What wild ecstasy?

Heard melodies are sweet, but those unheard
       Are sweeter; therefore, ye soft pipes, play on;
Not to the sensual ear, but, more endeared,
       Pipe to the spirit ditties of no tone:
Fair youth, beneath the trees, thou canst not leave
       Thy song, nor ever can those trees be bare;
               Bold lover, never, never canst thou kiss,
Though winning near the goal yet, do not grieve;
       She cannot fade, though thou hast not thy bliss,
               For ever wilt thou love, and she be fair!

Ah, happy, happy boughs! that cannot shed
         Your leaves, nor ever bid the spring adieu;
And, happy melodist, unwearied,
         For ever piping songs for ever new;
More happy love! more happy, happy love!
         For ever warm and still to be enjoy'd,
                For ever panting, and for ever young;
All breathing human passion far above,
         That leaves a heart high sorrowful and cloyed,
                A burning forehead, and a parching tongue.

Who are these coming to the sacrifice?
         To what green altar, O mysterious priest,
Lead'st thou that heifer lowing at the skies,
         And all her silken flanks with garlands dressed?
What little town by river or sea shore,
         Or mountain built with peaceful citadel,
                Is emptied of this folk, this pious morn?
And, little town, thy streets for evermore
         Will silent be; and not a soul to tell
                Why thou art desolate, can ever return.

O attic shape! fair attitude! with breed
         Of marble men and maidens overwrought,
With forest branches and the trodden weed;
         Thou, silent form, dost tease us out of thought
As doth eternity: cold pastoral!
         When old age shall this generation waste,
                Thou shalt remain, in midst of other woe
Than ours, a friend to man, to whom thou say'st,
         "Beauty is truth, truth beauty, that is all
                Ye know on earth, and all ye need to know.""",
    "Keats,  Ode on a Nightingale": """My heart aches, and a drowsy numbness pains 
         My sense, as though of hemlock I had drunk, 
Or emptied some dull opiate to the drains 
         One minute past, and lethe-wards had sunk: 
'Tis not through envy of thy happy lot, 
         But being too happy in thine happiness,— 
                That thou, light winged dryad of the trees 
                        In some melodious plot 
         Of beechen green, and shadows numberless, 
                Singest of summer in full throated ease. 

O, for a draught of vintage! that hath been 
         Cooled a long age in the deep delved earth, 
Tasting of flora and the country green, 
         Dance, and provençal song, and sunburnt mirth! 
O for a beaker full of the warm south, 
         Full of the true, the blushful hippocrene, 
                With beaded bubbles winking at the brim, 
                        And purple stained mouth; 
         That I might drink, and leave the world unseen, 
                And with thee fade away into the forest dim: 

Fade far away, dissolve, and quite forget 
         What thou among the leaves hast never known, 
The weariness, the fever, and the fret 
         Here, where men sit and hear each other groan; 
Where palsy shakes a few, sad, last gray hairs, 
         Where youth grows pale, and spectre thin, and dies; 
                Where but to think is to be full of sorrow 
                        And leaden eyed despairs, 
         Where Beauty cannot keep her lustrous eyes, 
                Or new Love pine at them beyond tomorrow. 

Away! away! for I will fly to thee, 
         Not charioted by bacchus and his pards, 
But on the viewless wings of poesy, 
         Though the dull brain perplexes and retards: 
Already with thee! tender is the night, 
         And haply the queen moon is on her throne, 
                Clustered around by all her starry fays; 
                        But here there is no light, 
         Save what from heaven is with the breezes blown 
                Through verdurous glooms and winding mossy ways. 

I cannot see what flowers are at my feet, 
         Nor what soft incense hangs upon the boughs, 
But, in embalmed darkness, guess each sweet 
         Wherewith the seasonable month endows 
The grass, the thicket, and the fruit tree wild; 
         White hawthorn, and the pastoral eglantine; 
                Fast fading violets covered up in leaves; 
                        And mid-may's eldest child, 
         The coming musk-rose, full of dewy wine, 
                The murmurous haunt of flies on summer eves. 

Darkling I listen; and, for many a time 
         I have been half in love with easeful death, 
Called him soft names in many a mused rhyme, 
         To take into the air my quiet breath; 
                Now more than ever seems it rich to die, 
         To cease upon the midnight with no pain, 
                While thou art pouring forth thy soul abroad 
                        In such an ecstasy! 
         Still wouldst thou sing, and I have ears in vain 
                   To thy high requiem become a sod. 

Thou wast not born for death, immortal bird! 
         No hungry generations tread thee down; 
The voice I hear this passing night was heard 
         In ancient days by emperor and clown: 
Perhaps the self same song that found a path 
         Through the sad heart of ruth, when, sick for home, 
                She stood in tears amid the alien corn; 
                        The same that oft times hath 
         Charmed magic casements, opening on the foam 
                Of perilous seas, in faery lands forlorn. 

Forlorn! the very word is like a bell 
         To toll me back from thee to my sole self! 
Adieu! the fancy cannot cheat so well 
         As she is famed to do, deceiving elf. 
Adieu! adieu! thy plaintive anthem fades 
         Past the near meadows, over the still stream, 
                Up the hill-side; and now 'tis buried deep 
                        In the next valley-glades: 
         Was it a vision, or a waking dream? 
                Fled is that music: do I wake or sleep? """,
    "Coleridge,  Kubla Khan": """In xanadu did kubla khan
A stately pleasure dome decree:
Where alph, the sacred river, ran
Through caverns measureless to man
   Down to a sunless sea.
So twice five miles of fertile ground
With walls and towers were girdled round;
And there were gardens bright with sinuous rills,
Where blossomed many an incense bearing tree;
And here were forests ancient as the hills,
Enfolding sunny spots of greenery.

But oh! that deep romantic chasm which slanted
Down the green hill athwart a cedarn cover!
A savage place! as holy and enchanted
As ever beneath a waning moon was haunted
By woman wailing for her demon lover!
And from this chasm, with ceaseless turmoil seething,
As if this earth in fast thick pants were breathing,
A mighty fountain momently was forced:
Amid whose swift half intermitted burst
Huge fragments vaulted like rebounding hail,
Or chaffy grain beneath the thresher’s flail:
And mid these dancing rocks at once and ever
It flung up momently the sacred river.
Five miles meandering with a mazy motion
Through wood and dale the sacred river ran,
Then reached the caverns measureless to man,
And sank in tumult to a lifeless ocean;
And amid this tumult kubla heard from far
Ancestral voices prophesying war!
   The shadow of the dome of pleasure
   Floated midway on the waves;
   Where was heard the mingled measure
   From the fountain and the caves.
It was a miracle of rare device,
A sunny pleasure dome with caves of ice!

   A damsel with a dulcimer
   In a vision once I saw:
   It was an abyssinian maid
   And on her dulcimer she played,
   Singing of mount abora.
   Could I revive within me
   Her symphony and song,
   To such a deep delight it would win me,
That with music loud and long,
I would build that dome in air,
That sunny dome! those caves of ice!
And all who heard should see them there,
And all should cry, beware! beware!
His flashing eyes, his floating hair!
Weave a circle round him thrice,
And close your eyes with holy dread
For he on honeydew hath fed,
And drunk the milk of paradise.""",
    "Stevens,  The Snow Man": """One must have a mind of winter
To regard the frost and the boughs
Of the pine trees crusted with snow;

And have been cold a long time
To behold the junipers shagged with ice,
The spruces rough in the distant glitter

Of the january sun; and not to think
Of any misery in the sound of the wind,
In the sound of a few leaves,

Which is the sound of the land
Full of the same wind
That is blowing in the same bare place

For the listener, who listens in the snow,
And, nothing himself, beholds
Nothing that is not there and the nothing that is.""",
    "Stevens,  Anecdote of the Jar": """I placed a jar in tennessee,   
And round it was, upon a hill.   
It made the slovenly wilderness   
Surround that hill.

The wilderness rose up to it,
And sprawled around, no longer wild.   
The jar was round upon the ground   
And tall and of a port in air.

It took dominion everywhere.   
The jar was gray and bare.
It did not give of bird or bush,   
Like nothing else in tennessee.""",
    "Stevens,  Earthy Anecdote": """Every time the bucks went clattering
Over oklahoma
A firecat bristled in the way.

Wherever they went,
They went clattering,
Until they swerved,
In a swift, circular line,
To the right,
Because of the firecat.

Or until they swerved,
In a swift, circular line,
To the left,
Because of the firecat.

The bucks clattered.
The firecat went leaping,
To the right, to the left,
And
Bristled in the way.

Later, the firecat closed his bright eyes
And slept.""",
    "Eliot,  The Hippopotamus": """The broad backed hippopotamus
Rests on his belly in the mud;
Although he seems so firm to us
He is merely flesh and blood.
 
Flesh and blood is weak and frail,
Susceptible to nervous shock;
While the true church can never fail
For it is based upon a rock.
 
The hippo's feeble steps may err
In compassing material ends,
While the true church need never stir
To gather in its dividends.
 
The hippopotamus can never reach
The mango on the mango tree;
But fruits of pomegranate and peach
Refresh the church from over sea.
 
At mating time the hippo's voice
Betrays inflexions hoarse and odd,
But every week we hear rejoice
The church, at being one with god.
 
The hippopotamus's day
Is passed in sleep; at night he hunts;
God works in a mysterious way--
The church can sleep and feed at once.
 
I saw the hippopotamus take wing
Ascending from the damp savannas,
And quiring angels round him sing
The praise of God, in loud hosannas.
 
Blood of the lamb shall wash him clean
And him shall heavenly arms enfold,
Among the saints he shall be seen
Performing on a harp of gold.
 
He shall be washed as white as snow,
By all the martyred virgins kissed,
While the true church remains below
Wrapt in the old miasmal mist.""",
    "Eliot,  Mr Eliot's Sunday Morning Service": """Polyphiloprogenitive	
The sapient sutlers of the lord	
Drift across the window panes.	
In the beginning was the  word.	
 
In the beginning was the  word.
Superfetation of ,	
And at the mensual turn of time	
Produced enervate origen.	
 
A painter of the umbrian school	
Designed upon a gesso ground
The nimbus of the baptized god.	
The wilderness is cracked and browned	
 
But through the water pale and thin	
Still shine the unoffending feet	
And there above the painter set
The father and the paraclete.

The sable presbyters approach	
The avenue of penitence;	
The young are red and pustular	
Clutching piaculative pence.
 
Under the penitential gates	
Sustained by staring seraphim	
Where the souls of the devout	
Burn invisible and dim.	
 
Along the garden wall the bees
With hairy bellies pass between	
The staminate and pistilate,	
Blessed office of the epicene.	
 
Sweeney shifts from ham to ham	
Stirring the water in his bath.
The masters of the subtle schools	
Are controversial, polymath.""",
    "Eliot,  Sweeney Among the Nightingales": """Apeneck sweeney spread his knees
Letting his arms hang down to laugh,
The zebra stripes along his jaw
Swelling to maculate giraffe.

The circles of the stormy moon
Slide westward toward the river plate,
Death and the raven drift above
And sweeney guards the horned gate.

Gloomy orion and the dog
Are veiled; and hushed the shrunken seas;
The person in the spanish cape
Tries to sit on sweeney’s knees

Slips and pulls the table cloth
Overturns a coffee cup,
Reorganised upon the floor
She yawns and draws a stocking up;

The silent man in mocha brown
Sprawls at the window sill and gapes;
The waiter brings in oranges
Bananas figs and hothouse grapes;

The silent vertebrate in brown
Contracts and concentrates, withdraws;
Rachel née rabinovitch
Tears at the grapes with murderous paws;

She and the lady in the cape
Are suspect, thought to be in league;
Therefore the man with heavy eyes
Declines the gambit, shows fatigue,

Leaves the room and reappears
Outside the window, leaning in,
Branches of wistaria
Circumscribe a golden grin;

The host with someone indistinct
Converses at the door apart,
The nightingales are singing near
The convent of the sacred heart,

And sang within the bloody wood
When agamemnon cried aloud
And let their liquid siftings fall
To stain the stiff dishonoured shroud.""",
    "Eliot,  Sweeney Erect": """Paint me a cavernous waste shore	
  Cast in the unstilled cyclades,	
Paint me the bold anfractuous rocks	
  Faced by the snarled and yelping seas.	
 
Display me aeolus above
  Reviewing the insurgent gales	
Which tangle ariadne’s hair	
  And swell with haste the perjured sails.	
 
Morning stirs the feet and hands	
  (Nausicaa and polypheme),
Gesture of orangutan	
  Rises from the sheets in steam.	
 
This withered root of knots of hair	
  Slitted below and gashed with eyes,	
This oval O cropped out with teeth:
  The sickle motion from the thighs	
 
Jackknifes upward at the knees	
  Then straightens out from heel to hip	
Pushing the framework of the bed	
  And clawing at the pillow slip.
 
Sweeney addressed full length to shave	
  Broadbottomed, pink from nape to base,	
Knows the female temperament	
  And wipes the suds around his face.	
 
The lengthened shadow of a man
  Is history, said emerson	
Who had not seen the silhouette	
  Of sweeney straddled in the sun.	
 
Tests the razor on his leg	
  Waiting until the shriek subsides.
The epileptic on the bed	
  Curves backward, clutching at her sides.	
 
The ladies of the corridor	
  Find themselves involved, disgraced,	
Call witness to their principles
  And deprecate the lack of taste	
 
Observing that hysteria	
  Might easily be misunderstood;	
Mrs. turner intimates	
  It does the house no sort of good.
 
But doris, towelled from the bath,	
  Enters padding on broad feet,	
Bringing sal volatile	
  And a glass of brandy neat.""",
    "Coleridge,  Frost at Midnight": """The frost performs its secret ministry, 
Unhelped by any wind. The owlet's cry 
Came loud and hark, again! loud as before. 
The inmates of my cottage, all at rest, 
Have left me to that solitude, which suits 
Abstruser musings: save that at my side 
My cradled infant slumbers peacefully. 
It is calm indeed! so calm, that it disturbs 
And vexes meditation with its strange 
And extreme silentness. Sea, hill, and wood, 
This populous village! Sea, and hill, and wood, 
With all the numberless goings on of life, 
Inaudible as dreams! the thin blue flame 
Lies on my low burnt fire, and quivers not; 
Only that film, which fluttered on the grate, 

Still flutters there, the sole unquiet thing. 
Methinks, its motion in this hush of nature 
Gives it dim sympathies with me who live, 
Making it a companionable form, 
Whose puny flaps and freaks the idling spirit 
By its own moods interprets, every where 
Echo or mirror seeking of itself, 
And makes a toy of thought. 

                      But O! how oft, 
How oft, at school, with most believing mind, 
Presageful, have I gazed upon the bars, 
To watch that fluttering stranger ! and as oft 
With unclosed lids, already had I dreamt 
Of my sweet birth place, and the old church tower, 
Whose bells, the poor man's only music, rang 
From morn to evening, all the hot fair day, 
So sweetly, that they stirred and haunted me 
With a wild pleasure, falling on mine ear 
Most like articulate sounds of things to come! 
So gazed I, till the soothing things, I dreamt, 
Lulled me to sleep, and sleep prolonged my dreams! 
And so I brooded all the following morn, 
Awed by the stern preceptor's face, mine eye 
Fixed with mock study on my swimming book: 
Save if the door half opened, and I snatched 
A hasty glance, and still my heart leaped up, 
For still I hoped to see the stranger's face, 
Townsman, or aunt, or sister more beloved, 
My playmate when we both were clothed alike! 

         Dear babe, that sleepest cradled by my side, 
Whose gentle breathings, heard in this deep calm, 
Fill up the interspersed vacancies 
And momentary pauses of the thought! 
My babe so beautiful! it thrills my heart 
With tender gladness, thus to look at thee, 
And think that thou shalt learn far other lore, 
And in far other scenes! For I was reared 
In the great city, pent amid cloisters dim, 
And saw nought lovely but the sky and stars. 
But thou, my babe! shall wander like a breeze 
By lakes and sandy shores, beneath the crags 
Of ancient mountain, and beneath the clouds, 
Which image in their bulk both lakes and shores 
And mountain crags: so shalt thou see and hear 
The lovely shapes and sounds intelligible 
Of that eternal language, which thy god 
Utters, who from eternity doth teach 
Himself in all, and all things in himself. 
Great universal teacher! he shall mould 
Thy spirit, and by giving make it ask. 

         Therefore all seasons shall be sweet to thee, 
Whether the summer clothe the general earth 
With greenness, or the redbreast sit and sing 
Betwixt the tufts of snow on the bare branch 
Of mossy apple tree, while the night thatch 
Smokes in the sun thaw; whether the eave drops fall 
Heard only in the trances of the blast, 
Or if the secret ministry of frost 
Shall hang them up in silent icicles, 
Quietly shining to the quiet moon."""
  }
  if len(sys.argv) > 1:
    arg1 = sys.argv[1]
    if arg1.isdigit():
      arg1 = int(arg1)
      text_list = list(texts.keys())
      length = len(text_list)
      if arg1 >= length:
        print(f"ERROR: {arg1} is greater than length of text list ({length})")
      else:
        text = texts[text_list[arg1]]
        print(f"running predictor on text: {text_list[arg1]}")
        run_predictor(text, nsp_only=True)
    elif arg1.lower() == 'all':
      text_list = list(texts.keys())
      for txt in text_list:
        print(f"running predictor on text: {txt}")
        run_predictor(texts[txt], nsp_only=True)
        print()
    else:
      print(f"ERROR: {arg1} is not a valid text")
  else:
    print(f"ERROR: no text specified")

run_texts()   
    
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

import formatters
#import matplotlib.pyplot as plt

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
      "nsp_score": 0,
      "sentence_similarity": 0
    }

  def add_item(self, res):
    if res["mask_result"] == 'CORRECT':
      self.data["mask_correct"] += 1
    elif res["mask_result"] == 'INCORRECT':
      self.data["mask_incorrect"] += 1
      if res["mask_similarity"] != "Not Found":
        self.data["mask_incorrect_similarity"] += res["mask_similarity"]
    if res["generate_result"] == 'CORRECT':
      self.data["generate_correct"] += 1
    elif res["generate_result"] == "INCORRECT":
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
    print(f"\nCorrect Predictions   = {self.data['mask_correct']} {formatters.get_percent(self.data['mask_correct'], total)}")
    print(f"Incorrect Predictions = {self.data['mask_incorrect']} {formatters.get_percent(self.data['mask_incorrect'], total)}")
    print(f"Total Predictions     = {total}")
    #print(f"\nAverage Index         = {round(self.data['index_sum'] / total, 1)}")
    #print(f"Indexes Not Found     = {self.data['not_found']} {formatters.get_percent(self.data['not_found'], total)}")
    print(f"Average Similarity    = {round(self.data['mask_similarity'] / total, 2)}")
    if (self.data['mask_incorrect'] != 0):
      print(f"Incorrect Similarity  = {round(self.data['mask_incorrect_similarity'] / self.data['mask_incorrect'], 2)}\n")

    total = self.get_total("generate")
    print("Generative Word Results:")
    print(f"\nCorrect Predictions   = {self.data['generate_correct']} {formatters.get_percent(self.data['generate_correct'], total)}")
    print(f"Incorrect Predictions = {self.data['generate_incorrect']} {formatters.get_percent(self.data['generate_incorrect'], total)}")
    print(f"Total Predictions     = {total}")
    #print(f"\nAverage Index         = {round(self.data['index_sum'] / total, 1)}")
    #print(f"Indexes Not Found     = {self.data['not_found']} {formatters.get_percent(self.data['not_found'], total)}")
    print(f"Average Similarity    = {round(self.data['generate_similarity'] / total, 2)}")
    if (self.data['generate_incorrect'] != 0):
      print(f"Incorrect Similarity  = {round(self.data['generate_incorrect_similarity'] / self.data['generate_incorrect'], 2)}")

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
      #print(f"{formatters.pad_word(txt, 22)}= {self.data['categories'][i]} {formatters.get_percent(self.data['categories'][i], total)}")
    #div = categories_sum / total
    #print(f"\nAverage Category      = {round(div, 1)} (~{categories_txt[round(div) - 1]})")

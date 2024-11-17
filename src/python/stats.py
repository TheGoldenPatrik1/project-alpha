import formatters

class Stats:
  def __init__(self):
    self.data = {
      "mask_correct": 0,
      "mask_incorrect": 0,
      "mask_similarity": 0,
      "generate_correct": 0,
      "generate_incorrect": 0,
      "generate_similarity": 0,
      "mask_incorrect_similarity": 0,
      "generate_incorrect_similarity": 0,
      "nsp_score": 0,
      "sentence_similarity": 0,
      "is_top_10": 0
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
    if res["mask_similarity"] != "Not Found":
      self.data["mask_similarity"] += res["mask_similarity"]
    if res["generate_similarity"] != "Not Found":
      self.data["generate_similarity"] += res["generate_similarity"]
    if res["is_top_10"] != "UNKNOWN":
      self.data["is_top_10"] += 1
  
  def add_obj(self, res):
    self.data["mask_correct"] += res["mask_correct"]
    self.data["mask_incorrect"] += res["mask_incorrect"]
    self.data["mask_similarity"] += res["mask_similarity"]
    self.data["mask_incorrect_similarity"] += res["mask_incorrect_similarity"]
    self.data["generate_correct"] += res["generate_correct"]
    self.data["generate_incorrect"] += res["generate_incorrect"]
    self.data["generate_similarity"] += res["generate_similarity"]
    self.data["generate_incorrect_similarity"] += res["generate_incorrect_similarity"]
    self.data["sentence_similarity"] += res["sentence_similarity"]
    self.data["nsp_score"] += res["nsp_score"]
    self.data["is_top_10"] += res["is_top_10"]

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
    if total == 0:
      print("No Mask Word Results to Show")
    else:
      print("Mask Word Results:")
      print(f"\nCorrect Predictions   = {self.data['mask_correct']} {formatters.get_percent(self.data['mask_correct'], total)}")
      print(f"Incorrect Predictions = {self.data['mask_incorrect']} {formatters.get_percent(self.data['mask_incorrect'], total)}")
      print(f"Total Predictions     = {total}")
      print(f"Average Similarity    = {round(self.data['mask_similarity'] / total, 2)}")
      if (self.data['mask_incorrect'] != 0):
        print(f"Incorrect Similarity  = {round(self.data['mask_incorrect_similarity'] / self.data['mask_incorrect'], 2)}")
      print(f"Top 10 Predictions    = {self.data['is_top_10']} {formatters.get_percent(self.data['is_top_10'], total)}")

    total = self.get_total("generate")
    if total == 0:
      print("\nNo Generative Word Results to Show")
    else:
      print("\nGenerative Word Results:")
      print(f"\nCorrect Predictions   = {self.data['generate_correct']} {formatters.get_percent(self.data['generate_correct'], total)}")
      print(f"Incorrect Predictions = {self.data['generate_incorrect']} {formatters.get_percent(self.data['generate_incorrect'], total)}")
      print(f"Total Predictions     = {total}")
      print(f"Average Similarity    = {round(self.data['generate_similarity'] / total, 2)}")
      if (self.data['generate_incorrect'] != 0):
        print(f"Incorrect Similarity  = {round(self.data['generate_incorrect_similarity'] / self.data['generate_incorrect'], 2)}")
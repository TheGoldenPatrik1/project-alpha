import re
import time
import math
import json
import requests
import sys

start = time.time()

from predictor import Predictor

def arg_parse():
  args = {
    "use_tokenizer": False,
    "sentence_format": False,
    "ignore_proper": False,
    "nsp_only": False,
    "args": [],
    "book": False,
    "essay": False,
    "poem": False,
    "logs": False,
    "full_input": False
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
      elif "logs" in arg: args["logs"] = True
      elif "full" in arg: args["full_input"] = True
    else:
      args["args"].append(arg)
  return args

args = arg_parse()
arguments = args["args"]

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

  pred = Predictor(args)
  
  pred.run_predictor(book, data=selection)

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
      print("ERROR: no book specified")
    
def run_texts(content_type):
  pred = Predictor(args)
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
          pred.run_predictor(text)
      elif arg1 == 'all':
        text_list = list(texts.keys())
        for txt in text_list:
          print(f"running predictor on text: {txt}")
          pred.run_predictor(texts[txt])
          print()
      else:
        print(f"ERROR: {arg1} is not a valid text")
    else:
      print("ERROR: no text specified")

if args["book"]: run_books()
elif args["essay"]: run_texts("essays")
elif args["poem"]: run_texts("poems")
else: print("ERROR: no content specified to run program on. Full list of args:\n-token\n-sent\n-ignore / -proper\n-nsp\n-book\n-poem\n-essay\n-logs\n-full") 
    
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

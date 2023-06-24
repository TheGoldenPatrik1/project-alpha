import sys
import json

def get_file(file_name):
  with open(f'./files/{file_name}.txt') as f:
    data = json.load(f)
    return data

def set_file(file_name, data):
  f = open(f'./files/{file_name}.txt', 'w')
  data = json.dumps(data, indent=2)
  f.write(data)
  f.close()

def add_book():
  short_name = input("Enter the short name: ")
  short_name = short_name.lower().replace("-", "_")
  book_type = input("Enter the book type: ")
  url = input("Enter the url: ")
  title = input("Enter the title: ")
  author = input("Enter the author: ")
  publish = input("Enter the publish date: ")
  publish = int(publish)
  
  data = get_file('books')
  data[short_name] = {
    "url": url,
    "type": book_type,
    "title": title,
    "author": author,
    "publish": publish
  }
  
  set_file('books', data)
  print("Added new book!")

def add_text():
  text_type = input("Enter the text type (essays or poems): ")
  text_type = text_type.lower()
  text_key = input("Enter the text key: ")
  text_value = input("Enter the text value: ")
  
  data = get_file('texts')
  if text_type not in list(data.keys()):
    print(f"ERROR: text type '{text_type}' not found")
    return
  data[text_type][text_key] = text_value

  set_file('texts', data)
  print("Added new text!")

def edit_book():
  short_name = input("Enter the book's short name: ")
  short_name = short_name.lower()
  data = get_file('books')
  book_list = list(data.keys())
  
  if short_name in book_list:
    data_key = input("Enter the data key you would like to edit: ")
    data_key = data_key.lower()
    
    if data_key in ["url", "type", "title", "author", "publish"]:
      data_value = input(f"Enter the new value for '{data_key}': ")
      if data_key == "publish":
        data_value = int(data_value)
      data[short_name][data_key] = data_value
      set_file('books', data)
      print(f"Successfully edited {short_name}!")
    elif "short" in data_key:
      data_value = input(f"Enter the new short name for '{short_name}': ")
      data_value = data_value.strip().lower().replace("-", "_")
      data[data_value] = data.pop(short_name)
      set_file('books', data)
      print(f"Successfully edited the short name to '{data_value}'!")
    else:
      print(f"ERROR: invalid data key '{data_key}'")
      
  else:
    print(f"ERROR: book with short_name '{short_name}' not found")

def remove_book():
  short_name = input("Enter the book's short name: ")
  short_name = short_name.lower()
  data = get_file('books')
  book_list = list(data.keys())
  
  if short_name in book_list:
    del data[short_name]
    set_file('books', data)   
    print(f"Successfully removed {short_name}!")
  else:
    print(f"ERROR: book with short_name '{short_name}' not found")

def remove_text():
  text_type = input("Enter the text type (essays or poems): ")
  text_type = text_type.lower()
  text_key = input("Enter the text key: ")

  data = get_file('texts')
  if text_type not in list(data.keys()):
    print(f"ERROR: text type '{text_type}' not found")
    return
  text_list = list(data[text_type].keys())
  
  if text_key in text_list:
    del data[text_type][text_key]
    set_file('texts', data)
    print("Removed new text!")
  else:
    print(f"ERROR: text with key '{text_key}' not found")

def list_books():
  data = get_file('books')
  book_list = list(data.keys())
  print(f"There are currently {len(book_list)} books:")
  for book in book_list:
    print(f"{data[book]['title']} by {data[book]['author']} [{book}]")

def list_texts():
  data = get_file('texts')
  type_list = list(data.keys())
  for k in type_list:
    spec_list = list(data[k].keys())
    print(f"There are currently {len(spec_list)} {k}:")
    for item in spec_list:
      print(f"- {item}")
    print()

validArgString = "Valid arguments: add, edit, remove, OR list"
file_name = ""
action_name = ""

if len(sys.argv) == 1:
  file_name = input("Please specify whether you'd like to access books or texts: ")
  file_name = file_name.lower()
else:
  file_name = sys.argv[1].lower()

while file_name != "books" and file_name != "texts":
  file_name = input("Please specify whether you'd like to access books or texts: ")
  file_name = file_name.lower()
  
if len(sys.argv) == 2:
  action_name = input(f"{validArgString}\nPlease specify one: ")
  action_name = action_name.lower()
else:
  action_name = sys.argv[2].lower()
  
if action_name == "add":
  if file_name == 'books': add_book() 
  else: add_text()
elif action_name == "edit":
  if file_name == 'books': edit_book()
  else: print("Editing a text is not supported at this time")
elif action_name == "remove":
  if file_name == 'books': remove_book()
  else: remove_text()
elif action_name == "list":
  if file_name == 'books': list_books()
  else: list_texts()
else:
  print(f"ERROR: invalid argument\n{validArgString}")

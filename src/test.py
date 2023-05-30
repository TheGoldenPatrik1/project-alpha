import sys
import json

def get_books():
  with open('books.txt') as f:
    data = json.load(f)
    return data

def set_books(data):
  f = open('books.txt', 'w')
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
  
  data = get_books()
  data[short_name] = {
    "url": url,
    "type": book_type,
    "title": title,
    "author": author,
    "publish": publish
  }
  
  set_books(data)
  print("Added new book!")

def edit_book():
  short_name = input("Enter the book's short name: ")
  short_name = short_name.lower()
  data = get_books()
  book_list = list(data.keys())
  
  if short_name in book_list:
    data_key = input("Enter the data key you would like to edit: ")
    data_key = data_key.lower()
    
    if data_key in ["url", "type", "title", "author", "publish"]:
      data_value = input(f"Enter the new value for '{data_key}': ")
      if data_key == "publish":
        data_value = int(data_value)
      data[short_name][data_key] = data_value
      set_books(data)
      print(f"Successfully edited {short_name}!")
    else if "short" in data_key:
      data_value = input(f"Enter the new short name for '{short_name}': ")
      data_value = data_value.strip().lower().replace("-", "_")
      data[data_value] = data.pop(short_name)
      set_books(data)
      rint(f"Successfully edited the short name to '{data_value}'!")
    else:
      print(f"ERROR: invalid data key '{data_key}'")
      
  else:
    print(f"ERROR: book with short_name '{short_name}' not found")

def remove_book():
  short_name = input("Enter the book's short name: ")
  short_name = short_name.lower()
  data = get_books()
  book_list = list(data.keys())
  
  if short_name in book_list:
    del data[short_name]
    set_books(data)   
    print(f"Successfully removed {short_name}!")
  else:
    print(f"ERROR: book with short_name '{short_name}' not found")

def list_books():
  data = get_books()
  book_list = list(data.keys())
  print(f"There are currently {len(book_list)} books:")
  for book in book_list:
    print(f"{data[book]['title']} by {data[book]['author']} [{book}]")

validArgString = "Valid arguments: add, edit, remove, OR list"
arg1 = ""

if len(sys.argv) == 1:
  arg1 = input(f"{validArgString}\nPlease specify one: ")
  arg1 = arg1.lower()
else:
  arg1 = sys.argv[1].lower()

if arg1 == "add":
  add_book()
elif arg1 == "edit":
  edit_book()
elif arg1 == "remove":
  remove_book()
elif arg1 == "list":
  list_books()
else:
  print(f"ERROR: invalid argument\n{validArgString}")

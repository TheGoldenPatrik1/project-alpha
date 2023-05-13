import sys
import json

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
        print(f"getting data for {book['title']}")
    else:
      arg1 = arg1.lower()
      if arg1 in book_list:
        book = books[arg1]
        print(f"getting data for {book['title']}")
      else:
        print(f"ERROR: {arg1} is not in book list")

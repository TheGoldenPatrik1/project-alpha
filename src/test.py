import sys
import json

def add_book():
  short_name = input("Enter the short name: ")
  print(short_name)

def edit_book():
  return

def remove_book():
  return

validArgString = "Valid arguments: add, edit, OR remove"
if len(sys.argv) == 1:
  print(f"ERROR: no arguments specified\n{validArgString}")
else:
  arg1 = sys.argv[1].lower()
  if arg1 == "add":
    add_book()
  elif arg1 == "edit":
    edit_book()
  elif arg1 == "remove":
    remove_book()
  else:
    print(f"ERROR: invalid argument\n{validArgString}")

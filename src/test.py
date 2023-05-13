import sys

print(len(sys.argv))
if len(sys.argv) > 1:
  arg1 = sys.argv[1]
  print(arg1)
  if arg1 == "yes":
    print("correct arg1")

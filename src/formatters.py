def pad_word(input_str, length):
  input_str = f"{input_str}"
  for x in range(length - len(input_str)):
    input_str += ' '
  return input_str

def get_percent(part, whole):
  if part == 0:
    return ""
  else:
    return f"({round((part / whole) * 100, 1)}%)"

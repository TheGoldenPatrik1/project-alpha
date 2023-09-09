def print_sep():
  print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

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

def print_word(
    masked_word="Masked Word",
    mask_predicted_word="Mask Predicted Word",
    mask_prediction_result="Mask Prediction Result",
    #correct_index="Index of Correct Word",
    mask_similarity="Mask Similarity",
    #top_predictions="Next Three Predictions",
    #prediction_category="Category",
    generate_predicted_word="Generative Predicted Word",
    generate_prediction_result="Generative Prediction Result",
    generate_similarity="Generative Similarity",
    stop_word="Stop Word"
):
  print(f"| {pad_word(masked_word, 16)} ", end = '')
  print(f"| {pad_word(mask_predicted_word, 21)} ", end = '')
  print(f"| {pad_word(mask_prediction_result, 22)} ", end = '')
  #print(f"| {pad_word(correct_index, 21)} ", end = '')
  print(f"| {pad_word(mask_similarity, 15)} ", end = '')
  #print(f"| {pad_word(top_predictions, 36)} ", end = '')
  #print(f"| {pad_word(prediction_category, 8)} ", end = '')
  print(f"| {pad_word(generate_predicted_word, 25)} ", end='')
  print(f"| {pad_word(generate_prediction_result, 28)} ", end='')
  print(f"| {pad_word(generate_similarity, 21)} ", end='')
  print(f"| {pad_word(stop_word, 9)} |")
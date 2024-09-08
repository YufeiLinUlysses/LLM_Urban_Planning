import random
def shuffle_answer(question, ans, explanation="undefined"): 
    # Initial data
    input_question = question
    correct_answer = ans

    # Split the question and options
    question_part = input_question.split('(A)')[0]
    options_part = input_question.split('(A)')[1]

    # Extract options
    options = options_part.split(' (')
    options = ['(A)' + options[0], '(' + options[1], '(' + options[2], '(' + options[3]]

    # Extract option texts
    option_texts = [opt[3:].strip() for opt in options]

    # Find the correct answer's text
    correct_text = option_texts[ord(correct_answer) - ord('A')]

    # Shuffle the option texts
    shuffled_texts = option_texts[:]
    random.shuffle(shuffled_texts)

    # Find the new correct answer index
    new_correct_index = shuffled_texts.index(correct_text)

    # Create the new options string with shuffled texts but ordered labels
    shuffled_options = [f'({chr(ord("A") + i)}){shuffled_texts[i]}' for i in range(4)]

    # Create the new question string
    shuffled_question = question_part + ' '.join(shuffled_options)

    # Output
    output = {
        "Input": shuffled_question,
        "Output": chr(ord('A') + new_correct_index),
        "Explanation": explanation
    }

    return output

# print(shuffle_answer(question="What is true for the data type of the column Location in the dataset? (A)It is boolean (B)It is integer (C)It is datetime (D)It is string",ans="D"))
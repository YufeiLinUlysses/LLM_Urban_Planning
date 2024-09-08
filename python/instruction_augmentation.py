from shuffle_answer import shuffle_answer
from qa_pair_gen import paraphrase,generate_false_output, Paraphrase
import json
import random
import copy

d_i = 4
datasets = [
    {
        "file":"ngsim_instructions",
        "website": "https://datahub.transportation.gov/stories/s/Next-Generation-Simulation-NGSIM-Open-Data/i5zb-xe34/"
    },
    {
        "file":"highD_instructions",
        "website": "https://levelxdata.com/highd-dataset/"
    },
    {
        "file":"road_networks_instructions",
        "website": "https://networkrepository.com/road.php"
    },
    {
        "file":"tlc_trip_instructions",
        "website": "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
    },
    {
        "file":"urban_flow_prediction_survey_instructions",
        "website": "https://www.sciencedirect.com/science/article/abs/pii/S1566253519303094?casa_token=h2CrTzVjno0AAAAA:LGA_QAyz6fDBc1hp5aOeMLrrqIt0hN5Ob18LQK4vZpXM_XJ-5QfYKQUMb9KEStOJ4WrdiVOt8JWx"
    }
]


def generate_characters(exclude):
    # Define the list of characters
    characters = ['A', 'B', 'C', 'D']
    
    # Exclude the specified character
    if exclude in characters:
        characters.remove(exclude)
    
    # Shuffle the remaining characters
    random.shuffle(characters)
    
    return characters[:2]
f_name = datasets[d_i]["file"]
instructions = json.load(open('./instructions/'+f_name+'.json', 'r'))
p = Paraphrase(website=datasets[d_i]["website"])
# print(instructions[0]["Positive Example"])
# Update multiple selection example
newpositive = []
newfalse = []
for example in instructions[0]["Positive Example"]:
    question = example["Input"].split("(A)")[0]
    output = example["Output"]
    # generate false examples
    for c in generate_characters(output):
        false_example = copy.deepcopy(example)
        false_example["Output"] = c
        false_example["Explanation"] = "The correct answer is: "+ example["Output"] + ". " + p.paraphrase_explanation(example=str(example))
        newfalse.append(false_example)
    # replace question to generate new example
    for i in range(10):
        newExample = copy.deepcopy(example)
        newExample["Input"] = paraphrase(question) + " (A)" + example["Input"].split("(A)")[1]
        newExample["Explanation"] = p.paraphrase_explanation(example=str(example))
        newpositive.append(newExample)
        for c in generate_characters(newExample["Output"]):
            false_example = copy.deepcopy(example)
            false_example["Output"] = c
            newfalse.append(false_example)
        # shuffle answer to generate new question pair
        for i in range(2):
            generated_example = shuffle_answer(question=newExample["Input"], ans=newExample["Output"],explanation=newExample["Explanation"])
            newExample["Explanation"] = p.paraphrase_explanation(example=str(example))
            newpositive.append(generated_example)
            for c in generate_characters(generated_example["Output"]):
                false_example = copy.deepcopy(generated_example)
                false_example["Output"] = c
                false_example["Explanation"] = "The correct answer is: "+ generated_example["Output"] + ". " + p.paraphrase_explanation(example=str(generated_example))
                newfalse.append(false_example)
            
# print(len(newpositive))
# print(len(newfalse))
instructions[0]["Positive Example"] += newpositive
instructions[0]["Negative Example"] += newfalse

# Update question and answer example
newpositive = []
newfalse = []
for example in instructions[1]["Positive Example"]:
    print(example)
    seperated_example = example["Input"].split("Question:")
    question = seperated_example[1]
    fact = seperated_example[0].strip()[6:]

    # update question
    for i in range(10):
        newExample = copy.deepcopy(example)
        newExample["Input"] = seperated_example[0] + " Question: " + paraphrase(question)
        newExample["Explanation"] = p.paraphrase_explanation(example=str(example))
        newpositive.append(newExample)

        # generate false output
        for i in range(2):
            false_example = copy.deepcopy(newExample)
            false_example["Output"] = generate_false_output(false_example["Output"])
            false_example["Explanation"] = "The correct answer is: "+ example["Output"] + ". " + p.paraphrase_explanation(example=str(example))
            newfalse.append(false_example)
    
    # update fact
    for i in range(10):
        newExample = copy.deepcopy(example)
        newExample["Input"] = "Fact: " + p.paraphrase_sentence(fact) + " Question: " + question
        newExample["Explanation"] = p.paraphrase_explanation(example=str(example))
        newpositive.append(newExample)
        for i in range(2):
            false_example = copy.deepcopy(newExample)
            false_example["Output"] = generate_false_output(false_example["Output"])
            false_example["Explanation"] = "The correct answer is: "+ newExample["Output"] + ". " + p.paraphrase_explanation(example=str(newExample))
            newfalse.append(false_example)

instructions[1]["Positive Example"] += newpositive
instructions[1]["Negative Example"] += newfalse

with open("./instructions/"+f_name+"_generated.json", 'w') as file:
    json.dump(instructions, file, indent=4)
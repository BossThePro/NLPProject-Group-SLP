import pandas as pd
import random

def read_iob2_file_wPOS(path):
    """
    Read provided Universal NER iob2 file
    
    :param path: path to read from
    :returns: list with sequences of words, NER labels and POS tags for each sentence
    """
    data = []
    current_words = []
    current_ner_tags = []
    current_pos_tag = []

    for line in open(path, encoding='utf-8'):
        line = line.strip()

        if line:
            if line[0] == '#':
                continue # skip comments
            tok = line.split(' ')
            #print(tok)
            current_words.append(tok[1])
            current_ner_tags.append(tok[2])
            current_pos_tag.append(tok[3])
        else:
            if current_words:  # skip empty lines
                data.append((current_words, current_ner_tags,current_pos_tag))
            current_words = []
            current_ner_tags = []
            current_pos_tag = []

    # check for last one
    if current_ner_tags != []:
        data.append((current_words, current_ner_tags,current_pos_tag))
    return data

female_dominated_dict = {
    "he" : "she",
    "his" : "hers",
    "him" : "her",
    "himself" : "herself",
    "they" : "she",
    "them" : "her",
    "theirs" : "his"
    }


male_dominated_dict = {
    "she" : "he",
    "hers" : "his",
    "her" : "him",
    "herself" : "himself",
    "they" : "he",
    "them" : "him",
    "theirs" : "his"
    }

gender_neutral_dict = {
    "he" : "they",
    "she" : "they",
    "him" : "them",
    "hers" : "them",
    "his" : "theirs",
    "her" : "theirs",
    "himself" : "themself",
    "herself" : "themself"
    }

def total_pronoun_swap(data, transform_dict):
    """Swaps the pronouns in a given conll file parsed using the read_iob2_file_wPOS function 
    Parameters
    ----------
    data : list 
        Pass the data returned from the parser function, should be a list ex. [([words],[NER tags],[POS tags]),.....]
    
    transform_dict : dict
        a dict which maps the pronouns that you would like to transfrom some helpful examples are provided above such as \
        gender_neutral_dict which trasfroms all pronouns into gender neutral terms
    Returns
    -------
    swapped_data : list 
        A list in the same format as the input, however with the pronous replaced based on the provided dict 
    Notes
    -----
    """
    change = 0
    for instance in data:
        for i in range(len(instance[2])):
            if (instance[2][i] == "PRP" or instance[2][i] == "PRP$"):
                  word_to_swap = instance[0][i]
                  swap_to = transform_dict.get(word_to_swap.casefold(),word_to_swap)
                  instance[0][i] = swap_to
                  change += 1

    print(f"Changed {change} tokens")
    return data



### CHANGING GENDER ###

def create_gender_name_dict(data,path="../data/gender_names/top_female_names.csv"):
    """Create a dictionary where each first name in the original dataset (key) gets a random value from a csv file \
        with names from only one gender, defaulted to a file with all female names
    ----------
    data : list 
        Pass the data returned from the parser function, should be a list ex. [([words],[NER tags],[POS tags]),.....]\
        here POS tags are optional 
    
    path : str
        the path to a csv file which includes the replacement names, default is "../data/gender_names/top_female_names.csv" \
        to get all male names change "female" to "male" in the file path
    Returns
    -------
    name_dict : dict
        A dict which for each name in the original dataset gets a random new name from the provided csv file
    Notes
    -----
    Currently you can only have unique keys but not values, since our dataset has about 1600 B-PER, but the csv files only\
    have 1000 names so there will be distinct key that will have the same value

    The code does not create keys for I-PERs, since these are mostly middle names or last names, and traditioally those \
    aren't gender specific (or highly dominated by male names), so our focus becomes first names
    """

    name_dict = {}
    df = pd.read_csv(path)
    name_index = 0

    if path == "../data/gender_names/top_female_names.csv":
        replacement_pool = df["Female Names"].tolist()
    else:
        replacement_pool = df["Male Names"].tolist()

    random.shuffle(replacement_pool)

    for instance in data:

        for i in range(len(instance[1])):

            if instance[1][i] == "B-PER":
                first_name = instance[0][i]

                if first_name not in name_dict.keys():
                    name_dict[first_name] = replacement_pool[name_index]
                    name_index += 1
                    
    return name_dict
                
                
def gender_name_swap(data,name_dict):
    """Based on the name_dict provided changes all the instances of B-PER using in the data using the name_dict
    ----------
    data : list 
        Pass the data returned from the parser function, should be a list ex. [([words],[NER tags],[POS tags]),.....]\
        here POS tags are optional 
    
    name_dict : dict
        a dict returned from the create_gender_name_dict function, that has all the original names as keys and new \
        names as values
    Returns
    -------
    data : list 
        A list in the same format as the input, however with the names replaced based on the provided dict 
    Notes
    -----
    """
    
    for instance in data:

        for i in range(len(instance[1])):
            if instance[1][i] == "B-PER":
                instance[0][i] = name_dict[instance[0][i]]
    
    return data

def recomplie_data(data,file_path):
    """Turns the swapped_data back into an iob2 file
    ----------
    data : list 
        The data returned from any of the above swappers, as list of lists
    
    file_path : str
        file path to where you want to save the swapped data
    """
    with open (file_path, "w") as F:
        for line in data:
            for i in range(len(line[0])):
                F.write(f"{i+1} {line[0][i]} {line[1][i]} {line[2][i]}\n")
            F.write("\n")


if __name__ == "__main__":
    
    ### FOR THE PRONOUNS TEST SETS ###
    data = read_iob2_file_wPOS("../data/train_conll.iob2")
    neutral_pronouns_swapped = total_pronoun_swap(data,gender_neutral_dict)
    female_pronouns_swapped = total_pronoun_swap(data,female_dominated_dict)
    male_pronouns_swapped = total_pronoun_swap(data,male_dominated_dict)
    recomplie_data(neutral_pronouns_swapped,"../data/pronouns/neutral_pronouns_test.iob2")
    recomplie_data(female_pronouns_swapped,"../data/pronouns/female_pronouns_test.iob2")
    recomplie_data(male_pronouns_swapped,"../data/pronouns/male_pronouns_test.iob2")

    ### FOR THE GENDERED NAMES TEST SETS ###
    female_names_dict = create_gender_name_dict(data,path="../data/gender_names/top_female_names.csv")
    male_names_dict = create_gender_name_dict(data,path="../data/gender_names/top_male_names.csv")
    female_names= gender_name_swap(data,female_names_dict)
    male_names = gender_name_swap(data,male_names_dict)
    recomplie_data(female_names,"../data/gender_names/female_names_test.iob2")
    recomplie_data(male_names,"../data/gender_names/male_names_test.iob2")
    

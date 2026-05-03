import re
import string
import pandas as pd 
from collections import defaultdict
from collections import Counter
import random
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""
This file contains the code for grabbing, swapping and saving swapped entity names within the test data.
"""


def read_file(file: str):
    """Reads the file as input for further use later. In order for this function to work in later code, it expects a specific line formatting such as the file found in data/test.txt.
    Parameters
    ----------
    file : str
        The path (either relative or absolute) of the input file.
    Returns
    -------
    line_list : list 
        a list consisting of each line in the input file
    """
    line_list = []
    ### Opens file
    with open(file, "r") as input:
        ### Appends each line to list
        for line in input:
            line_list.append(line)
        input.close()
    ### Returns our list consisting of each line in the input
    return line_list 



def process_line(line_list : list, ignore_bio: bool=False):
    """Processes each line in line_list to prepare it for entity swapping.
    Parameters
    ----------
    line_list : list 
        List containing each line of input.
    ignore_bio : bool
        Boolean to ignore BIO part of tagging, hence B-PER would become PER if enabled, and I-PER would also become PER. 
    Returns
    -------
    entity_dict : dict 
        Dictionary containing key (tag) and values (entity) pairs. This only keeps B tags, as they are the only important tags to count here.
    """
    ### Little helper comment here: line[0] is the word number in the sentence, so e.g. "I had a good day today", here "good" would be equal to 4, since its the fourth word 
    ### line[1] is equal to the actual entity name, so this would be equal to "good" in the previous sentence
    ### line[2] is the associated NER tag, so for the word "good" it would be O.
    ### line[3] is the associated POS tag, but is not used in this part of the code, as this part looks for entities, and not pronouns.
    entity_tag_dict = defaultdict(list)
    for line in line_list:

        line = line.split() 
        if not line:
            continue
        ### This is probably not needed either due to new way of processing. I will keep it in case we need to remove BIO part of tagging at any point though.
        if ignore_bio:
            line[2] = re.sub(r"\w+[-]", "" ,line[2])
        tag_start = re.search(r"^B", line[2])
        ### TODO: dict of key (tag) and value (entity) pairs 
        if tag_start:
            entity_tag_dict[str(line[2])].append(str(line[1]).lower())
    return entity_tag_dict 


def entity_count(entity_tag_dict : defaultdict[str, list[str]]):
    """Counts the amount of entities of different (relevant) tags. Also counts the amount of unique entities of different (relevant) tags.
    Parameters
    ----------
    entity_tag_dict : dict[str, str]
        Dictionary containing unique entities of each tag type (B-PER, B-ORG, B-LOC, B-MISC)
    Prints
    -------
    str 
        Counts of each tag (B-PER, B-ORG, B-LOC, B-MISC), both unique and total.
    Returns
    -------
    unique_dict : dict 
        Dictionary containing keys (B-PER, B-ORG, B-LOC, B-MISC) and values (unique number of entities)
    """
    unique_dict = {}
    for key in entity_tag_dict.keys():
        print(f"{str(key)} contains {len(entity_tag_dict[key])} values")
        print(f"{str(key)} contains {len(set(entity_tag_dict[key]))} unique values")
        unique_dict[str(key)] = len(set(entity_tag_dict[key]))
    return unique_dict


def export_swaps(swapped_list : list, file_name : str):
    """Exports the swaps performed in one of the swap functions to an iob2 file.
    Parameters 
    ----------
    swapped_list : list
        The new list containing the new sentences with the swapped entities 
    file_name : str
        The name of the exported file   
    """
    with open(file_name, "w") as f:
        for line in swapped_list:
            if line == "\n":
                f.write("\n")
            else:
                f.write(line + "\n")

### PERSON SPECIFIC FUNCTIONS

def count_span_lengths_person_tags(line_list: list):
    """Counts the distribution of PER span lengths (B-PER + subsequent I-PERs).
    Parameters
    ----------
    line_list : list
        List containing each line of input.
    Returns
    -------
    Counter
        Keys are span lengths, values are how many spans of that length exist.
        e.g. Counter({2: 4521, 1: 1203, 3: 47}) means 4521 spans of length 2, etc.
    """
    span_lengths = []
    i = 0
    lines = [line.split() for line in line_list if line.split() and "#" not in line.split()[0]]

    while i < len(lines):
        if lines[i][2] == "B-PER":
            length = 1
            j = i + 1
            while j < len(lines) and lines[j][2] == "I-PER":
                length += 1
                j += 1
            span_lengths.append(length)
            i = j
        else:
            i += 1

    return Counter(span_lengths)



def name_reader(file_path: str):
    """Reads files with names 
    Parameters
    ----------
    file_path : str 
        The path of the csv file with names. 
    Returns
    -------
    name_list : list 
        List of names to be used in further processing.
    """
    name_list = pd.read_csv(file_path)
    return list(name_list["Name"])


def person_swap(entity_tag_dict: defaultdict, replacement_first_names: list, replacement_last_names: list):
    """Creates the mapping for swapping PER names within the dataset. 
    Parameters
    ----------
    entity_tag_dict : defaultdict
        Dictionary containing unique entities of each tag type (B-PER, B-ORG, B-LOC, B-MISC)
    replacement_first_names : list 
        List of first names that will replace original last names. NOTE: here any B-PER is considered a first name, although that will not always be the case. 
    replacement_last_names : list 
        List of last names that will replace original last names. Always considered as the last I-PER tag in a B-PER/I-PER sequence.
    Returns
    -------
    person_swaps : dict 
        Dictionary mapping for new replacement first names, based on first name of old names. e.g. "John" -> "Mohammed". 
    shuffled_middle_names : list 
        List containing middle names to be used in swap_function_entities 
    shuffled_last_names : list 
        List containing last names to be used in swap_function_entities
    Notes
    -----
    Justification for replacement_first_names
        A) first names and last names are typically similar, so we can still capture regional differences (in some places first names can even be last names). B) Good NER models should be able to find entitites no matter if its the first name or last name given, so it is still relevant to our research. C) Most cases of B-PER in the dataset does appear to be first names.
    """
    person_swaps = {}
    per_names = entity_tag_dict.get("B-PER", [])
    unique_names = list(set(per_names))
    
    shuffled_first_names = random.sample(replacement_first_names, len(unique_names))
    ### Due to middle names typically just being another first name, we can simply randomly sample these. In the conll test dataset the longest span is 4 so with this we dont need to worry about edge cases. (full names with more than 4 total names)
    shuffled_middle_names = random.sample(replacement_first_names, len(unique_names))
    ### Since we potentially have less last names than first names, we randomly sample with replacement. This also matches typical naming across the world, where many people share the same last name
    shuffled_last_names = random.choices(replacement_last_names, k=len(unique_names))
    
    for original, first, in zip(unique_names, shuffled_first_names):
        person_swaps[original] = {
            "first": first
        }
    
    return person_swaps, shuffled_middle_names, shuffled_last_names 



def swap_person_entities(line_list: list, person_swaps: dict, shuffled_middle_names : list, shuffled_last_names : list):
    """Swaps B-PER and I-PER entities from old names into new names
    Parameters
    ----------
    line_list : list 
        List of lines gathered from the read_file function.
    person_swaps : dict 
        Dictionary containing mappings from old names to new names, gathered from the person_swap function.
    shuffled_middle_names : list 
        List of middle names to be used to create random middle names in this function (such that all Johns do not become "Muhammed [same middle name] [same last name]")
    shuffled_last_names : list 
        List of last names to be used to create random last names in this function (such that all Johns do not become "Muhammed [same middle name] [same last name]")
    Returns
    -------
    swapped_list: list 
        A list containing each sentence of the entire conll test dataset, with the names replaced with new names where appropriate. 
    """
    swapped_list = []
    lines = [line.split() for line in line_list]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        ### Empty line (new sentence)
        if not line:
            swapped_list.append("\n")
            i += 1
            continue
        ### Check for total span length of section, in order to know how many of the replacements are needed. 
        if line[2] == "B-PER":
            span_length = 1
            j = i + 1
            ### Check that we havent reached the end of file, end of sentence and that next word is still I-PER
            while j < len(lines) and lines[j] and lines[j][2] == "I-PER":
                span_length += 1
                j += 1
            
            original_name = line[1].lower()
            mapping = person_swaps.get(original_name, None)
            ### If there exists a mapping in the dictionary, do the following: 
            if mapping:
                ### Replace B-PER with first name
                line[1] = mapping["first"]
                swapped_list.append(" ".join(line))
                
                ### Now fill in I-PERs based on span length
                for k, iper_idx in enumerate(range(i + 1, i + span_length)):
                    iper_line = lines[iper_idx]
                    if span_length == 2:
                        iper_line[1] = random.choice(shuffled_last_names)
                    elif span_length == 3:
                        iper_line[1] = random.choice(shuffled_middle_names) if k == 0 else random.choice(shuffled_last_names) 
                    elif span_length >= 4:
                        iper_line[1] = random.choice(shuffled_middle_names) if k == 0 else random.choice(shuffled_middle_names) if k == 1 else random.choice(shuffled_last_names)
                    swapped_list.append(" ".join(iper_line))
                
                i = i + span_length
            else:
                swapped_list.append(" ".join(line))
                i += 1
        else:
            swapped_list.append(" ".join(line))
            i += 1
    
    return swapped_list 




### SPECIFIC FOR RANDOM STRING GENERATION


def length_distribution_names(line_list: list):
    """Gathers the distribution of names in each category, to be used to generate random strings of similar lengths.
    Parameters
    ----------
    line_list : list
        List of lines gathered from the read_file function 
    Returns
    -------
    distribution_dict : defaultdict(lambda: defaultdict(int)) 
        Dictionary containing the distributions of the different types, split by tag type (PER, ORG, LOC, MISC) (B-I part of tagging is ignored here)
    """

    distribution_dict = defaultdict(lambda: defaultdict(int))

    for line in line_list:

        line = line.split() 
        if not line:
            continue

        if line[2] == "B-PER" or line[2] == "I-PER":
            distribution_dict["PER"][len(line[1])] += 1 
        elif line[2] == "B-ORG" or line[2] == "I-ORG":
            distribution_dict["ORG"][len(line[1])] += 1 
        elif line[2] == "B-LOC" or line[2] == "I-LOC":
            distribution_dict["LOC"][len(line[1])] += 1 
        elif line[2] == "B-MISC" or line[2] == "I-MISC":
            distribution_dict["MISC"][len(line[1])] += 1  
        else:
            continue

    return distribution_dict




def random_string_generation_swap(line_list: list, distribution_dict: dict):
    """Generates random strings based on distributions, and swaps the names of PER, LOC, ORG and MISC entities. 
    Parameters
    ----------
    line_list : list 
        List of lines gathered from the read_file function 
    distribution_dict : defaultdict(lambda: defaultdict(int))
        Dictionary containing the distributions of the different entity types (PER, LOC, ORG, MISC) gathered from the length_distribution_names function.
    Returns
    -------
    swapped_list : list 
        A list containing each sentence of the entire conll test dataset, with the entity names replaced with random strings. 
    Notes
    -----
    Currently generates strings with all possible ASCII characters, can be changed if needed
    """
    swapped_list = []

    per_lengths = list(distribution_dict["PER"].keys())
    org_lengths = list(distribution_dict["ORG"].keys())
    loc_lengths = list(distribution_dict["LOC"].keys())
    misc_lengths = list(distribution_dict["MISC"].keys())

    per_counts = list(distribution_dict["PER"].values())
    org_counts = list(distribution_dict["ORG"].values())
    loc_counts= list(distribution_dict["LOC"].values())
    misc_counts = list(distribution_dict["MISC"].values())
    for line in line_list:

        line = line.split()

        if not line:
            swapped_list.append("\n")
            continue

        if line[2] == "B-PER" or line[2] == "I-PER":
            word_len = random.choices(per_lengths, weights=per_counts, k=1)[0]
            line[1] = "".join(random.choices(string.printable, k=int(word_len)))
            swapped_list.append(" ".join(line))

        elif line[2] == "B-ORG" or line[2] == "I-ORG":
            word_len = random.choices(org_lengths, weights=org_counts, k=1)[0]
            line[1] = "".join(random.choices(string.printable, k=int(word_len)))
            swapped_list.append(" ".join(line))

        elif line[2] == "B-LOC" or line[2] == "I-LOC":
            word_len = random.choices(loc_lengths, weights=loc_counts, k=1)[0]
            line[1] = "".join(random.choices(string.printable, k=int(word_len)))
            swapped_list.append(" ".join(line))

        elif line[2] == "B-MISC" or line[2] == "I-MISC":
            word_len = random.choices(misc_lengths, weights=misc_counts, k=1)[0]
            line[1] = "".join(random.choices(string.printable, k=int(word_len)))
            swapped_list.append(" ".join(line))
            
        else:
            swapped_list.append(" ".join(line))
            continue
    return swapped_list

### SPECIFIC FOR EXONYMS VS ENDONYMS
def loc_grabber(line_list: list, file_name: str):
    """Grabs the B-LOC and I-LOC names and puts them into a file of their own for annotations.
    Parameters
    ----------
    line_list : list 
        List of lines gathered from the read_file function 
    file_name : str 
        Path of the file to be exported
    """
    lines = set() 
    i = 0
    split_lines = [line.split() for line in line_list]

    while i < len(split_lines):
        line = split_lines[i]
        if not line:
            i += 1
            continue
        if line[2] == "B-LOC":
            full_name = [line[1]]
            j = i + 1
            while j < len(split_lines) and split_lines[j] and split_lines[j][2] == "I-LOC":
                full_name.append(split_lines[j][1])
                j += 1
            lines.add(" ".join(full_name))
            i = j
        else:
            i += 1
    with open(file_name, "w") as f:
        for line in lines:
                f.write("".join(line) + "\n")


def group_sentences(line_list):
    """Groups sentences as lists of token rows [position, word, ner_tag, pos_tag] for use in replace_locs_in_sentence to find spans.
    Parameters
    ----------
    line_list : list 
        List of lines gathered from the read_file function
    Returns
    -------
    sentences : list 
        List of sentences containing each word in list format: [position, word, ner_tag, pos_tag]
    """
    sentences = []
    sentence = []
    for line in line_list:
        parts = line.split()
        if not parts:
            if sentence:
                sentences.append(sentence)
            sentence = []
        else:
            sentence.append(parts)
    if sentence:
        sentences.append(sentence)
    return sentences


def build_lookup(file_name: str):
    """ Reads the file containing (exonym, endonym, latinization) and creates a lookup dictionary for use in replace_locs_in_sentence function
    Parameters
    ----------
    file_name : str 
        Path of the file containing the exonyms, endonyms and latinizations of the locations
    Returns
    -------
    lookup : dict 
        Dictionary consisting of key: exonym and value: (endonym, latinization) for use in replace_locs_in_sentence function 
    """
    lookup = {}
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("; ")
            lookup[parts[0].lower()] = (parts[1], parts[2])
    return lookup


def replace_locs_in_sentence(sentence: list, lookup: dict):
    """ Replaces LOC spans using the lookup table gathered from the build_lookup function, and rebuilds the correct sentence structure in terms of position (number) etc.
    Parameters
    ----------
    sentence : list 
        Sentence gathered from the group_sentences function
    lookup : dict 
        Dictionary consisting of key (original value) and value (endonym, latinization) 

    Returns
    -------
    out_orig : list 
        The original (endonym) version of the sentence 
    out_latin : list 
        The latinized version of the sentence
    """
    i = 0
    out_orig, out_latin = [], []

    while i < len(sentence):
        if sentence[i][2] == "B-LOC":
            span = [sentence[i]]
            j = i + 1
            while j < len(sentence) and sentence[j][2] == "I-LOC":
                span.append(sentence[j])
                j += 1

            entity = " ".join(tok[1] for tok in span)
            if entity.lower() not in lookup:
                print(f"Missing entity: '{entity}'")
                continue
            endo, latin = lookup[entity.lower()]

            for k, token in enumerate(endo.split()):
                out_orig.append(["_", token, "B-LOC" if k == 0 else "I-LOC", "_"])
            for k, token in enumerate(latin.split()):
                out_latin.append(["_", token, "B-LOC" if k == 0 else "I-LOC", "_"])

            i = j
        else:
            out_orig.append(sentence[i])
            out_latin.append(sentence[i])
            i += 1

    for k, tok in enumerate(out_orig):
        tok[0] = str(k + 1)
    for k, tok in enumerate(out_latin):
        tok[0] = str(k + 1)

    return out_orig, out_latin


def loc_replacer(line_list: list, file_name: str):
    """Helper function that swaps locations for the exonym & endonym part of our research question 
    Parameters
    ----------
    line_list : list 
        List of lines gathered from the read_file function 
    file_name : str
        Name of file containing the original, endonym and latinization of endonym for each location.

    Returns
    -------
    swapped_list_endonym : list 
        List of sentences with the locations swapped with the endonyms of those locations
    swapped_list_latin : list 
        List of sentences with the locations swapped with the latinization of those locations
    """
    lookup = build_lookup(file_name)
    swapped_list_endonym = []
    swapped_list_latin = []

    for sentence in group_sentences(line_list):
        endonym, latin = replace_locs_in_sentence(sentence, lookup)
        swapped_list_endonym.append(endonym)
        swapped_list_latin.append(latin)

    return swapped_list_endonym, swapped_list_latin


def sentences_to_lines(sentences: list):
    """Converts the sentences gathered from the loc_replacer helper function into clean list format, to work correctly with the export_swaps function 
    Parameters
    ----------
    sentences : list
        List of sentences gathered from loc_replacer / replace_locs_in_sentence function

    Returns
    ------
    lines : list
        List of lines compatible with the export_swaps function
    """
    lines = []
    for sentence in sentences:
        for tok in sentence:
            lines.append(" ".join(tok))
        lines.append("\n")
    return lines

def test_set_creator(region: str):
    """Helper function to create test sets for the person names 
    Parameters 
    ----------
    region : str 
    The name of the region / country, possibilities are: "African", "American", "Arabic", "European" or "Indian"

    """
    conll_file_path = "../data/test_conll.iob2"
    lines = read_file(conll_file_path)
    entity_dict = process_line(lines, ignore_bio=False)
    file_path_first_names = f"../data/person/{region.lower()}/{region.lower()}_first_names.csv"
    file_path_last_names = f"../data/person/{region.lower()}/{region.lower()}_last_names.csv"
    first_name_list = name_reader(file_path_first_names)
    last_name_list = name_reader(file_path_last_names)
    for i in range(10):
        person_swaps, middle_names, last_names = person_swap(entity_dict, first_name_list, last_name_list)
        swapped = swap_person_entities(lines, person_swaps, middle_names, last_names)
        os.makedirs(f"../data/person/{region.lower()}/testSets", exist_ok=True)
        export_swaps(swapped, f"../data/person/{region.lower()}/testSets/{region.lower()}_conll_{i+1}.iob2")
        print(f"Finished dataset {i+1} out of 10: Region = {region}")
if __name__ == "__main__":
    print("Hello World!")
    test_set_creator("indian")
    # conll_file_path = "../data/test_conll.iob2"
    # lines = read_file(conll_file_path)
    # swapped_list_endonym, swapped_list_latin = loc_replacer(lines, "../data/location_exonym_endonym/locations.iob2") 
    # swapped_list_endonym = sentences_to_lines(swapped_list_endonym)
    # swapped_list_latin = sentences_to_lines(swapped_list_latin)
    # export_swaps(swapped_list_endonym, "../data/location_exonym_endonym/location_endonym.iob2")
    # export_swaps(swapped_list_latin, "../data/location_exonym_endonym/location_latin.iob2")
    # entity_dict = process_line(lines, ignore_bio=False)
    # unique_counts = entity_count(entity_dict)
    # total_counts_span = count_span_lengths_person_tags(lines)
    # print(total_counts_span)
    # file_path_first_names = "../data/person/arabic_first_names.csv"
    # file_path_last_names = "../data/person/arabic_last_names.csv"
    # first_name_list = name_reader(file_path_first_names)
    # last_name_list = name_reader(file_path_last_names)
    # # print(first_name_list)
    # person_swaps, middle_names, last_names = person_swap(entity_dict, first_name_list, last_name_list)
    # swapped = swap_person_entities(lines, person_swaps, middle_names, last_names) 
    # export_swaps(swapped, "../data/person/test_person.iob2")
    # dist_dict = length_distribution_names(lines)
    # #print(dict(dist_dict["LOC"]))
    # swapped = random_string_generation_swap(lines, dist_dict)
    # export_swaps(swapped, "../data/random/test_random.iob2")
    #loc_grabber(lines, "../data/location_exonym_endonym/locations.iob2")

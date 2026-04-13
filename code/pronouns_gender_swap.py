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
    If documentation is unclear hmu @ +46764321744, I'll be happy to explain
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
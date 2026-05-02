def read_iob2_file(path):
    """
    Read provided Universal NER iob2 file
    
    :param path: path to read from
    :returns: list with sequences of words and NER labels for each sentence
    """
    data = []
    current_words = []
    current_ner_tags = []

    for line in open(path, encoding='utf-8'):
        line = line.strip()

        if line:
            if line[0] == '#':
                continue # skip comments
            tok = line.split(' ')
            #print(tok)
            current_words.append(tok[1])
            current_ner_tags.append(tok[2])
        else:
            if current_words:  # skip empty lines
                data.append((current_words, current_ner_tags))
            current_words = []
            current_ner_tags = []

    # check for last one
    if current_ner_tags != []:
        data.append((current_words, current_ner_tags))
    return data


#TODO Change the code such that sub-tokens divided by the model are also accounted for, by using indexing

def run_model(iob2file,model):
    """Runs a given model (imported from hugging face) on a given iob2 file
    ----------
    iob2file : string
        path to an iob2file with tokens and NER tags in the first 2 columns
    
    model : transformers.pipelines.token_classification.TokenClassificationPipeline
        the model pipeline to be tested, should include tokeniser and model

    Returns
    -------
    data : list 
        A list of tuples where each tuple is ([tokens], [ground_truth], [predictions])
    """
    data = read_iob2_file(iob2file)

    #combining all the sentences together to ensure one call to transformer
    combined_data = [" ".join(sentence[0] for sentence in data)]

    #running the model on the combined data
    all_results = model(combined_data, batch_size = 16)

    #A place to save the predictions and the old data
    new_data = []

    #Iterating over the original data and the results we got for each sentence
    for (tokens,actual), results in zip(data,all_results):
        pred = ['O'] * len(tokens)

        #getting indexes for each token for faster processing
        word_idx = {word : i for i,word in enumerate(tokens)}

        #check the results for the current sentence if matching token found, add its prediction using idx
        for j in results:
            if j["word"] in tokens:
                pred[word_idx[j["word"]]] = j["entity"]
        
        #append everything to new data
        new_data.append(tokens,actual,pred)
    
    return new_data

def recomplie_data(data,file_path):
    with open (file_path, "w") as F:
        for line in data:
            for i in range(len(line[0])):
                F.write(f"{i+1} {line[0][i]} {line[1][i]} {line[2][i]}\n")
            F.write("\n")
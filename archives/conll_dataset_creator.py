"""This file aims to create the dataset for conll-2003 as there is no download link on huggingface, hence we need to import it and export it to iob2 format"""
from datasets import load_dataset

dataset = load_dataset("lhoestq/conll2003")
output_folder = "../data"
### Labels are grabbed from the following link: https://huggingface.co/datasets/lhoestq/conll2003/blob/main/dataset_infos.json
ner_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
pos_labels = ["\"", "''", "#", "$", "(", ")", ",", ".", ":", "``", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "NN|SYM", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]
def to_iob2(dataset, ner_label_list, pos_label_list):
    """Converts the dataset into iob2 format, to match typical format of bio tagging 
    Parameters
    ----------
    dataset : Dataset 
        The dataset as input
    ner_label_list : list 
        A list of named entity labels, with corresponding positions matching the numerical representation of the dataset. In our case this means that e.g. 0 is O, since O has the id 0 in the conll-2003 dataset.
    pos_label_list : list
        A list of Part-of-Speech labels, with corresponding positions matching the numerical representation of the dataset. In our case this means that e.g. 0 is "\", since "\" has the id 0 in the conll-2003 dataset.
    Returns
    -------
    final_dataset : list 
    A list containing an entry for each word, along with its position in the
    sentence and the corresponding NER and POS tags.
    """
    final_dataset = []
    for line in dataset:
        ### Grabs each word in a given line and appends it to the final dataset
        for i, (token, tag_id, pos_id) in enumerate(zip(line["tokens"], line["ner_tags"], line["pos_tags"]), start=1):
            final_dataset.append([str(i), token, ner_label_list[tag_id], pos_label_list[pos_id]])
        ### Separator between sentences, this is technically not needed but its a nice to have when inspecting manually.
        final_dataset.append([])  
    return final_dataset 

def export_iob2(data, filename):
    """Exports the conll 2003 datasets to iob2 format.
    Parameters
    ----------
        data : list 
            List from the associated to_iob2() function
        filename : str
            Name of the exported iob2 file.
    """

    with open(filename, "w") as f:
        for row in data:
            if row:  # if not empty (sentence separator)
                f.write(" ".join(row) + "\n")
            else:
                f.write("\n")  # blank line between sentences



if __name__ == "__main__":
    train_dataset = to_iob2(dataset["train"], ner_labels, pos_labels)
    test_dataset = to_iob2(dataset["test"], ner_labels, pos_labels)
    export_iob2(train_dataset, f"{output_folder}/train_conll.iob2")
    export_iob2(test_dataset, f"{output_folder}/test_conll.iob2")



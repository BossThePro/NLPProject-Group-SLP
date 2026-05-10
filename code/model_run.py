import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

tokenizer_mono = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model_mono = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

tokenizer_multi = AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")
model_multi = AutoModelForTokenClassification.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")

test_dir = "../TestSets"
target_dir = "../predictions"
categories = ["random_person", "random_location"]
# "location_exonym_endonym","original_test","person","pronouns","gender_names", "random"
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


def run_model(file_path,tokenizer,model):

    data = read_iob2_file(file_path)
    
    ### Changes the device to the GPU if it exists, otherwise stays on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### Moves the predictions to the correct device (see above for clarification)
    model = model.to(device)
    ### list to store final output from all sentences
    final_output = []

    file_name = file_path[0:-5]
    
    for i in range(len(data)):
        ### Splitting tokens and ner tags, since both are included when inputting
        tokens, ner_tags = data[i]
        ### Creating a string sentence instead of list for computation
        sentence = " ".join(tokens)

        ### Applies the tokenizer to the input, with pytorch tensor format instead of general list format. 
        inputs = tokenizer(sentence, return_tensors="pt")

        ### Converts the actual integer id's, mask and token types to the correct device defined outside of the loop.
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        token_type_ids = inputs["token_type_ids"].to(device)
        
        ## Gets the outputs from the actual BERT model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        ### Picks the most probable NER tag
        predictions = torch.argmax(outputs.logits, dim=2)

        ### Converts the actual integer ids back into human readaable words, e.g. (SOCCER JAPAN ....)
        pred_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        ### Merges the subword tokens given by "##SUBWORD"
        merged = []
        for token, pred in zip(pred_tokens, predictions[0]):
            label = model.config.id2label[pred.item()]
            if token.startswith("##"):
                merged[-1]['word'] += token[2:]
            else:
                merged.append({'word': token, 'entity': label})
    
        ### Remove [CLS] and [SEP] tags, which are created by the BERT model for separation and classification (classification does not really apply to NER tasks)
        merged = [t for t in merged if t['word'] not in ['[CLS]', '[SEP]']]

        ### Makes a list with the format: original word, ground truth NER tag, predicted NER tag
        for gt_word, gt_tag, pred in zip(tokens, ner_tags, merged):
            final_output.append(f"{gt_word} {gt_tag} {pred["entity"]}")
        final_output.append("\n")

    return final_output, file_name

def recomplie_data(run_output,file_name):
    ### Outputs to a file with the format: original word, ground truth NER tag, predicted NER tag 
    with open(file_name, "w") as f:
        for line in run_output:
            if line == "\n":
                f.write("\n")
            else:
                f.write(line + "\n")


if __name__ == "__main__":
    total = 0
    for folder in categories:

        files = os.listdir(os.path.join(test_dir,folder))

        for file in files:
            print(f"Working on data {total+1}")
            output_mono, final_name_mono = run_model(os.path.join(test_dir,folder,file),tokenizer_mono,model_mono)
            mono_dir = os.path.join(target_dir, folder, "mono")
            os.makedirs(mono_dir,exist_ok=True)
            print(final_name_mono)
            file_name = file[:-5]
            final_loc_mono = os.path.join(mono_dir, file_name + "_results_mono.iob2")
            recomplie_data(output_mono, final_loc_mono)
            print(f"Finished mono model. File saved to: {final_loc_mono}")

            output_multi, final_name_multi = run_model(os.path.join(test_dir,folder,file),tokenizer_multi,model_multi)
            multi_dir = os.path.join(target_dir, folder, "multi")
            os.makedirs(multi_dir,exist_ok=True)
            final_loc_multi = os.path.join(multi_dir, file_name + "_results_multi.iob2")
            recomplie_data(output_multi, final_loc_multi)
            print(f"Finished multi model. File saved to: {final_loc_multi}")

            total +=1
            print(f"Finished dataset {total}")
            

#This is similar to span_f1.py however it is adjusted so our prediction files can be compitable with the code
#Just import the evaluate function into a jupyter notebook and run it with a file path and
# you will have 9 different metrics for that file

def readNlu(path):
    """Reads given iob2 file based on path, returns gold truth and predictions as seperate lists
    ----------
    path : str
        path to an iob2 file where the second column is the ground truth and the third column is the predictions
    
    Returns
    ----------
    annotations : list
        a list with all the ground truths where each list within is a seperate sentence
    
    predicition : list
        a list with all the predictions where each list within is a seperate sentence
    """
    annotations = []
    cur_annotation = []

    prediction = []
    cur_prediction = []

    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if line == '':
            annotations.append(cur_annotation)
            cur_annotation = []

            prediction.append(cur_prediction)
            cur_prediction = []
        elif line[0] == '#' and len(line.split(' ')) == 1:
            continue
        else:
            cur_annotation.append(line.split(' ')[1])
            cur_prediction.append(line.split(' ')[2])
    return annotations, prediction

#all the other functions are the same

def toSpans(tags):
    # Converts a list of tags to a list of spans
    # in: ['B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O']
    # out: {'7-9:ORG', '0-2:PER'}
    spans = set()
    for beg in range(len(tags)):
        if tags[beg][0] == 'B':
            end = beg
            for end in range(beg+1, len(tags)):
                if tags[end][0] != 'I':
                    break
            spans.add(str(beg) + '-' + str(end) + ':' + tags[beg][2:])
    return spans

def getBegEnd(span):
    return [int(x) for x in span.split(':')[0].split('-')]

def getLooseOverlap(spans1, spans2):
    # returns the overlap of spans without taking the exact boundaries
    # into account. If entities overlap they also count as found.
    found = 0
    for spanIdx, span in enumerate(spans1):
        spanBeg, spanEnd = getBegEnd(span)
        label = span.split(':')[1]
        match = False
        for span2idx, span2 in enumerate(spans2):
            span2Beg, span2End = getBegEnd(span2)
            label2 = span2.split(':')[1]
            if label == label2:
                if span2Beg >= spanBeg and span2Beg <= spanEnd:
                    match = True
                if span2End <= spanEnd and span2End >= spanBeg:
                    match = True
        if match:
            found += 1
    return found

def getUnlabeled(spans1, spans2):
    # Counts the overlap in spans after removing the labels
    return len(set([x.split(':')[0] for x in spans1]).intersection([x.split(':')[0] for x in spans2]))

def evaluate(file_path):
    gold_ners, pred_ners = readNlu(file_path)

    tp = 0
    fp = 0
    fn = 0

    recall_loose_tp = 0
    recall_loose_fn = 0
    precision_loose_tp = 0
    precision_loose_fp = 0

    tp_ul = 0
    fp_ul = 0
    fn_ul = 0 

    for gold_ner, pred_ner in zip(gold_ners, pred_ners):
        gold_spans = toSpans(gold_ner)
        pred_spans = toSpans(pred_ner)
        overlap = len(gold_spans.intersection(pred_spans))
        tp += overlap
        fp += len(pred_spans) - overlap
        fn += len(gold_spans) - overlap
        
        overlap_ul = getUnlabeled(gold_spans, pred_spans)
        tp_ul += overlap_ul
        fp_ul += len(pred_spans) - overlap_ul
        fn_ul += len(gold_spans) - overlap_ul

        overlap_loose = getLooseOverlap(gold_spans, pred_spans)
        recall_loose_tp += overlap_loose
        recall_loose_fn += len(gold_spans) - overlap_loose

        overlap_loose = getLooseOverlap(pred_spans, gold_spans)
        precision_loose_tp += overlap_loose
        precision_loose_fp += len(pred_spans) - overlap_loose

    print(f"For file {file_path}:")
    print()
    prec = 0.0 if tp+fp == 0 else tp/(tp+fp)
    rec = 0.0 if tp+fn == 0 else tp/(tp+fn)
    print('recall:   ', rec)
    print('precision:', prec)
    f1 = 0.0 if prec+rec == 0.0 else 2 * (prec * rec) / (prec + rec)
    print('slot-f1:  ', f1)

    print()
    print('loose (partial overlap with same label)')
    l_prec = 0.0 if precision_loose_tp + precision_loose_fp == 0 else precision_loose_tp/(precision_loose_tp+precision_loose_fp)
    l_rec = 0.0 if recall_loose_tp+recall_loose_fn == 0 else recall_loose_tp/(recall_loose_tp+recall_loose_fn)
    print('l_recall:   ', rec)
    print('l_precision:', prec)
    l_f1 = 0.0 if prec+rec == 0.0 else 2 * (prec * rec) / (prec + rec)
    print('l_slot-f1:  ', f1)

    tp = tp_ul
    fp = fp_ul
    fn = fn_ul
    print()
    print('unlabeled')
    ul_prec = 0.0 if tp+fp == 0 else tp/(tp+fp)
    ul_rec = 0.0 if tp+fn == 0 else tp/(tp+fn)
    print('ul_recall:   ', rec)
    print('ul_precision:', prec)
    ul_f1 = 0.0 if prec+rec == 0.0 else 2 * (prec * rec) / (prec + rec)
    print('ul_slot-f1:  ', f1)

    return prec, rec, f1, l_rec, l_prec, l_f1, ul_rec, ul_prec, ul_f1

        
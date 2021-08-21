import json
from numpy.lib.utils import source 
import spacy 
from spacy.lang.en import English 
from spacy.pipeline import EntityRuler 
from spacy.language import Language 

def load_data(file): 
    with open(file,"r", encoding='utf-8') as f: 
        data = json.load(f)
    return data 

def generate_better_characters(file): 
    data = load_data(file) 
    new_characters = []
    for item in data:
        new_characters.append(item) 
    
    for item in data: 
        item  = item.replace("The","").replace('the','').replace('and','')
        names = item.split(" ")
        for name in names: 
            name = name.strip() 
            new_characters.append(name) 
        
        if "(" in item: 
            names = item.split("(")
            for name in names: 
                name = name.replace(")","").strip() 
                new_characters.append(name) 


        if "," in item: 
            names = item.split(",")
            for name in names: 
                if " " in name:
                    new_names = name.split()
                    for new_name in new_names: 
                        new_characters.append(new_name.strip())
            new_characters.append(name.replace("and","").strip()) 
    final_characters = []
    titles = ['Dr.',"Professor","Mr.","Mrs.","Ms.","Miss","Aunt","Uncle","Mr. and Mrs."]
    for character in new_characters: 
        if "" != character: 
            final_characters.append(character) 
            for title in titles: 
                titled_char = f"{title} {character}" 
                final_characters.append(titled_char) 
    
    final_characters = list(set(final_characters))  
    final_characters = sorted(final_characters) 

    return final_characters 

def create_training_data(file, type): 
    data = generate_better_characters(file) 
    patterns = [] 
    for item in data: 
        pattern = {
            'label':type,
            'pattern':item
        }
        patterns.append(pattern)
    return patterns

def generate_rules(patterns): 
    nlp = English()
    source_nlp = spacy.load("en_core_web_sm") 
    nlp.add_pipe("ner", source=source_nlp) 
    reuler = EntityRuler(nlp) 
    reuler.add_patterns(patterns)
    nlp.to_disk('hp_ner') 

def test_model(nlp, text):
    doc = nlp(text) 
    results = []
    for ent in doc.ents:
        results.append(ent.text) 
    return results 


def save_data(file, data): 
    with open(file,"w", encoding='utf-8') as f: 
        json.dump(data,f, indent=4)


nlp = spacy.load('hp_ner') 
ie_data = {} 

with open('hp.txt','r', encoding='utf-8') as f: 
    text = f.read() 
    chapters = text.split('CHAPTERS')[1:]
    for chapter in chapters: 
        chapter_num, chapter_title = chapter.split('\n\n')[:2] 
        chapter_num = chapter_num.strip()
        chapter_title = chapter_title.strip() 
        segments = chapter.split('\n\n')[2:] 
        hits = [] 
        for segment in segments: 
            segment = segment.strip()
            segment = segment.replace('\n','') 
            results = test_model(nlp, segment)
            for result in results:
                hits.append(result)
        ie_data[chapter_num] = hits 

def train_spacy(data, iterations):
    from tqdm.notebook import tqdm 
    import random
    TRAIN_DATA = data 
    TRAIN_DATA2 = []
    for val  in TRAIN_DATA:
        if val:
            TRAIN_DATA2.append(val)
    
    nlp = spacy.blank('en')
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe('ner', last=True) 
    
    for _, annotations in TRAIN_DATA2:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2]) 
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner'] 
    if len(other_pipes) == 0:
        optimizer = nlp.begin_training() 
        for itn in tqdm(range(iterations)):
            print('starting iteration:' + str(itn))
            random.shuffle(TRAIN_DATA2)
            losses = {} 
            from spacy.training.example import Example 

            for batch in spacy.util.minibatch(TRAIN_DATA2, size = 256):
                for text, annotations in batch: 
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], losses=losses, drop = 0.3, sgd = optimizer)
    return nlp 




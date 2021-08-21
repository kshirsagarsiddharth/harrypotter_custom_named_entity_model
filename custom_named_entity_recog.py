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



import stanfordnlp

# Function to extract triplets
def extract_triplets(sentence, pipeline):
    doc = pipeline(sentence)
    triplets = []
    for sent in doc.sentences:
        subject, verb, obj = None, None, None
        for word in sent.words:
            if 'subj' in word.dependency_relation:
                subject = word.text
            elif 'VERB' in word.upos:
                verb = word.text
            elif 'obj' in word.dependency_relation:
                obj = word.text

        if subject and verb and obj:
            triplets.append((subject, verb, obj))
    return triplets

# Initialize StanfordNLP pipeline for English
nlp_en = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang='en', use_gpu=False)

# Initialize StanfordNLP pipeline for Spanish
nlp_es = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang='es', use_gpu=False)

# Test sentences
sentence_en = "Apple Inc. acquired Beats by Dre for innovation."
sentence_es = "Juan desayuna a las 9am. María camina al gimnasio a las 18:00. El gato duerme durante la noche. El tren llega a las 14:45. Sara y Emily estudian al mediodía. La reunión comienza por la mañana. La tienda abre 24/7. El avión parte al amanecer. Mike y Paul juegan fútbol los fines de semana. El festival empieza el primer día del verano."

# Extract triplets
triplets_en = extract_triplets(sentence_en, nlp_en)
triplets_es = extract_triplets(sentence_es, nlp_es)

# Display the triplets
print(f"English Triplets: {triplets_en}")
print(f"Spanish Triplets: {triplets_es}")

import spacy

from os import chdir, path

# Ensure the script's working directory is set to its location.
chdir(path.dirname(__file__))

# Step 1:Load the English language model for natural language processing.
nlp = spacy.load("en_core_web_md")


def movie_recommender(input_sentence):
    ''' 
    Step 2: Recommends movies based on input sentences.
    Process each movie description to extract its semantic meaning.
    Store similarity scores between the input sentence and movie 
    descriptions.
    Read movie descriptions from a file.
    Iterate through each movie description to compare it with the input 
    sentence.
    '''
    
    input_doc = nlp(input_sentence)
    similarity_scores = []
    
    with open('movies.txt', 'r', encoding="utf-8") as file:
        movie_descriptions = file.readlines()

        for sentence in movie_descriptions:
            
            sentence_doc = nlp(sentence)
            similarity_score = input_doc.similarity(sentence_doc)
            similarity_scores.append(similarity_score)

    # Step 3: Identify  the index of the highest score.
    top_indexes = [i for i, score in enumerate(similarity_scores) 
                   if score == max(similarity_scores)]

    # Step 4: Map the highest similarity score to corresponding movie title.
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    # Step 5: Return the recommended movie.
    return f'You should watch movie {letters[top_indexes[0]]}.'

# Step 6: Example usage of the movie recommender function.
input_sentence = """Will he save their world or destroy it? 
When the Hulk becomes too dangerous for the Earth, the Illuminati 
trick Hulk into a shuttle and launch him into space to a planet
where the Hulk can live in peace. Unfortunately, Hulk lands on
the planet Sakaar where he is sold into slavery and trained 
as a gladiator."""

recommendation = movie_recommender(input_sentence)
print(recommendation)
# import streamlit library to build the web application
import streamlit as st
# import joblib library to load pretrained models
import joblib
# display the headers and subheaders of the web applications
st.header('MOVIE/SERIES GENRE IDENTIFIER ')
st.subheader('Identify genre from plot')
# Create textareas to enter text
text_input = st.text_area('Enter the plot/description')
# provide button
if st.button('Click to identify'):
        tf_idf= joblib.load('tfidf.pkl') # loads pretrained  tfidf models
        model = joblib.load('genre_identifyer.pkl') # loads pretrainned genre_identifier model
        text_input = tf_idf.transform([text_input, ])  # transforms user entered text using TFIDF model
        prediction = model.predict(text_input) # predicts genre based on transformed data
        # define predefined order of genre
        predefined_order = ['action & adventure', 'anime', 'british tv shows',
                            'children & family movies', 'classic & cult tv', 'classic movies',
                            'comedies', 'crime', 'cult movies', 'docuseries and documentries',
                            'dramas', 'faith & spirituality', 'horror', 'independent movies',
                            'international', "kids' tv", 'korean tv shows', 'lgbtq movies',
                            'music & musicals', 'reality tv', 'romance', 'sci-fi & fantasy',
                            'science & nature tv', 'spanish-language tv shows', 'sports',
                            'stand-up & talk shows', 'stand-up and talk shows',
                            'teen tv shows', 'thrillers', 'tv mysteries']
        # extract the genres present based on prediction results by filtering out the ones with a value of 1.
        genres_present = [genre for genre, is_present in zip(predefined_order, prediction[0]) if is_present == 1]
        st.write(f'Genre:\n\n{genres_present}') # Displays the predicted genres 


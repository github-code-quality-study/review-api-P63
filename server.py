import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            query_string = environ.get('QUERY_STRING', '')
            query_params = parse_qs(query_string)

            location_filter = query_params.get('location', [None])[0]
            start_date_filter = query_params.get('start_date', [None])[0]
            end_date_filter = query_params.get('end_date', [None])[0]

            filtered_reviews = []
            for review in reviews:
                if location_filter and review['Location'] != location_filter:
                    continue
                if start_date_filter and review['Timestamp'] < start_date_filter:
                    continue
                if end_date_filter and review['Timestamp'] > end_date_filter:
                    continue
                filtered_reviews.append(review)

            for review in filtered_reviews:
                sentiment_scores = self.analyze_sentiment(review['ReviewBody'])
                review['sentiment'] = sentiment_scores

            # Sort the filtered reviews by sentiment's compound score in descending order
            filtered_reviews_sorted = sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)
            response_body = json.dumps(filtered_reviews_sorted, indent=2).encode("utf-8")

            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])

            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            try:
                content_length = int(environ.get('CONTENT_LENGTH', '0'))
                post_body = environ['wsgi.input'].read(content_length)
                post_params = parse_qs(post_body.decode('utf-8'))

                location = post_params.get('Location', [None])[0]
                review_text = post_params.get('ReviewBody', [None])[0]

                if not location or not review_text:
                    raise ValueError("Missing Location or ReviewBody")

                # Ensure location is valid
                valid_locations = [
                    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
                    "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
                    "El Paso, Texas", "Escondido, California", "Fresno, California",
                    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
                    "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
                    "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
                ]
                if location not in valid_locations:
                    raise ValueError("Invalid Location")

                review_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                new_review = {
                    "ReviewId": review_id,
                    "Location": location,
                    "Timestamp": timestamp,
                    "ReviewBody": review_text
                }

                reviews.append(new_review)

                response_body = json.dumps(new_review, indent=2).encode("utf-8")

                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])

                return [response_body]

            except Exception as e:
                error_message = {"error": str(e)}
                error_response = json.dumps(error_message).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(error_response)))
                ])
                return [error_response]
            return

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
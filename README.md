# Disease-Classification-with-AI and Web App deployment


This Flask web application predicts disease classification based on given features using machine learning. It provides predictions for various diseases such as Anemia, Diabetes, Healthy, Thalassemia, and Thrombosis.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/disease-classification-web-app.git
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Make sure you have a trained machine learning model saved as `disease_classifier_model.pkl` in the project directory.

## Usage

1. Run the Flask app:

    ```bash
    python app.py
    ```

2. The app will start running on `http://127.0.0.1:5000/`.

3. To predict disease classification, send a POST request to the `/predict` endpoint with the required features in JSON format.

4. Example usage:

    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}' http://127.0.0.1:5000/predict
    ```

5. The response will contain the predicted disease, its probability, and a slight funny line to lighten the mood.

## API Endpoints

- `/predict`: Endpoint to predict disease classification based on given features.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

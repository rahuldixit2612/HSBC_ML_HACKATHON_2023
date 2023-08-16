import pickle as pkl

class Inference():
    def __init__(self, prepared_data):
        self.prepared_data = prepared_data
        self.thrs = 0.5

    def load_model(self, model_path):
        with open(model_path, 'rb') as model_file:
            model = pkl.load(model_file)
        return model

    def prediction(self):
        model = self.load_model('best_model.pkl')
        transition_ids = self.prepared_data['V3'].tolist()
        confidence_scores = model.predict_proba(self.prepared_data.drop(columns=['V3'], axis=1))

        results = []
        for i in range(len(self.prepared_data)):
            non_fraud_confidence = confidence_scores[i][0]
            res = 0 if non_fraud_confidence >= self.thrs else 1
            confidence_score = 100 * confidence_scores[i][res]

            output = "Fraud" if res == 1 else "Non-Fraud"
            result = {'output': output,
                      'transition_id': transition_ids[i],
                      'confidence_score': confidence_score}
            results.append(result)

        return results

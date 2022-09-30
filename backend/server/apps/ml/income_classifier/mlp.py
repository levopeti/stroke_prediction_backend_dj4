from tensorflow import keras
import numpy as np


class MLP(object):
    def __init__(self):
        path_to_artifacts = "./research/"
        self.model = keras.models.load_model(path_to_artifacts + "model_90_1000000_all")
        self.model.summary()

    @staticmethod
    def preprocessing(input_data):
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)

        return input_data

    def predict(self, input_data):
        y_pred_list = self.model.predict(input_data)
        label = np.array(y_pred_list).argmax(axis=1)
        probability = [y_pred_list[i][label[i]] for i in range(len(label))]

        threshold = 4.5
        is_stroke = label.mean() < threshold

        return {"probability": probability, "label": label.tolist(), "is_stroke": is_stroke}

    def compute_prediction(self, input_data, timestamps):
        preprocessed_data = self.preprocessing(input_data)
        result_dict = self.predict(preprocessed_data)
        result_dict["timestamps"] = timestamps

        return result_dict


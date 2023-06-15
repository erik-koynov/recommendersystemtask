from similarity_learning.inference import Model
from similarity_learning.inference.model_output import ModelOutput
from similarity_learning.loading_pipeline import UnlabeledPairsDataFrame
from typing import List, Dict, Any

class Search:
    def __init__(self, model: Model) -> None:
        self.model = model

    def match(self, talent: dict, job: dict) -> dict:
        """
        ==> Method description <==
                This method takes a talent and job as input and uses the machine learning
                model to predict the label. Together with a calculated score, the dictionary
                returned has the following schema:

                {
                  "talent": ...,
                  "job": ...,
                  "label": ...,
                  "score": ...
                }
        """
        data = UnlabeledPairsDataFrame.from_full_json(job, talent, add_language_suffix=True)
        output: ModelOutput = self.model.predict(data)
        return {"talent": talent, "job": job, "label": output.labels[0], "score": output.similarity_scores[0]}

    def match_bulk(self, talents: list[dict], jobs: list[dict]) -> list[dict]:
        """
        ==> Method description <==
        This method takes a multiple talents and jobs as input and uses the machine
        learning model to predict the label for each combination. Together with a
        calculated score, the list returned (sorted descending by score!) has the
        following schema:

        [
          {
            "talent": ...,
            "job": ...,
            "label": ...,
            "score": ...
          },
          {
            "talent": ...,
            "job": ...,
            "label": ...,
            "score": ...
          },
          ...
        ]
        """
        data = UnlabeledPairsDataFrame.from_full_json(jobs, talents, add_language_suffix=True)
        output: ModelOutput = self.model.predict(data)
        return self.format_batch_output(talents, jobs, output)

    def format_batch_output(self,
                            talents: List[Dict[str, Any]],
                            jobs: List[Dict[str, Any]],
                            model_output: ModelOutput)->List[Dict[str, Any]]:
        outputs = []

        for talent, job, (label, score) in zip(talents, jobs, model_output):
            outputs.append({"talent": talent, "job": job, "label": label, "score": score})

        return outputs
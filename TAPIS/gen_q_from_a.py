import pandas as pd
from pipelines import pipeline
import nltk

nltk.download("punkt")

df_a = pd.read_csv("answers.csv")
df_q = pd.DataFrame()
nlp = pipeline("e2e-qg")
for idx, answer in enumerate(df_a["answerText"]):
    questions = nlp(answer)
    for q in questions:
        df_q.append({"Question":q, "Answer Label":idx, "Answer":answer}, ignore_indexc=True)

df_q.to_csv("questions.csv")
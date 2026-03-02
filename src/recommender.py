import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SHLRecommender:

    def __init__(self, assessment_file, role_file):

        self.assessments = pd.read_csv(assessment_file)
        self.assessments["skills"] = self.assessments["skills"].str.lower()
        self.roles = pd.read_csv(role_file)

        self.vectorizer = TfidfVectorizer()

        self.skill_vectors = self.vectorizer.fit_transform(
            self.assessments["skills"]
        )


    def role_to_skills(self, role):
        role_row = self.roles[
                 self.roles["job_role"].str.lower() == role.lower()
        ]
        if role_row.empty:
            return None
        
        return role_row.iloc[0]["skills"]

    def recommend(self, user_skills, top_n=3):

        user_vec = self.vectorizer.transform([user_skills.lower()])

        similarity = cosine_similarity(user_vec, self.skill_vectors)

        scores = similarity.flatten()

        results = self.assessments.copy()
        results["match_score"] = scores

        results = results.sort_values(by="match_score", ascending=False)

        results = results[results["match_score"] > 0]
        return results.head(top_n)
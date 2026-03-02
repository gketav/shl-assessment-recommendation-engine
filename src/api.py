from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.recommender import SHLRecommender

app = FastAPI()

recommender = SHLRecommender(
    assessment_file="data/assessment.csv",
    role_file="data/roles.csv"
)

@app.get("/")
def home():
    return RedirectResponse(url="/docs")


@app.get("/recommend/skills")
def recommend_skills(skills: str):

    results = recommender.recommend(skills).head(5)

    return results[
        ["assessment_name", "job_role", "skills", "match_score"]
    ].to_dict(orient="records")


@app.get("/recommend/role")
def recommend_role(role: str):

    role_row = recommender.roles[
        recommender.roles["job_role"].str.lower() == role.lower()
    ]

    if role_row.empty:
        return {"error": "Role not found"}

    skills = role_row.iloc[0]["skills"]

    results = recommender.recommend(skills).head(5)

    return results[
        ["assessment_name", "job_role", "skills", "match_score"]
    ].to_dict(orient="records")


@app.get("/recommend/full")
def recommend_full(role: str):

    skills = recommender.role_to_skills(role)

    if skills is None:
        return {"error": "Role not found"}

    results = recommender.recommend(skills).head(5)

    return results[
        ["assessment_name", "job_role", "skills", "match_score"]
    ].to_dict(orient="records")
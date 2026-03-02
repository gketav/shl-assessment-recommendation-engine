from src.recommender import SHLRecommender

engine = SHLRecommender(
    "data/assessment.csv",
    "data/roles.csv"
)

print("\n--- Skill Based Example ---\n")
print(engine.recommend("python backend api database"))

print("\n--- Frontend Example ---\n")
print(engine.recommend("html css javascript"))

print("\n--- Role Based Example ---\n")
skills = engine.role_to_skills("frontend developer")
print(engine.recommend(skills))

print("\n--- ML Example ---\n")
print(engine.recommend("python machine learning statistics"))
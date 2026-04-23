SELECTED_LABELS = [
    "Blockchain", "Data Science", "Technology", "Programming", "Poetry",
    "Cryptocurrency", "Machine Learning", "Life", "Writing", "Politics",
    "Startup", "Life Lessons", "Self Improvement", "Covid 19", "Software Development",
    "Love", "Python", "Business", "Health", "Mental Health",
    "JavaScript", "Relationships", "Education", "Artificial Intelligence",
    "Culture", "Design", "Self", "Marketing", "Entrepreneurship", "Personal Development",
]

TAG_TO_IDX = {tag: idx for idx, tag in enumerate(SELECTED_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(SELECTED_LABELS)}
LABEL2ID = {label: i for i, label in enumerate(SELECTED_LABELS)}

TARGET_TAGS = ["Blockchain", "Data Science", "Technology", "Programming", "Poetry"]
MODEL_NAME = "DistilBERT_Tagger"

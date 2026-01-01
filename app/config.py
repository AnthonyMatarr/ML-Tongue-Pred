CHOSEN_MODEL_DICT = {
    "surg": "svc",
    "bleed": "lgbm",
    "asp": "lr",
    # "mort": "stack",
    "reop": "svc",
}
# Configuration
OUTCOMES = {
    "Aspiration Complications": "asp",
    "Bleeding Transfusion": "bleed",
    "Unplanned Reoperation": "reop",
    # "Death": "mort",  # binning is so bad for this, would be detrimental to include
    "Surgical Complications": "surg",
}

# COPUS-ML
Applying ML to COPUS analysis via. Computer Vision


# Evaluating
Basic eval with multi-label:
```
python copus_evaluation.py data/processed/20241201/20241201_001.mp4 --verbose
```

Batch eval with custom thresholds:
```
python copus_evaluation.py data/processed/20241201/ --batch --threshold 0.25 --max-labels 4 --output results.json
```

Compare w/ multiple ground truth labels:
```
python copus_evaluation.py video.mp4 --ground-truth instructor_lecturing instructor_real_time_writing student_listening
```
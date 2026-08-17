## Lung and Colon cancer detection

# 📊 Where is the database from?

The [database] and information is from Kaggle, from a user called Pulkit Sanan. (https://www.kaggle.com/code/pulkitsanan/cancer/notebook)

# 🔥 How does this model work?

1. The algorythm extracts features from the selected database. (Stage 1 in the code)

'''
python run.py extract --data-dir "path/to/directory" --limit-per-class 300 --out feats_small.npz
'''

where you put your data directory that contains the images.

2. Train the model

It uses the basic 80/20 train test split. To train the model use:

'''
python run.py train --features feats_small.npz --model-out model.joblib
'''

3. Finally, to test the model, please use the following command:

'''
python run.py predict --model model.joblib --image "path/to/test_image.jpeg"
'''


# ✅ Results?

My model got a 93.67% accuracy, correctly predicting a adenocarcinoma with 96% confidence.
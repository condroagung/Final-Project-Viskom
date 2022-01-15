YOLO STEP BY STEP DETECTION AGE GENDER

DOWNLOAD PRETRAINED MODEL (AGEGENDER, FACE)

1. Jalankan get_model.py

FACE DETECTION ( dataset FDBB )

1. Download File FDBB ( OriginalPics dan FDDB-folds )
2. Buat folder fdbb di dalam folder dataset dan extract file ke folder fdbb tersebut
3. Run Preprocess_fdbb.py

ANNOTATION CONVERT AND CHECK

1. Run Annotasi Convert
2. View Annotasi dataset FDBB ( run show_annotate.py )

AGE GENDER CLASSIFICATION ( dataset IMDB )

1. Download File IMDB
2. Extract file ke dalam folder dataset
3. Run Preprocess_imdb.py

Training Phase ( yolo_train.py )

Testing Phase ( realtime_predict.py )

-----------------------------------------------------------------------------------------------

UNTUK MELAKUKAN DETEKSI LANGSUNG

1. Jalankan get_model.py
2. Jalankan realtime_predict.py
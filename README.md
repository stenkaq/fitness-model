## How to run

1. Create and activate a virtual environment (Python 3.11):  
python -m venv venv

   **On Windows:**  
   .\venv\Scripts\activate

   **On Linux/macOS:**  
   source venv/bin/activate

2. Install dependencies:  
pip install -r requirements.txt

3. Download dataset from Kaggle:  
Go to the link: https://www.kaggle.com/datasets/adnanelouardi/600k-fitness-exercise-and-workout-program-dataset/  
Download the archive, extract the files, and place them in the `data/` folder of your project as follows:  
- data/PROGRAM_SUMMARY.csv  
- data/PROGRAM_EXERCISES.csv

4. Run the application:  
python run.py

5. Open in your browser:  
http://127.0.0.1:5000

# # app.py
# from fastapi import FastAPI, File, UploadFile
# import pandas as pd
# from io import StringIO
# from sklearn.model_selection import train_test_split

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# import pandas as pd
# import joblib


# app = FastAPI()

# class TrainRequest(BaseModel):
#     algorithm: str  # Algorithm chosen by the user
#     data: dict      # Your data should come as a dictionary of lists
    
    
# ALGORITHMS = {
#     "random_forest": RandomForestClassifier,
#     "logistic_regression": LogisticRegression,
#     "decision_tree": DecisionTreeClassifier
# }

# # Add a new route to handle the model training
# @app.post("/train")
# def train_model(request: TrainRequest):
#     # Convert the input data to a DataFrame
#     df = pd.DataFrame(request.data)
    
#     # Check if the target column is present
#     if 'target' not in df.columns:
#         raise HTTPException(status_code=400, detail="Missing 'target' column in the data.")
    
#     # Separate features and target
#     X = df.drop(columns=['target'])
#     y = df['target']
    
#     # Split the data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
#     # Select the algorithm based on user input
#     Algorithm = ALGORITHMS.get(request.algorithm)
#     if not Algorithm:
#         raise HTTPException(status_code=400, detail="Algorithm not supported.")
    
#     # Train the selected algorithm
#     model = Algorithm()
#     model.fit(X_train, y_train)
    
#     # Make predictions and calculate accuracy
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
    
#     # Return accuracy and the algorithm used
#     return {"accuracy": accuracy, "algorithm": request.algorithm}


# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the AutoML Platform"}

# @app.post("/upload-data/")
# async def upload_data(file: UploadFile = File(...)):
#     # Read the uploaded file
#     contents = await file.read()
#     df = pd.read_csv(StringIO(contents.decode("utf-8")))

#     # Basic data information
#     info = {
#         "columns": list(df.columns),
#         "shape": df.shape,
#         "preview": df.head().to_dict()
#     }
#     return {"data_info": info}

# @app.post("/preprocess-data/")
# async def preprocess_data(file: UploadFile = File(...)):
#     contents = await file.read()
#     df = pd.read_csv(StringIO(contents.decode("utf-8")))

#     X = df.iloc[:, :-1]  # All columns except the last (features)
#     y = df.iloc[:, -1]   # Last column (target)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Return basic shapes of the split data
#     return {
#         "X_train_shape": X_train.shape,
#         "X_test_shape": X_test.shape,
#         "y_train_shape": y_train.shape,
#         "y_test_shape": y_test.shape
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
    
    
# # app.py (Extended)
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# @app.post("/train-model/")
# async def train_model(file: UploadFile = File(...)):
#     # Load and preprocess data
#     contents = await file.read()
#     df = pd.read_csv(StringIO(contents.decode("utf-8")))
    
#     X = df.iloc[:, :-1]  # Features
#     y = df.iloc[:, -1]   # Target
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train a simple model (Random Forest in this case)
#     model = RandomForestClassifier()
#     model.fit(X_train, y_train)

#     # Make predictions and evaluate the model
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)

#     # Save the model for later use
#     with open('model.pkl', 'wb') as f:
#         joblib.dump(model, f)

#     return {
#         "accuracy": accuracy,
#         "X_test_shape": X_test.shape,
#         "y_test_shape": y_test.shape
#     }
    


from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from io import StringIO
import joblib

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALGORITHMS = {
    "random_forest": RandomForestClassifier,
    # You can add other algorithms here as needed
}

class TrainRequest(BaseModel):
    algorithm: str  # Algorithm chosen by the user
    data: dict      # Your data should come as a dictionary of lists

@app.get("/")
def read_root():
    return {"message": "Welcome to the AutoML Platform"}

@app.post("/upload-data/")
async def upload_data(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    # Basic data information
    info = {
        "columns": list(df.columns),
        "shape": df.shape,
        "preview": df.head().to_dict()
    }
    return {"data_info": info}

# @app.post("/train-model/")
# async def train_model(file: UploadFile = File(...), algorithm: str = "random_forest"):
#     contents = await file.read()
#     df = pd.read_csv(StringIO(contents.decode("utf-8")))

#     # Assuming the last column is the target
#     X = df.iloc[:, :-1]  # Features
#     y = df.iloc[:, -1]   # Target

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train the selected algorithm
#     Algorithm = ALGORITHMS.get(algorithm)
#     if not Algorithm:
#         raise HTTPException(status_code=400, detail="Algorithm not supported.")
    
#     model = Algorithm()
#     model.fit(X_train, y_train)

#     # Make predictions and calculate accuracy
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)

#     # Save the model for later use
#     with open('model.pkl', 'wb') as f:
#         joblib.dump(model, f)

#     return {
#         "accuracy": accuracy,
#         "X_test_shape": X_test.shape,
#         "y_test_shape": y_test.shape
#     }
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

@app.post("/train-model/")
async def train_model(file: UploadFile = File(...), algorithm: str = Form("random_forest")):
    print(f"Received file: {file.filename}, algorithm: {algorithm}")  # Debugging info
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        X = df.iloc[:, :-1]  # Features
        y = df.iloc[:, -1]   # Target
        
        # Check for the chosen algorithm
        if algorithm not in ALGORITHMS:
            raise HTTPException(status_code=400, detail="Invalid algorithm specified.")

        # Train the model
        model = ALGORITHMS[algorithm]()
        model.fit(X, y)

        # Assuming you're returning the accuracy of the model
        accuracy = model.score(X, y)  # Example for returning accuracy
        return {"accuracy": accuracy}
    except Exception as e:
        print(f"Error: {e}")  # Print the error in the console
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



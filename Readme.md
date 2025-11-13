# andycshang-mlops-f25
# A0
This is my first assignment of 596C. Before building the model, my goal is to first study the tutorials, then complete 
an initial setup of the model to obtain preliminary prediction results, and afterward proceed with parameters testing 
and more complex data processing.

### The process of building my model is as following:

First, by studying the tutorial I learned the model-building workflow, the purpose and underlying principles of each 
step, and produced an initial model. At this stage I set the number of iterations and learning rate to iterations = 1000
and alpha = 1.0e-6, and initialized the weight vector w and bias term b to zero. The minimum cost I obtained was 39.35, 
and the MSE was relatively high at 66.89.

By adjusting different iteration counts and learning rates, it can be observed that within a certain range, a higher 
number of iterations results in a lower cost. When the number of iterations is set to 10,000 and the learning rate 
to 1e-6, the cost reaches 32.94 and the training MSE is 32.48. However, the learning rate needs to be kept 
within an appropriate range — if it is too small, the convergence will be slow and accuracy may be affected with fewer 
iterations, while if it is too large, the model may diverge.


Next, I started to consider which factors influence the cost — in other words, how to improve model performance. In the 
initial model I had not normalized or standardized the dataset.I noticed that the feature values in the dataset varied 
greatly, such as between 'B' and 'AGE'. Therefore, following the approach in the tutorial, I standardized the 
features and obtained the following results:

*Iteration  900: Cost   302.61*  
*Training result: w = [-0.00353237  0.00287454 -0.00437161  0.00177588 -0.00386185  0.00661378
 -0.00315909  0.00217926 -0.00359845 -0.00427178 -0.00456878  0.00315981
 -0.00684981], b = 0.022785151571493724*  
*Training MSE = 302.53316961619663*

I observed that using the standardized data led to a significant increase in both the cost and the training MSE. I 
suspect that the reason might be that I only standardized the 'X' data and did not standardize 'y', while the MSE 
is calculated based on the comparison between the two. This likely caused the noticeable discrepancy in the results.
Therefore, I attempted to standardize the y data as well, using the following code:

*y_scaler = StandardScaler()*  
*y_train_norm = y_scaler.fit_transform(y_train.values.reshape(-1, 1))*  
*y_test_norm = y_scaler.transform(y_test.values.reshape(-1, 1))*

However, after standardizing both X and y, a dimension mismatch occurred between the data matrix and the weight
vector w. This issue is relatively complex for me at the moment, and I have not yet fully resolved it. I am 
currently continuing to revise and debug the model. And this is the most difficult part so far.

# A1
### my URL: 54.90.229.217:8000/docs
### To avoid conflicts with the A0 assignment repository, I created a new public repository for A1 and cloned the model completed in A0 into this repository to continue development.

### 1. Write the FastAPI configuration file

I wrote a FastAPI configuration file `api.py`, created a FastAPI instance `app`, and added the core function `predict`:

```
@app.post("/predict")
def predict_endpoint(input: InputData):
    X = np.array([input.data])
    X_scaled = x_scaler.transform(X)
    y_pred = np.dot(X_scaled, w) + b
    return {"prediction": float(y_pred[0])}
```

Users can input a list on the web interface as `InputData`, which will be passed into the model to compute prediction results.

### 2. Create a Dockerfile

In the local project, I created a `Dockerfile` to configure the code environment, working directory, imported files, dependencies to install, exposed port, and FastAPI startup command.

### 3. Build a Docker image locally

* `docker build -t fastapi-model:latest .`

Check Docker status:

* `docker images`
* `docker ps`

### 4. Run and test the Docker container

* `docker run -d -p 8000:8000 fastapi-model`

Access in browser: [http://localhost:8000](http://localhost:8000)

### 5. Export and upload the image to EC2

Export as a `.tar` file:

* `docker save -o fastapi-model.tar fastapi-model:latest`

Upload to EC2:

* `scp -i mymodel.pem fastapi-model.tar ec2-user@54.90.229.217:/home/ec2-user/`

On EC2, import and run the image:

```
ssh -i mymodel.pem ec2-user@54.90.229.217
sudo docker load -i fastapi-model.tar
sudo docker images
sudo docker run -d -p 8000:8000 fastapi-model
```

### The above completes the manual process of building and deploying the Docker container to AWS.

Next, we automate this process.

### 6. Create a deploy script

In the local project directory, create a `deploy.sh` file and write:

```
nano deploy.sh
```

```
#!/bin/bash
set -e  # exit if false

PROJECT_NAME="andycshang-mlops-f25"
IMAGE_NAME="fastapi-model"
EC2_USER="ec2-user"
EC2_IP="54.90.229.217"
PEM_KEY="mymodel.pem"
REMOTE_PATH="/home/ec2-user"
PORT="8000"

echo "Step 1/6: Building Docker image..."
docker build -t $IMAGE_NAME:latest .

echo "Step 2/6: Saving Docker image to tar file..."
docker save -o ${IMAGE_NAME}.tar ${IMAGE_NAME}:latest

echo "Step 3/6: Uploading tar to EC2..."
scp -i $PEM_KEY ${IMAGE_NAME}.tar ${EC2_USER}@${EC2_IP}:${REMOTE_PATH}/

echo "Step 4/6: Cleaning up local tar file..."
rm -f ${IMAGE_NAME}.tar

echo "Step 5/6: Loading image and restarting container on EC2..."
ssh -i $PEM_KEY ${EC2_USER}@${EC2_IP} << EOF
    set -e
    echo "Loading Docker image..."
    sudo docker load -i ${REMOTE_PATH}/${IMAGE_NAME}.tar
    echo "Stopping old container (if exists)..."
    sudo docker stop ${IMAGE_NAME} || true
    sudo docker rm ${IMAGE_NAME} || true
    echo "Starting new container..."
    sudo docker run -d -p ${PORT}:${PORT} --name ${IMAGE_NAME} ${IMAGE_NAME}:latest
    echo "Deployment complete! Running container:"
    sudo docker ps | grep ${IMAGE_NAME}
EOF

echo "Step 6/6: Deployment finished successfully!"
```

### 7. Deploy by running the script

* `./deploy.sh`


## test example:
{
    "CRIM": 0.03,
    "ZN": 18.0,
    "INDUS": 2.3,
    "CHAS": 0,
    "NOX": 0.4,
    "RM": 6.5,
    "AGE": 45.0,
    "DIS": 5.2,
    "RAD": 1,
    "TAX": 290.0,
    "PTRATIO": 17.8,
    "B": 390.0,
    "LSTAT": 12.5
}


# A3

### Adding Data Input Validation to the API

We defined the `HouseData` class to validate inputs:

```python
from pydantic import BaseModel, Field

class HouseData(BaseModel):
    CRIM: float = Field(..., ge=0)
    ZN: float = Field(..., ge=0)
    INDUS: float
    CHAS: int = Field(..., ge=0, le=1)
    NOX: float = Field(..., ge=0, le=1)
    RM: float = Field(..., ge=0)
    AGE: float = Field(..., ge=0, le=100)
    DIS: float = Field(..., ge=0)
    RAD: int = Field(..., ge=1)
    TAX: float = Field(..., ge=0)
    PTRATIO: float = Field(..., ge=0)
    B: float = Field(..., ge=0)
    LSTAT: float = Field(..., ge=0)
```

**Explanation of the validation rules:**

* `Field(...)` ensures that a value must be provided; an error is raised if the user omits it.
* `int`, `float`, etc., enforce the expected data type.
* `ge` (greater than or equal) and `le` (less than or equal) define the valid range for the input.

---

### Updated `predict` Endpoint

The API's `predict` method was modified to use the validated input:

```python
@app.post("/predict")
def predict_endpoint(input_data: HouseData):
    X = np.array([[input.CRIM, input.ZN, input.INDUS, input.CHAS, input.NOX,
                   input.RM, input.AGE, input.DIS, input.RAD, input.TAX,
                   input.PTRATIO, input.B, input.LSTAT]])
    X_scaled = x_scaler.transform(X)
    y_pred = np.dot(X_scaled, w) + b
    return {"prediction": float(y_pred[0])}
```

---

### Input Validation Example

When input values exceed the allowed range, the API responds with an error:

**Request:**

* `CHAS` = 2 (should be ≤ 1)

**Response:**

```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "CHAS"],
      "msg": "Input should be less than or equal to 1",
      "input": 2,
      "ctx": {"le": 1}
    }
  ]
}
```

---

### Automated Testing

We automated this validation in `test_api.py`:

* **Valid input:** returns HTTP 200 + prediction
* **Missing required field:** returns HTTP 422 (Unprocessable Entity)
* **Wrong data type:** returns HTTP 422

**Example test results:**

```
test_api.py::test_predict_happy_path PASSED                              [ 33%]
test_api.py::test_predict_missing_field PASSED                           [ 66%]
test_api.py::test_predict_invalid_type PASSED                            [100%]
```

> "PASSED" indicates that the test cases correctly handle different scenarios.

---

### Troubleshooting

One of the most persistent errors encountered was:

```
========================= MLFcode/test_api.py:None (MLFcode/test_api.py)
MLFcode/test_api.py:2: in <module> from api import app 
MLFcode/api.py:11: in <module> with open("model.pkl", "rb") as f: 
E FileNotFoundError: [Errno 2] No such file or directory: 'model.pkl'
```

Even though `model.pkl`, `test_api.py`, and `api.py` were in the same folder, the file could not be found.

**Solution:**
Changed the pytest working directory in **Edit Configuration** to the current folder. After this adjustment, the error was resolved.

---

### Adding Versioning for Model Deployment

To support multiple model versions, the following was added to `api.py`:

```python
import os

MODEL_DIR = "models"  # Directory to store different model versions
VERSION = "1.0"       # Default latest model version
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, f"model_v{VERSION}.pkl")
```

* A new folder `models` was created to store versioned models.
* `VERSION` defines the default model version used by the API.
* `MODEL_PATH` constructs the full path to the versioned model file.













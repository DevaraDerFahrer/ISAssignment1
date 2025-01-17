# ISAssignment1

Link to Google Drive
`https://drive.google.com/file/d/1jrywqqlVqQHRqj1vFLoWKuDcBOm6udMA/view?usp=sharing`

Link to Github
`https://github.com/DevaraDerFahrer/ISAssignment1`
 or
`https://github.com/DevaraDerFahrer/ISAssignment1.git`

First of all, run this code to install mylibrary locally as local project module:
`pip install -e .`

First initialize the dataset by running these commands:
`cd dataprocessor`
`python process.py`
`python process.py`

To train each of individual models first go to its folder, for example:
`cd model8_SVM1_CNN2`
then run the main code by:
`python main.py`

to test average ensemble model, first extract the weights by running these code:
`cd ensemble_model_average`
`python getWeights.py`
then run the main code:
`python main.py`

to test voting ensemble model run these code:
`cd ensemble_model_voting`
`python main.py`

to test stacking ensemble model, first train the model by running these code:
`cd ensemble_model_stacking`
`python main.py`
then to test the model, run this code:
`python test.py`
import threading
import pandas as pd
import pandas_datareader.data as web
import datetime
from sklearn import linear_model
from time import sleep
import joblib
import requests

# This class is to give colors to the print statements
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# This function will give the price after the delay of sleep_time units of seconds
def getprice(label="TSLA", sleep_time=1):
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{label}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    }
    # Pause for sleep_time number of seconds
    sleep(sleep_time)

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        content = response.content.decode('utf-8')
        lines = content.strip().split('\n')
        latest_line = lines[-1]
        price = float(latest_line.split(',')[4])
        current_time = datetime.datetime.now()
        return price, current_time
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving data for {label}: {e}")
        return None, None


# This function trains the model using the input data(dataframe)
def train(input):
    print("\nModel updating...", end=" ")
    # We take the last column of the features as the target and the rest are taken as attributes
    featureMat = input.iloc[:, : len(input.columns) - 1]
    label = input[input.columns[-1]]
    # Here we are using linear regression model
    model = linear_model.LinearRegression()

    with lock:
        model.fit(featureMat, label)
        joblib.dump(model, "modelLR.pkl")

    print("[Completed]")


##########################  Training over  ################################

# Increase the values of these variables to get improved results
# but if you will increase them then you have to wait for
#
#               (number_of_features X training_record_criteran) X sleep_time units seconds
#
# for the first training

number_of_features = 5  # This indicates how many columns the dataframe will have.
training_record_criterian = 5  # This decides how frequently the model will update [5 new features -> retrain the model]
number_of_predictions = 3  # Tells how many predictions in a series you want

##################################################################################

data = pd.DataFrame(columns=range(number_of_features))  # creating an empty dataframe
predict_input = list()
lock = threading.Lock()  # initialize the lock

while True:
    feature = list()  # stores the features for a single record for the dataframe

    for i in range(number_of_features):
        price = getprice()[0]
        feature.append(price)
        predict_input.append(price)

        try:  # this will throw an exception in two cases:
            # 1> model is not yet trained and saved
            # 2> model prediction is not working.

            first_predict = True  # flag for detecting the first prediction in predicted series
            model = joblib.load("modelLR.pkl")  # trying to open the saved model (can throw an exception)
            print("")
            inputlist = predict_input.copy()  # copying the list to make the prediction if the model is ready
            #   printing latest 3 prices
            for feature_value in inputlist[-(3):]:
                print(f"{bcolors.WARNING} --> ", int(feature_value * 100) / 100, end=" ")
            #   taking the latest price
            price = getprice(sleep_time=0)[0]
            #   Starting the predictions
            for i in range(number_of_predictions):
                pre_price = model.predict([inputlist[-(number_of_features - 1):]])
                #   printing the predicted values one by one in the series
                print(f"{bcolors.OKBLUE} --> ", int(pre_price[0] * 100) / 100, end=" ")
                #   This block will only run for the first prediction in the series
                if first_predict:
                    # When the prediction indicates an increase in price
                    if pre_price[0] - inputlist[-1] > 0:
                        print(f"{bcolors.OKGREEN}  \u2191", end="")
                        #   Calculating the % increase the program predicts and printing.
                        print(f"{bcolors.BOLD}[", int((pre_price[0] - price) * 1000000 / price) / 10000, "%] ", end=" ")
                        print(f"{bcolors.OKCYAN} Actual: ", price, end="")
                    # When the prediction says that no change will happen
                    elif pre_price[0] - inputlist[-1] == 0:
                        print(f"{bcolors.HEADER} \u2022", end="")
                        print(f"{bcolors.BOLD}[", int((pre_price[0] - price) * 1000000 / price) / 10000, "%] ", end=" ")
                        print(f"{bcolors.OKCYAN} Actual: ", price, end="")
                    # When the prediction is about a decrease in price
                    else:
                        print(f"{bcolors.FAIL}  \u2193", end="")
                        print(f"{bcolors.BOLD}[", int(-(pre_price[0] - price) * 1000000 / price) / 10000, "%] ",
                              end=" ")
                        print(f"{bcolors.OKCYAN} Actual: ", price, end="")
                    # Next statement talks about what happened actually
                    if price - inputlist[-1] > 0:
                        print(f"{bcolors.OKGREEN}  \u2191", end=" ")
                    elif price - inputlist[-1] == 0:
                        print(f"{bcolors.HEADER} \u2022", end="")
                    else:
                        print(f"{bcolors.FAIL}  \u2193", end=" ")

                    first_predict = False

                #   pushing the predicted price to the back of the input array
                #   it will be used in predicting the next element in the series
                inputlist.append(pre_price[0])

        except:
            print("Please Wait while the model is getting ready...")

    #   Adding the feature in the dataframe
    data.loc[len(data.index)] = feature
    #   If the number of elements present in the dataframe is a multiple of the training record criterion, then retrain
    if len(data.index) % training_record_criterian == 0:
        # print(data)
        #   training in a separate thread
        trainer = threading.Thread(target=train, args=(data,))
        trainer.start()
        trainer.join()
    # Clear predict

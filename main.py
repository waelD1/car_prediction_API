from fastapi import FastAPI, Request, Form, File
from fastapi.templating import Jinja2Templates
import pickle
from fastapi.responses import HTMLResponse
from datetime import datetime, time, timedelta
import pandas as pd
import xgboost

app = FastAPI()

templates = Jinja2Templates(directory="./website/templates")

@app.get("/")
async def read_item(request: Request ):

    # importing the elements of the different variables to create the dropdown lists automaticaly

    with open('model/mapping_model.pkl', 'rb') as f:
        mapping_model_bmw = pickle.load(f)

    with open('model/list_car_type.pkl', 'rb') as f:
        list_car_type = pickle.load(f)

    with open('model/list_fuel.pkl', 'rb') as f:
        list_fuel_type = pickle.load(f)

    with open('model/list_color.pkl', 'rb') as f:
        list_color_option = pickle.load(f)


    list_car_model = list(mapping_model_bmw.keys())

    #Return the lists to have automatic dropdown lists
    return templates.TemplateResponse("base.html", {
        "request": request, 
        "model_key": list_car_model, 
        "car_type" : list_car_type,
        "fuel_type" : list_fuel_type,
        "color_option" : list_color_option
        })


#retrive the data and make prediction
@app.post('/result_pred')
async def prediction( request : Request, 
                    model_key : str = Form(...), 
                    car_type : str = Form(...), 
                    fuel_type : str = Form(...),
                    color_option : str = Form(...),
                    Registration_date : datetime = Form(...),
                    sold_date : datetime = Form(...),
                    Mileage : int = Form(...),
                    Engine_Power : int = Form(...)
                    ):

        with open('model/mapping_model.pkl', 'rb') as f:
            mapping_model_bmw = pickle.load(f)

        #Creating variables
        mileage = Mileage
        engine_power = Engine_Power
        time_on_sale = (sold_date - Registration_date).days
        model_key_cat = mapping_model_bmw[model_key]

        fuel_diesel = 0
        fuel_electro = 0
        fuel_hybrid_petrol = 0
        fuel_petrol = 0

        if fuel_type == 'diesel':
            fuel_diesel = 1

        elif fuel_type == 'electro':
            fuel_electro = 1

        elif fuel_type == 'hybrid_electro':
            fuel_hybrid_petrol = 1
        
        elif fuel_type == 'petrol':
            fuel_petrol = 1


        color_beige = 0
        color_black	= 0
        color_blue = 0
        color_brown	= 0
        color_green	= 0
        color_grey = 0 
        color_orange = 0	
        color_red = 0
        color_silver = 0	
        color_white = 0

        if color_option == 'beige':
            color_beige = 1

        elif color_option == 'black':
            color_black = 1

        elif color_option == 'blue':
            color_blue = 1
        
        elif color_option == 'brown':
            color_brown = 1

        elif color_option == 'green':
            color_green = 1

        elif color_option == 'grey':
            color_grey = 1

        elif color_option == 'orange':
            color_orange = 1
        
        elif color_option == 'red':
            color_red = 1

        elif color_option == 'silver':
            color_silver = 1
        
        elif color_option == 'white':
            color_white = 1
        


        car_type_convertible = 0
        car_type_coupe = 0	
        car_type_estate = 0	
        car_type_hatchback = 0	
        car_type_sedan = 0	
        car_type_subcompact = 0	
        car_type_suv = 0	
        car_type_van = 0

        if car_type == 'convertible':
            car_type_convertible = 1

        elif car_type == 'coupe':
            car_type_coupe = 1

        elif car_type == 'estate':
            car_type_estate = 1
        
        elif car_type == 'hatchback':
            car_type_hatchback = 1

        elif car_type == 'sedan':
            car_type_sedan = 1

        elif car_type == 'subcompact':
            car_type_subcompact = 1

        elif car_type == 'suv':
            car_type_suv = 1
        
        elif car_type == 'van':
            car_type_van = 1

        features = {
            'mileage' : mileage,
            'engine_power' : engine_power, 
            'time_on_sale' : time_on_sale, 
            'model_key_cat' : model_key_cat,
            'fuel_diesel' : fuel_diesel,
            'fuel_electro' : fuel_electro,
            'fuel_hybrid_petrol' : fuel_hybrid_petrol,
            'fuel_petrol' : fuel_petrol,
            'color_beige' : color_beige,
            'color_black' : color_black,
            'color_blue' : color_blue,
            'color_brown' : color_brown,
            'color_green' : color_green,
            'color_grey' : color_grey,
            'color_orange' : color_orange,
            'color_red' : color_red,
            'color_silver' : color_silver,
            'color_white' : color_white,
            'car_type_convertible' : car_type_convertible,
            'car_type_coupe' : car_type_coupe,
            'car_type_estate' : car_type_estate,
            'car_type_hatchback' : car_type_hatchback,
            'car_type_sedan' : car_type_sedan,
            'car_type_subcompact' : car_type_subcompact,
            'car_type_suv' : car_type_suv,
            'car_type_van' : car_type_van
        }

        # Put the data into the right format
        data = pd.DataFrame(features, index=[0])
        data_to_pred = xgboost.DMatrix(data.values)


        # Loading the model
        xgb_model = xgboost.Booster()
        xgb_model.load_model('model/xgb_model.bin')
        
        # Prediction
        prediction = int(xgb_model.predict(data_to_pred))

        return templates.TemplateResponse("result_pred.html", {'request' : request, 'prediction' : prediction})

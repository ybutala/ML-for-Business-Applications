from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, Union, List  # Optional should be imported from typing
from sklearn import preprocessing

# Correctly using the variables in FastAPI initialization
api_title = "Real Estate Unit Price App..."
api_description = """Real Estate Unit Price App for prediction"""
api = FastAPI(title=api_title, description=api_description)

ML_MODEL = joblib.load("./model.joblib")

class HouseFeatures(BaseModel):
    MSSubClass: int
    MSZoning: str
    LotFrontage: Optional[float] = None
    LotArea: int
    Street: str
    Alley: Optional[str] = None
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: Optional[str] = None
    MasVnrArea: Optional[float] = None
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: Optional[str] = None
    BsmtCond: Optional[str] = None
    BsmtExposure: Optional[str] = None
    BsmtFinType1: Optional[str] = None
    BsmtFinSF1: int
    BsmtFinType2: Optional[str] = None
    BsmtFinSF2: int
    BsmtUnfSF: int
    TotalBsmtSF: int
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: Optional[str] = None
    FirstFlrSF: int = Field(..., alias="1stFlrSF")
    SecondFlrSF: int = Field(..., alias="2ndFlrSF")
    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: int
    BsmtHalfBath: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: Optional[str] = None
    GarageType: Optional[str] = None
    GarageYrBlt: Optional[float] = None
    GarageFinish: Optional[str] = None
    GarageCars: int
    GarageArea: int
    GarageQual: Optional[str] = None
    GarageCond: Optional[str] = None
    PavedDrive: str
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    ThreeSsnPorch: int = Field(..., alias="3SsnPorch")
    ScreenPorch: int
    PoolArea: int
    PoolQC: Optional[str] = None
    Fence: Optional[str] = None
    MiscFeature: Optional[str] = None
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: str
    SaleCondition: str

class SalePrice(BaseModel):
    SalePrice: float  # Changed to snake_case

@api.post("/unitprice", response_model=SalePrice)
def sale_price(house_features: HouseFeatures) -> SalePrice:
    features_dict = house_features.model_dump(by_alias=True)
    features_df = pd.DataFrame([features_dict])
    predicted_price = ML_MODEL.predict(features_df)[0]
    return SalePrice(SalePrice=predicted_price)
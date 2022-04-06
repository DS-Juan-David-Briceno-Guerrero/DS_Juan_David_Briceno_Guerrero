import urllib.parse
from urllib.request import urlopen
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import warnings
import numpy as np
#import StringIO
import io
import base64
from io import BytesIO
#from PIL import Image
from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
from flask_wtf import Form, validators  
from wtforms.fields import StringField
from wtforms import TextField, BooleanField, PasswordField, TextAreaField, validators
from wtforms.widgets import TextArea
import xgboost

def load_data():
    data = {'id':['01', '02', '03', '04'], 'name':['Tom', 'nick', 'krish', 'jack']}
    df = pd.DataFrame(data)
    return data, df

def client_name_from_id(id_client, df):
    df_copy = df[df["id"].isin([id_client])]
    name = df_copy["name"].values
    name = name[0]
    return name

def create_points():
    x = [ i for i in range(0,10)]
    names = ['Tom', 'nick', 'krish', 'jack']
    x_vals = []
    y_vals = []
    for i in range(0,len(names)):
        x_vals.append(x)
        y = np.sin(x)*i + i -2
        y_vals.append(y)

    new_dict = {i:[j, k] for i, j, k in zip(names, x_vals, y_vals)}
    return new_dict

def plot_person_points(id_client, df, dict_data, path):
    name = client_name_from_id(id_client, df)
    x = dict_data[name][0]
    y = dict_data[name][1]
    img_name = 'new_plot_'+id_client+'.jpg'
    img = io.BytesIO()
    plt.figure(figsize=(200,270))
    plt.title("hola",fontsize=40)
    plt.plot(x, y, color='black')
    #plt.tight_layout()
    plt.savefig(path+'/'+img_name)
    return img_name
 
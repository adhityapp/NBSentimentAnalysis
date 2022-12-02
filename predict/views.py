from django.shortcuts import render
import pandas as pd
from . import testing
from .count import count
import os
import csv


def index(request):
    dataset = pd.read_csv('data/dataset.csv')
    return render(request, 'index.html', {'data': dataset.iterrows()})


def predict(request):
    if 'load' in request.POST and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        name = myfile.name
        filename = os.path.basename(name).split('.')[0]
        df_test = pd.read_csv(myfile)
        context = {
            'Posts': df_test.iterrows(),
            'name': filename,
            'result_present': True,
        }
        return render(request, 'predict.html', context)
    elif 'predict' in request.POST and request.FILES['myfile']:

        myfile = request.FILES['myfile']
        name = myfile.name
        df_test = pd.read_csv(myfile)
        train = pd.read_pickle(r'train_model.pickle')

        test = testing.predict_dataset(
            df_test,
            train['df_prior'],
            train['df_likelihood']
        )

        pd.to_pickle(test, r'test_model.pickle')

        test_data = pd.read_pickle(r'test_model.pickle')
        df_prediksi = test_data['df_pred']
        df_prediksi = df_prediksi[df_prediksi['pred'] != 0]
        filename = os.path.basename(name).split('.')[0]
        jml = count(df_prediksi['pred'])
        context = {
            'Posts': df_prediksi.iterrows(),
            'name': filename,
            'result_present': True,
            'jml': jml,
        }
        return render(request, 'result.html', context)
    else:
        return render(request, 'predict.html')


def result(request):
    test_data = pd.read_pickle(r'test_model.pickle')
    df_prediksi = test_data['df_pred']
    df_prediksi = df_prediksi[df_prediksi['pred'] != 0]

    jml = count(df_prediksi['pred'])
    context = {
        'Posts': df_prediksi.iterrows(),
        'result_present': True,
        'jml': jml,
    }
    return render(request, 'result.html', context)

    # return render(request, 'result.html')


def about(request):
    return render(request, 'about.html')

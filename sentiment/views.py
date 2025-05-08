from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from django.http import FileResponse, HttpResponseRedirect
from django.urls import reverse
from django.core.files.storage import default_storage
from django.conf import settings
import shutil
import os
import json
import os

from .analyzer import analyze_csv, create_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'src/Sentiment Model.keras')
model = load_model(model_path)

tokenizer_path = os.path.join(BASE_DIR, 'src/tokenizer.json')
with open(tokenizer_path, 'r') as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_data)

# Параметры обработки текста
maxlen = 150

def index(request):
    result = None
    if request.method == 'POST':
        text = request.POST.get('comment')  # Получаем текст от пользователя
        if text:
            tokenized_text = tokenizer.texts_to_sequences([text])

            padded_text = pad_sequences(tokenized_text, maxlen=maxlen)

            prediction = model.predict(padded_text)
            sentiment = {0: 'Нейтральный', 1: 'Позитивный', 2: 'Негативный'}
            result = sentiment[prediction.argmax()]

    return render(request, 'sentiment/index.html', {'result': result})

def clear_media_folder():
    folder = settings.MEDIA_ROOT
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # удалить файл/ссылку
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # удалить папку
        except Exception as e:
            print(f'Ошибка при удалении {file_path}: {e}')

def upload_csv(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        # Очистка папки media перед началом обработки
        clear_media_folder()

        csv_file = request.FILES['csv_file']
        filename = default_storage.save('temp_uploaded.csv', csv_file)
        csv_path = os.path.join(settings.MEDIA_ROOT, filename)

        try:
            analyzed_df = analyze_csv(csv_path, model, tokenizer, maxlen)
        except Exception as e:
            return render(request, 'sentiment/index.html', {
                'error_message': f'Ошибка анализа CSV: {e}'
            })

        analyzed_csv_path = os.path.join(settings.MEDIA_ROOT, 'analyzed_results.csv')
        analyzed_df.to_csv(analyzed_csv_path, index=False)

        report_path = os.path.join(settings.MEDIA_ROOT, 'report.docx')
        create_report(analyzed_df, report_path)

        report_url = settings.MEDIA_URL + 'report.docx'
        return render(request, 'sentiment/index.html', {
            'report_generated': True,
            'report_url': report_url
        })

    return HttpResponseRedirect(reverse('index'))

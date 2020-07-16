import io
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import requests
import pytz, datetime
import time
import os 

from googletrans import Translator
translator = Translator()

def crawl_data_from_RTE():
    required_columns = ['Périmètre', 'Nature', 'Date', 'Heures', 'Consommation',
        'Prévision J-1', 'Prévision J', 'Fioul', 'Charbon', 'Gaz', 'Nucléaire',
        'Eolien', 'Solaire', 'Hydraulique', 'Pompage', 'Bioénergies',
        'Ech. physiques', 'Taux de Co2', 'Ech. comm. Angleterre',
        'Ech. comm. Espagne', 'Ech. comm. Italie', 'Ech. comm. Suisse',
        'Ech. comm. Allemagne-Belgique', 'Fioul - TAC', 'Fioul - Cogén.',
        'Fioul - Autres', 'Gaz - TAC', 'Gaz - Cogén.', 'Gaz - CCG',
        'Gaz - Autres', 'Hydraulique - Fil de l?eau + éclusée',
        'Hydraulique - Lacs', 'Hydraulique - STEP turbinage',
        'Bioénergies - Déchets', 'Bioénergies - Biomasse',
        'Bioénergies - Biogaz']

    url_list = ["https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Annuel-Definitif_2017.zip",
                "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Annuel-Definitif_2018.zip",
                "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-Consolide.zip",
            "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-TR.zip"]


    df_list = []

    for url in url_list:
        content = requests.get(url)
        zf = ZipFile(BytesIO(content.content))

        for item in zf.namelist():
            print("File in zip: "+  item)

        # find the first matching csv file in the zip:
        match = [s for s in zf.namelist() if ".xls" in s][0]
        
        tmp_df = pd.read_table(zf.open(match), index_col=False, usecols = required_columns, encoding='ISO-8859-1').head(-1)
        
        df_list.append(tmp_df)
        
    df = pd.concat(df_list).reset_index(drop=True)

    translated_columns = [translator.translate(i, src='fr', dest='en').text for i in df.columns]
    df.columns = translated_columns

    local = pytz.timezone ("Europe/Paris")

    df['datetime'] = df['Dated'] + " " + df['Hours']

    df['datetime'] = df['datetime'].apply(lambda x: local.localize(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M"), is_dst=True).astimezone(pytz.utc))

    wf_list = ["guitrancourt", "lieusaint", "lvs-pussay", "parc-du-gatinais", "arville", "boissy-la-riviere", "angerville-1", "angerville-2",
    "guitrancourt-b", "lieusaint-b", "lvs-pussay-b", "parc-du-gatinais-b", "arville-b", "boissy-la-riviere-b", "angerville-1-b", "angerville-2-b"]

    forecast_df_list = []

    for forecast in wf_list:
        
        hist_url = "https://ai4impact.org/P003/historical/" + forecast +".csv"

        r = requests.get(hist_url)
        data = r.content.decode('utf8').split("UTC\n")[1]
        hist_tmp_df = pd.read_csv(io.StringIO(data))
        
        current_url = "https://ai4impact.org/P003/" + forecast +".csv"

        r = requests.get(current_url)
        data = r.content.decode('utf8').split("UTC\n")[1]
        current_tmp_df = pd.read_csv(io.StringIO(data))
        
        tmp_df = pd.concat([hist_tmp_df,current_tmp_df]).reset_index(drop=True).rename(columns={'Speed(m/s)':forecast + '_Speed(m/s)', 'Direction (deg N)':forecast + '_Direction (deg N)'})
        
        tmp_df['datetime'] = pd.to_datetime(tmp_df['Time'].str.replace("UTC", ""), utc = True))
        
        tmp_df = tmp_df.drop(columns=['Time'])
        
        forecast_df_list.append(tmp_df)
        
    main_df = df.copy()

    for i in forecast_df_list:
        main_df = main_df.merge(i, how='left', on='datetime')
    
    main_df.to_csv("combined_energy_data.csv", index=False)

    os.system('gsutil cp combined_energy_data.csv gs://ai4impact-hkdragons')

    print("finished crawling and export to gcs")

if __name__=="__main__":

    while True:
        crawl_data_from_RTE()
        for i in range(3600):
            time.sleep(1)